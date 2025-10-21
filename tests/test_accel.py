import importlib
import sys
import time
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _reload_accel(monkeypatch, **env):
    for key in ("TIGER_ACCEL_DISABLE", "TIGER_ACCEL_PREFER"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    for name in list(sys.modules):
        if name == "tiger_optim.accel" or name.startswith("tiger_optim.accel."):
            sys.modules.pop(name)
    return importlib.import_module("tiger_optim.accel")


def test_fast_paths_match_eager_when_disabled(monkeypatch):
    accel = _reload_accel(monkeypatch, TIGER_ACCEL_DISABLE="all")
    x = torch.randn(8, dtype=torch.float32)
    tau = 1e-3

    expected_softsign = x / (x.abs() + tau)
    expected_rms = x.pow(2).mean().sqrt()
    expected_norm = torch.linalg.vector_norm(x)

    softsign = accel.fast_softsign(x, tau)
    rms = accel.fast_rms(x)
    norm = accel.fast_norm(x)

    assert torch.allclose(softsign, expected_softsign)
    assert torch.allclose(rms, expected_rms)
    assert torch.allclose(norm, expected_norm)


def test_configure_backends_runtime_disable(monkeypatch):
    accel = _reload_accel(monkeypatch)
    try:
        accel.configure_backends(disabled=["all"])
        x = torch.randn(4)
        tau = 0.5
        expected = x / (x.abs() + tau)
        actual = accel.fast_softsign(x, tau)
        assert torch.allclose(actual, expected)
    finally:
        accel.reset_backend_configuration()


def test_available_backends_shape(monkeypatch):
    accel = _reload_accel(monkeypatch, TIGER_ACCEL_DISABLE="all")
    status = accel.available_backends()
    assert set(status.keys()) == {"rust", "julia", "go"}
    assert all(isinstance(flag, bool) for flag in status.values())


def test_backend_failure_suppression(monkeypatch):
    accel = _reload_accel(monkeypatch)

    accel.configure_backends(preferred=["rust"], disabled=["julia", "go"])

    class FailingBackend:
        def __init__(self) -> None:
            self.calls = 0

        @staticmethod
        def is_available() -> bool:
            return True

        def rms(self, _: torch.Tensor) -> float:
            self.calls += 1
            raise RuntimeError("boom")

    failing = FailingBackend()
    try:
        monkeypatch.setitem(accel._BACKEND_MODULES, "rust", failing)
        accel.refresh_backend_state(reset_metrics=True)

        x = torch.randn(8)
        for _ in range(4):
            accel.fast_rms(x)

        diagnostics = accel.backend_diagnostics()
        assert diagnostics["rust"]["failures"] >= 3
        assert diagnostics["rust"]["suppressed"] is True

        expected = x.pow(2).mean().sqrt()
        actual = accel.fast_rms(x)
        assert torch.allclose(actual, expected)
    finally:
        accel.reset_backend_configuration()


def test_backend_priority_reorders_by_latency(monkeypatch):
    accel = _reload_accel(monkeypatch)
    accel.configure_backends(disabled=["go"])

    class FakeBackend:
        def __init__(self, delay: float) -> None:
            self.delay = delay
            self._cache: Optional[float] = None

        @staticmethod
        def is_available() -> bool:
            return True

        def norm(self, x: torch.Tensor) -> float:
            if self._cache is None:
                self._cache = float(torch.linalg.vector_norm(x).item())
            if self.delay:
                time.sleep(self.delay)
            return self._cache

    slow = FakeBackend(delay=0.01)
    fast = FakeBackend(delay=0.0)
    try:
        monkeypatch.setitem(accel._BACKEND_MODULES, "rust", slow)
        monkeypatch.setitem(accel._BACKEND_MODULES, "julia", fast)
        accel.refresh_backend_state(reset_metrics=True)

        x = torch.randn(128)
        for _ in range(5):
            accel.fast_norm(x)

        order = accel.current_backend_priority()
        assert order and order[0] == "julia"

        diagnostics = accel.backend_diagnostics()
        assert diagnostics["rust"]["successes"] > 0
        assert diagnostics["julia"]["successes"] > 0
    finally:
        accel.reset_backend_configuration()


def test_tensor_backend_results_are_normalized(monkeypatch):
    accel = _reload_accel(monkeypatch)

    class TensorBackend:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def softsign(x: torch.Tensor, tau: float) -> torch.Tensor:
            return torch.full_like(x, 0.5, dtype=torch.float64)

        @staticmethod
        def rms(x: torch.Tensor) -> torch.Tensor:
            return torch.tensor(3.0, dtype=torch.float64)

        @staticmethod
        def norm(x: torch.Tensor) -> torch.Tensor:
            return torch.tensor(4.0, dtype=torch.float64)

    backend = TensorBackend()
    try:
        accel.configure_backends(disabled=["julia", "go"])
        monkeypatch.setitem(accel._BACKEND_MODULES, "rust", backend)
        monkeypatch.setitem(accel._BACKEND_MODULES, "julia", None)
        monkeypatch.setitem(accel._BACKEND_MODULES, "go", None)
        accel.refresh_backend_state(reset_metrics=True)

        x = torch.randn(7, dtype=torch.float16)
        softsign = accel.fast_softsign(x, tau=0.3)
        assert softsign.dtype == x.dtype
        assert softsign.device == x.device
        assert torch.allclose(softsign, torch.full_like(x, 0.5))

        rms = accel.fast_rms(x)
        assert rms.dtype == x.dtype
        assert rms.device == x.device
        assert torch.allclose(rms, x.new_tensor(3.0))

        norm = accel.fast_norm(x)
        assert norm.dtype == x.dtype
        assert norm.device == x.device
        assert torch.allclose(norm, x.new_tensor(4.0))
    finally:
        accel.reset_backend_configuration()
        accel.refresh_backend_state(reload=True, reset_metrics=True)


def test_fast_paths_skip_accel_when_requires_grad(monkeypatch):
    accel = _reload_accel(monkeypatch)

    class TrackingBackend:
        def __init__(self) -> None:
            self.softsign_calls = 0
            self.rms_calls = 0
            self.norm_calls = 0

        @staticmethod
        def is_available() -> bool:
            return True

        def softsign(self, x: torch.Tensor, tau: float) -> torch.Tensor:
            self.softsign_calls += 1
            return torch.full_like(x, 0.0)

        def rms(self, x: torch.Tensor) -> torch.Tensor:
            self.rms_calls += 1
            return torch.tensor(0.0)

        def norm(self, x: torch.Tensor) -> torch.Tensor:
            self.norm_calls += 1
            return torch.tensor(0.0)

    backend = TrackingBackend()
    try:
        accel.configure_backends(disabled=["julia", "go"])
        monkeypatch.setitem(accel._BACKEND_MODULES, "rust", backend)
        monkeypatch.setitem(accel._BACKEND_MODULES, "julia", None)
        monkeypatch.setitem(accel._BACKEND_MODULES, "go", None)
        accel.refresh_backend_state(reset_metrics=True)

        x = torch.randn(5, requires_grad=True)
        tau = 0.25

        softsign = accel.fast_softsign(x, tau)
        assert backend.softsign_calls == 0
        assert softsign.requires_grad
        expected_softsign = x / (x.abs() + tau)
        assert torch.allclose(softsign, expected_softsign)

        rms = accel.fast_rms(x)
        assert backend.rms_calls == 0
        assert rms.requires_grad
        expected_rms = x.pow(2).mean().sqrt()
        assert torch.allclose(rms, expected_rms)

        norm = accel.fast_norm(x)
        assert backend.norm_calls == 0
        assert norm.requires_grad
        expected_norm = torch.linalg.vector_norm(x)
        assert torch.allclose(norm, expected_norm)
    finally:
        accel.reset_backend_configuration()
        accel.refresh_backend_state(reload=True, reset_metrics=True)


def test_accelerated_results_preserve_requires_grad(monkeypatch):
    accel = _reload_accel(monkeypatch)

    class GradBackend:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def softsign(x: torch.Tensor, tau: float) -> torch.Tensor:
            return torch.full_like(x, 0.25, requires_grad=True)

        @staticmethod
        def rms(x: torch.Tensor) -> torch.Tensor:
            value = x.new_tensor(1.5)
            value.requires_grad_()
            return value

        @staticmethod
        def norm(x: torch.Tensor) -> torch.Tensor:
            value = x.new_tensor(2.5)
            value.requires_grad_()
            return value

    backend = GradBackend()
    try:
        accel.configure_backends(disabled=["julia", "go"])
        monkeypatch.setitem(accel._BACKEND_MODULES, "rust", backend)
        monkeypatch.setitem(accel._BACKEND_MODULES, "julia", None)
        monkeypatch.setitem(accel._BACKEND_MODULES, "go", None)
        accel.refresh_backend_state(reset_metrics=True)

        x = torch.randn(6, dtype=torch.float32)
        tau = 0.1

        softsign = accel.fast_softsign(x, tau)
        assert softsign.requires_grad

        rms = accel.fast_rms(x)
        assert rms.requires_grad

        norm = accel.fast_norm(x)
        assert norm.requires_grad
    finally:
        accel.reset_backend_configuration()
        accel.refresh_backend_state(reload=True, reset_metrics=True)


def test_softsign_backend_shape_mismatch_falls_back(monkeypatch):
    accel = _reload_accel(monkeypatch)

    class BadSoftsignBackend:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def softsign(x: torch.Tensor, tau: float) -> torch.Tensor:
            return torch.ones(x.shape[:-1], dtype=x.dtype)

    backend = BadSoftsignBackend()
    try:
        accel.configure_backends(disabled=["julia", "go"])
        monkeypatch.setitem(accel._BACKEND_MODULES, "rust", backend)
        monkeypatch.setitem(accel._BACKEND_MODULES, "julia", None)
        monkeypatch.setitem(accel._BACKEND_MODULES, "go", None)
        accel.refresh_backend_state(reset_metrics=True)

        x = torch.randn(6)
        tau = 0.75
        expected = x / (x.abs() + tau)

        actual = accel.fast_softsign(x, tau)
        assert torch.allclose(actual, expected)

        diagnostics = accel.backend_diagnostics()
        assert diagnostics["rust"]["failures"] >= 1
    finally:
        accel.reset_backend_configuration()
        accel.refresh_backend_state(reload=True, reset_metrics=True)


def test_scalar_backends_validate_shape(monkeypatch):
    accel = _reload_accel(monkeypatch)

    class BadScalarBackend:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def rms(x: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(x)

        @staticmethod
        def norm(x: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(x)

    backend = BadScalarBackend()
    try:
        accel.configure_backends(disabled=["julia", "go"])
        monkeypatch.setitem(accel._BACKEND_MODULES, "rust", backend)
        monkeypatch.setitem(accel._BACKEND_MODULES, "julia", None)
        monkeypatch.setitem(accel._BACKEND_MODULES, "go", None)
        accel.refresh_backend_state(reset_metrics=True)

        x = torch.randn(10)
        expected_rms = x.pow(2).mean().sqrt()
        expected_norm = torch.linalg.vector_norm(x)

        actual_rms = accel.fast_rms(x)
        actual_norm = accel.fast_norm(x)

        assert torch.allclose(actual_rms, expected_rms)
        assert torch.allclose(actual_norm, expected_norm)

        diagnostics = accel.backend_diagnostics()
        assert diagnostics["rust"]["failures"] >= 2
    finally:
        accel.reset_backend_configuration()
        accel.refresh_backend_state(reload=True, reset_metrics=True)
