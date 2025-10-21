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
