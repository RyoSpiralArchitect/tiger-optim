# ============================================================================
#  Project: SpiralReality / Tiger Optimizer
#  Copyright (c) 2025 Ryo ∴ SpiralArchitect and SpiralReality
#
#  This file is part of SpiralReality.
#
#  SpiralReality is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  SpiralReality is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with SpiralReality.  If not, see <https://www.gnu.org/licenses/>.
# ============================================================================

from __future__ import annotations

import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from time import perf_counter
from types import ModuleType
from typing import Callable, Dict, Iterator, Optional, Tuple

import torch

__all__ = [
    "fast_softsign",
    "fast_rms",
    "fast_norm",
    "available_backends",
    "configure_backends",
    "reset_backend_configuration",
    "refresh_backend_state",
    "backend_diagnostics",
    "current_backend_priority",
]

_BACKEND_MODULES: Dict[str, Optional[ModuleType]] = {}
_BACKEND_ORDER: Tuple[str, ...] = ("julia",)
_CONFIG_LOCK = threading.RLock()
_METRICS_LOCK = threading.RLock()
_PROFILE_LOCK = threading.RLock()
_PREFERENCE_OVERRIDE: Optional[Tuple[str, ...]] = None
_DISABLED_OVERRIDE: Optional[Tuple[str, ...]] = None
_SUPPRESSED_BACKENDS: set[str] = set()
_SUPPRESSION_THRESHOLD = 3
_PROFILED_ATTRS: set[str] = set()


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return ``True`` when ``left`` and ``right`` share the same underlying storage."""

    if left is right:
        return True
    try:  # Prefer the dedicated alias query when available.
        return bool(torch._C._is_alias_of(left, right))  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError, TypeError):
        pass
    for attr in ("untyped_storage", "storage"):
        get_storage = getattr(left, attr, None)
        other_storage = getattr(right, attr, None)
        if get_storage is None or other_storage is None:
            continue
        try:
            if get_storage().data_ptr() == other_storage().data_ptr():  # type: ignore[operator]
                return True
        except (AttributeError, RuntimeError):
            continue
    return False


def _unwrap_backend_tensor(value: object, *, _seen: Optional[set[int]] = None) -> Optional[torch.Tensor]:
    """Best-effort extraction of a tensor payload from ``value`` containers."""

    if isinstance(value, torch.Tensor):
        return value
    if value is None:
        return None
    if _seen is None:
        _seen = set()
    obj_id = id(value)
    if obj_id in _seen:
        return None
    _seen.add(obj_id)

    for attr in ("values", "value"):
        maybe = getattr(value, attr, None)
        if maybe is not None:
            unwrapped = _unwrap_backend_tensor(maybe, _seen=_seen)
            if unwrapped is not None:
                return unwrapped

    if isinstance(value, Mapping):
        priority_keys = ("values", "value", "tensor", "result", "payload", "data")
        for key in priority_keys:
            if key in value:
                unwrapped = _unwrap_backend_tensor(value[key], _seen=_seen)
                if unwrapped is not None:
                    return unwrapped
        for item in value.values():
            unwrapped = _unwrap_backend_tensor(item, _seen=_seen)
            if unwrapped is not None:
                return unwrapped

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        try:
            candidates = [value[index] for index in range(min(len(value), 8))]
        except TypeError:
            candidates = []
        except IndexError:
            candidates = []
        if not candidates:
            iterator = iter(value)
            for _ in range(8):
                try:
                    candidates.append(next(iterator))
                except StopIteration:
                    break
        for candidate in candidates:
            unwrapped = _unwrap_backend_tensor(candidate, _seen=_seen)
            if unwrapped is not None:
                return unwrapped

    return None


def _coerce_backend_value(reference: torch.Tensor, value: object) -> torch.Tensor:
    """Return ``value`` as a tensor matching ``reference``'s dtype and device."""

    tensor = _unwrap_backend_tensor(value)
    if tensor is None:
        try:
            tensor = torch.as_tensor(value, dtype=reference.dtype)
        except (TypeError, ValueError):
            tensor = reference.new_tensor(value)
        else:
            if tensor.device != reference.device:
                tensor = tensor.to(device=reference.device)
            if tensor.dtype != reference.dtype:
                tensor = tensor.to(dtype=reference.dtype)

    if tensor.device != reference.device or tensor.dtype != reference.dtype:
        tensor = tensor.to(dtype=reference.dtype, device=reference.device)
    if _shares_storage(reference, tensor):
        tensor = tensor.clone()
    return tensor


@dataclass
class _BackendMetrics:
    name: str
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_time: float = 0.0
    ema_us: Optional[float] = None
    last_error: Optional[str] = None

    def record_success(self, duration: float) -> None:
        self.calls += 1
        self.successes += 1
        self.total_time += duration
        sample = duration * 1e6
        if self.ema_us is None:
            self.ema_us = sample
        else:
            self.ema_us = 0.8 * self.ema_us + 0.2 * sample
        self.last_error = None

    def record_failure(self, error: Exception | str) -> None:
        self.calls += 1
        self.failures += 1
        self.last_error = str(error)

    @property
    def average_us(self) -> Optional[float]:
        if self.successes == 0:
            return None
        return (self.total_time / self.successes) * 1e6


_BACKEND_METRICS: Dict[str, _BackendMetrics] = {}


def _normalize_backend_name(name: str) -> str:
    name = name.strip().lower()
    if name.endswith("_backend"):
        name = name[: -len("_backend")]
    return name


def _load_backend(name: str) -> Optional[ModuleType]:
    try:
        return import_module(f".{name}_backend", __name__)
    except Exception:
        return None


def _ensure_metrics(names: Optional[Sequence[str]] = None) -> None:
    target = _BACKEND_MODULES.keys() if names is None else names
    with _METRICS_LOCK:
        for item in target:
            if item not in _BACKEND_METRICS:
                _BACKEND_METRICS[item] = _BackendMetrics(name=item)


def _needs_profiling(attr: str) -> bool:
    with _PROFILE_LOCK:
        return attr not in _PROFILED_ATTRS


def _mark_profiled(attr: str) -> None:
    with _PROFILE_LOCK:
        _PROFILED_ATTRS.add(attr)


for _candidate in _BACKEND_ORDER:
    _BACKEND_MODULES[_candidate] = _load_backend(_candidate)

_ensure_metrics()


def configure_backends(
    *, preferred: Optional[Sequence[str]] = None, disabled: Optional[Sequence[str]] = None
) -> None:
    """Configure backend ordering/disablement at runtime.

    ``preferred`` controls the iteration order for backends (front of list is tried first).
    ``disabled`` prevents the listed backends from being queried. Values may include the
    ``"*_backend"`` suffix; matching is case-insensitive.
    """

    global _PREFERENCE_OVERRIDE, _DISABLED_OVERRIDE
    with _CONFIG_LOCK:
        if preferred is not None:
            normalized = tuple(
                _normalize_backend_name(item)
                for item in preferred
                if _normalize_backend_name(item) in _BACKEND_MODULES
            )
            _PREFERENCE_OVERRIDE = normalized
        if disabled is not None:
            if any(_normalize_backend_name(item) in {"all", "*"} for item in disabled):
                _DISABLED_OVERRIDE = tuple(_BACKEND_MODULES.keys())
            else:
                normalized = tuple(
                    _normalize_backend_name(item)
                    for item in disabled
                    if _normalize_backend_name(item) in _BACKEND_MODULES
                )
                _DISABLED_OVERRIDE = normalized
    _is_available.cache_clear()


def reset_backend_configuration() -> None:
    """Clear any runtime backend overrides applied via :func:`configure_backends`."""

    global _PREFERENCE_OVERRIDE, _DISABLED_OVERRIDE
    with _CONFIG_LOCK:
        _PREFERENCE_OVERRIDE = None
        _DISABLED_OVERRIDE = None
    _is_available.cache_clear()


def refresh_backend_state(*, reload: bool = False, reset_metrics: bool = False) -> None:
    """Refresh cached backend state and optionally reload backend modules.

    Parameters
    ----------
    reload:
        If ``True`` the optional backend modules are re-imported, allowing newly built
        shared libraries to be picked up without restarting the interpreter.
    reset_metrics:
        When ``True`` the runtime diagnostics and suppression counters are cleared so
        every backend can be re-evaluated from a clean slate.
    """

    if reload:
        with _CONFIG_LOCK:
            for name in _BACKEND_ORDER:
                _BACKEND_MODULES[name] = _load_backend(name)
    with _METRICS_LOCK:
        if reset_metrics or reload:
            _BACKEND_METRICS.clear()
        _SUPPRESSED_BACKENDS.clear()
    with _PROFILE_LOCK:
        _PROFILED_ATTRS.clear()
    _ensure_metrics()
    _env_preferred.cache_clear()
    _env_disabled.cache_clear()
    _is_available.cache_clear()


def _env_list(name: str) -> Tuple[str, ...]:
    raw = os.getenv(name, "")
    if not raw:
        return ()
    return tuple(
        token.strip().lower()
        for token in raw.split(",")
        if token.strip()
    )


@lru_cache(maxsize=None)
def _env_preferred() -> Tuple[str, ...]:
    return tuple(
        _normalize_backend_name(token)
        for token in _env_list("TIGER_ACCEL_PREFER")
        if _normalize_backend_name(token) in _BACKEND_MODULES
    )


@lru_cache(maxsize=None)
def _env_disabled() -> Tuple[str, ...]:
    tokens = _env_list("TIGER_ACCEL_DISABLE")
    if any(token in {"1", "true", "yes", "all", "*"} for token in tokens):
        return tuple(_BACKEND_MODULES.keys())
    return tuple(
        _normalize_backend_name(token)
        for token in tokens
        if _normalize_backend_name(token) in _BACKEND_MODULES
    )


def _current_preferred() -> Tuple[str, ...]:
    with _CONFIG_LOCK:
        if _PREFERENCE_OVERRIDE is not None:
            return _PREFERENCE_OVERRIDE
    return _env_preferred()


def _current_disabled() -> Tuple[str, ...]:
    with _CONFIG_LOCK:
        if _DISABLED_OVERRIDE is not None:
            return _DISABLED_OVERRIDE
    return _env_disabled()


def _current_suppressed() -> Tuple[str, ...]:
    with _METRICS_LOCK:
        if not _SUPPRESSED_BACKENDS:
            return ()
        return tuple(sorted(_SUPPRESSED_BACKENDS))


def _dynamic_backend_order() -> Tuple[str, ...]:
    with _METRICS_LOCK:
        ranked = [
            (
                metrics.ema_us if metrics.ema_us is not None else metrics.average_us,
                name,
            )
            for name, metrics in _BACKEND_METRICS.items()
            if metrics.successes > 0
        ]
    ranked.sort(key=lambda item: item[0])
    return tuple(name for _, name in ranked)


def _ordered_backend_names() -> Tuple[str, ...]:
    disabled = set(_current_disabled()) | set(_current_suppressed())
    preferred = _current_preferred()
    dynamic = _dynamic_backend_order()
    ordered = []
    seen = set()

    def _append(name: str) -> None:
        if name in seen or name in disabled:
            return
        seen.add(name)
        ordered.append(name)

    for name in preferred:
        _append(name)
    for name in dynamic:
        _append(name)
    for name in _BACKEND_ORDER:
        _append(name)
    return tuple(ordered)


@lru_cache(maxsize=None)
def _is_available(name: str) -> bool:
    with _METRICS_LOCK:
        if name in _SUPPRESSED_BACKENDS:
            return False
    _ensure_metrics((name,))
    module = _BACKEND_MODULES.get(name)
    if module is None:
        return False
    check = getattr(module, "is_available", None)
    if check is None:
        return False
    try:
        return bool(check())
    except Exception:
        return False


def _get_metrics(name: str) -> _BackendMetrics:
    _ensure_metrics((name,))
    with _METRICS_LOCK:
        metrics = _BACKEND_METRICS.get(name)
        if metrics is None:
            metrics = _BackendMetrics(name=name)
            _BACKEND_METRICS[name] = metrics
        return metrics


def _handle_failure(name: str, metrics: _BackendMetrics, error: Exception | str) -> None:
    suppress = False
    with _METRICS_LOCK:
        metrics.record_failure(error)
        if metrics.successes == 0 and metrics.failures >= _SUPPRESSION_THRESHOLD:
            if name not in _SUPPRESSED_BACKENDS:
                _SUPPRESSED_BACKENDS.add(name)
                suppress = True
    if suppress:
        _is_available.cache_clear()


def _record_success(name: str, metrics: _BackendMetrics, duration: float) -> None:
    with _METRICS_LOCK:
        metrics.record_success(duration)


def _iter_backends() -> Iterator[Tuple[str, ModuleType]]:
    for name in _ordered_backend_names():
        module = _BACKEND_MODULES.get(name)
        if module is None:
            continue
        if not _is_available(name):
            continue
        yield name, module


def _dispatch(
    name: str,
    module: ModuleType,
    attr: str,
    validator: Callable[[object], object],
    *args,
) -> Optional[object]:
    func = getattr(module, attr, None)
    if func is None:
        return None
    metrics = _get_metrics(name)
    start = perf_counter()
    try:
        result = func(*args)
    except Exception as exc:  # pragma: no cover - exercised indirectly
        _handle_failure(name, metrics, exc)
        return None
    duration = perf_counter() - start
    if result is None:
        _handle_failure(name, metrics, "returned None")
        return None
    try:
        validated = validator(result)
    except Exception as exc:
        _handle_failure(name, metrics, exc)
        return None
    _record_success(name, metrics, duration)
    return validated


def _profile_backends(attr: str, validator: Callable[[object], object], *args) -> Optional[object]:
    available = list(_iter_backends())
    if len(available) <= 1:
        _mark_profiled(attr)
        return None

    results: Dict[str, object] = {}
    best_name: Optional[str] = None
    best_score = float("inf")

    for name, module in available:
        result = _dispatch(name, module, attr, validator, *args)
        if result is None:
            continue
        results[name] = result
        metrics = _get_metrics(name)
        with _METRICS_LOCK:
            sample = metrics.ema_us if metrics.ema_us is not None else metrics.average_us
        score = float(sample) if sample is not None else float("inf")
        if best_name is None or score < best_score:
            best_name = name
            best_score = score

    _mark_profiled(attr)
    if best_name is None:
        return None
    return results.get(best_name)


def _accelerated_result(
    attr: str, validator: Callable[[object], object], *args
) -> Optional[object]:
    if _needs_profiling(attr):
        profiled = _profile_backends(attr, validator, *args)
        if profiled is not None:
            return profiled
    for name, module in _iter_backends():
        result = _dispatch(name, module, attr, validator, *args)
        if result is not None:
            return result
    return None


def fast_softsign(x: torch.Tensor, tau: float) -> torch.Tensor:
    """Return a softsign-transformed copy of ``x`` leveraging native accelerators when possible."""

    if x.numel() == 0:
        return x.clone()
    if x.requires_grad:
        return x / (x.abs() + tau)
    if x.device.type == "cpu":
        def _validate(result: object) -> torch.Tensor:
            tensor = _coerce_backend_value(x, result)
            if tensor.shape != x.shape:
                raise ValueError(
                    "softsign backend must return a tensor matching the input shape"
                )
            return tensor

        result = _accelerated_result("softsign", _validate, x, tau)
        if result is not None:
            return result
    return x / (x.abs() + tau)


def fast_rms(x: torch.Tensor) -> torch.Tensor:
    """Compute RMS using optional acceleration backends when available."""

    if x.numel() == 0:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    if x.requires_grad:
        return x.pow(2).mean().sqrt()
    if x.device.type == "cpu":
        def _validate(result: object) -> torch.Tensor:
            tensor = _coerce_backend_value(x, result)
            if tensor.dim() != 0:
                raise ValueError("rms backend must return a scalar tensor")
            return tensor

        result = _accelerated_result("rms", _validate, x)
        if result is not None:
            return result
    return x.pow(2).mean().sqrt()


def fast_norm(x: torch.Tensor) -> torch.Tensor:
    """Compute the vector norm with optional accelerator support."""

    if x.numel() == 0:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    if x.requires_grad:
        return torch.linalg.vector_norm(x)
    if x.device.type == "cpu":
        def _validate(result: object) -> torch.Tensor:
            tensor = _coerce_backend_value(x, result)
            if tensor.dim() != 0:
                raise ValueError("norm backend must return a scalar tensor")
            return tensor

        result = _accelerated_result("norm", _validate, x)
        if result is not None:
            return result
    return torch.linalg.vector_norm(x)


def current_backend_priority() -> Tuple[str, ...]:
    """Return the runtime-resolved backend ordering."""

    return _ordered_backend_names()


def backend_diagnostics() -> Dict[str, Dict[str, object]]:
    """Return runtime metrics describing backend health and performance."""

    snapshot: Dict[str, Dict[str, object]] = {}
    with _METRICS_LOCK:
        suppressed = set(_SUPPRESSED_BACKENDS)
        for name in _BACKEND_MODULES:
            metrics = _BACKEND_METRICS.get(name)
            if metrics is None:
                metrics = _BackendMetrics(name=name)
                _BACKEND_METRICS[name] = metrics
            snapshot[name] = {
                "calls": metrics.calls,
                "successes": metrics.successes,
                "failures": metrics.failures,
                "average_us": metrics.average_us,
                "ema_us": metrics.ema_us,
                "suppressed": name in suppressed,
                "last_error": metrics.last_error,
            }
    return snapshot


def available_backends() -> Dict[str, bool]:
    """Return a dictionary describing which high-performance backends are usable."""

    return {name: _is_available(name) for name in _BACKEND_MODULES}
