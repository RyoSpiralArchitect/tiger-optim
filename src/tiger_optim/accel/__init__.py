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

"""TorchScript accelerated primitives for Tiger."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

__all__ = [
    "fast_softsign",
    "fast_rms",
    "fast_norm",
    "available_backends",
]


def _script_or_none(fn: Callable) -> Optional[Callable]:
    try:  # pragma: no cover - TorchScript not always available in tests
        return torch.jit.script(fn)  # type: ignore[misc]
    except Exception:  # pragma: no cover - TorchScript compile can fail
        return None


def _softsign_eager(x: torch.Tensor, tau: float) -> torch.Tensor:
    return x / (x.abs() + tau)


def _rms_eager(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x * x))


def _norm_eager(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x)


# Cache optional scripted variants that fuse scalar parameters in the graph.
_softsign_script = _script_or_none(_softsign_eager)
_rms_script = _script_or_none(_rms_eager)
_norm_script = _script_or_none(_norm_eager)


def fast_softsign(x: torch.Tensor, tau: float) -> torch.Tensor:
    """Return the softsign transform with optional TorchScript acceleration."""
    if _softsign_script is not None and x.is_contiguous():
        return _softsign_script(x, float(tau))
    return _softsign_eager(x, float(tau))


def fast_rms(x: torch.Tensor) -> torch.Tensor:
    """Return the root-mean-square of *x* using a scripted kernel when possible."""
    if x.numel() == 0:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    if _rms_script is not None and x.is_contiguous():
        return _rms_script(x)
    return _rms_eager(x)


def fast_norm(x: torch.Tensor) -> torch.Tensor:
    """Return the L2 norm of *x* using a scripted kernel when possible."""
    if x.numel() == 0:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    if _norm_script is not None and x.is_contiguous():
        return _norm_script(x)
    return _norm_eager(x)


def available_backends() -> Dict[str, bool]:
    """Report whether TorchScript kernels are available for each primitive."""
    return {
        "torchscript_softsign": _softsign_script is not None,
        "torchscript_rms": _rms_script is not None,
        "torchscript_norm": _norm_script is not None,
    }
