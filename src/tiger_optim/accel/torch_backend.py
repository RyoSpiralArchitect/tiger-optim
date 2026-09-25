# ============================================================================
#  Project: SpiralReality / Tiger Optimizer
#  Copyright (c) 2025 Ryo ∴ SpiralArchitect and SpiralReality
#
#  This file is part of SpiralReality.
#
#  SpiralReality is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, version 3 of the License.
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

import math

import torch

__all__ = ["softsign", "rms", "norm", "is_available", "supports"]


def is_available() -> bool:
    """The pure-Torch backend is always available when PyTorch imports."""

    return True


def supports(attr: str, *args) -> bool:
    """Accept tensor workloads on CPU, CUDA, and MPS devices."""

    if not args or not isinstance(args[0], torch.Tensor):
        return False
    x = args[0]
    return x.device.type in {"cpu", "cuda", "mps"}


def softsign(x: torch.Tensor, tau: float) -> torch.Tensor:
    return x / (x.abs() + tau)


def _scaled_l2(x: torch.Tensor, *, rms: bool = False) -> torch.Tensor:
    """Reduce after scaling by the largest element to avoid square overflow."""

    if not x.is_floating_point() and not x.is_complex():
        x = x.to(torch.float64 if x.device.type == "cpu" and x.dtype == torch.int64 else torch.float32)
    if not x.numel():
        return (x.real if x.is_complex() else x).new_zeros(())
    if x.device.type == "cpu" and x.dtype in (torch.float16, torch.bfloat16, torch.float32):
        # FP64 accumulation is faster than a second reduction on CPU and can
        # square every finite value representable in these input dtypes.
        length = torch.linalg.vector_norm(x, dtype=torch.float64)
        if rms:
            length = length / math.sqrt(x.numel())
        return length.to(x.dtype)
    magnitude = x.abs()
    scale = magnitude.amax()
    safe_scale = torch.where(
        torch.isfinite(scale) & (scale > 0), scale, torch.ones_like(scale)
    )
    work_dtype = (
        torch.float32
        if magnitude.dtype in (torch.float16, torch.bfloat16)
        else magnitude.dtype
    )
    normalized = magnitude.to(work_dtype) / safe_scale.to(work_dtype)
    length = torch.linalg.vector_norm(normalized)
    if rms:
        length = length / math.sqrt(x.numel())
    return (safe_scale.to(work_dtype) * length).to(magnitude.dtype)


def rms(x: torch.Tensor) -> torch.Tensor:
    return _scaled_l2(x, rms=True)


def norm(x: torch.Tensor) -> torch.Tensor:
    return _scaled_l2(x)
