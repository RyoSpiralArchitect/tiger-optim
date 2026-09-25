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


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.square().sum().div(float(x.numel())).sqrt()


def norm(x: torch.Tensor) -> torch.Tensor:
    return x.mul(x).sum().sqrt()
