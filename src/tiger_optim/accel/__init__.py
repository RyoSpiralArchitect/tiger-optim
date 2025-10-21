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

from . import julia_backend, rust_backend

__all__ = [
    "fast_softsign",
    "fast_rms",
    "fast_norm",
    "available_backends",
]


def fast_softsign(x: torch.Tensor, tau: float) -> torch.Tensor:
    """Return a softsign-transformed copy of ``x`` leveraging native accelerators when possible."""
    if x.numel() == 0:
        return x.clone()
    if x.device.type == "cpu":
        out = rust_backend.softsign(x, tau)
        if out is not None:
            return out
        out = julia_backend.softsign(x, tau)
        if out is not None:
            return out
    # Fallback: PyTorch eager path
    return x / (x.abs() + tau)


def fast_rms(x: torch.Tensor) -> torch.Tensor:
    """Compute RMS using Rust/Julia acceleration when available."""
    if x.numel() == 0:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    if x.device.type == "cpu":
        val = rust_backend.rms(x)
        if val is not None:
            return torch.tensor(val, dtype=x.dtype, device=x.device)
        val = julia_backend.rms(x)
        if val is not None:
            return torch.tensor(val, dtype=x.dtype, device=x.device)
    return x.pow(2).mean().sqrt()


def fast_norm(x: torch.Tensor) -> torch.Tensor:
    """Compute the vector norm with optional accelerator support."""
    if x.numel() == 0:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    if x.device.type == "cpu":
        val = rust_backend.norm(x)
        if val is not None:
            return torch.tensor(val, dtype=x.dtype, device=x.device)
        val = julia_backend.norm(x)
        if val is not None:
            return torch.tensor(val, dtype=x.dtype, device=x.device)
    return torch.linalg.vector_norm(x)


def available_backends() -> dict:
    """Return a dictionary describing which high-performance backends are usable."""
    return {
        "rust": rust_backend.is_available(),
        "julia": julia_backend.is_available(),
    }
