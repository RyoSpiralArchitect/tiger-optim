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

from typing import Optional

import numpy as np
import torch

try:  # pragma: no cover - optional dependency
    from juliacall import Main as jl
except Exception:  # pragma: no cover - Julia is optional
    jl = None

__all__ = ["softsign", "rms", "norm", "is_available"]

if jl is not None:
    jl.seval(
        """
        module TigerAccel
        export softsign32!, softsign64!, rms32, rms64, norm32, norm64

        function softsign32!(dest::Array{Float32}, src::Array{Float32}, tau::Float32)
            @inbounds @simd for i in eachindex(dest)
                dest[i] = src[i] / (abs(src[i]) + tau)
            end
            return dest
        end

        function softsign64!(dest::Array{Float64}, src::Array{Float64}, tau::Float64)
            @inbounds @simd for i in eachindex(dest)
                dest[i] = src[i] / (abs(src[i]) + tau)
            end
            return dest
        end

        rms32(src::Array{Float32}) = sqrt(mean(abs2, src))
        rms64(src::Array{Float64}) = sqrt(mean(abs2, src))

        norm32(src::Array{Float32}) = sqrt(sum(abs2, src))
        norm64(src::Array{Float64}) = sqrt(sum(abs2, src))
        end
        """
    )
    _SOFTSIGN32 = jl.eval("TigerAccel.softsign32!")
    _SOFTSIGN64 = jl.eval("TigerAccel.softsign64!")
    _RMS32 = jl.eval("TigerAccel.rms32")
    _RMS64 = jl.eval("TigerAccel.rms64")
    _NORM32 = jl.eval("TigerAccel.norm32")
    _NORM64 = jl.eval("TigerAccel.norm64")
else:
    _SOFTSIGN32 = _SOFTSIGN64 = _RMS32 = _RMS64 = _NORM32 = _NORM64 = None


def _numpy_view(x: torch.Tensor) -> Optional[np.ndarray]:
    if x.device.type != "cpu":
        return None
    if not x.is_contiguous():
        x = x.contiguous()
    try:
        return x.detach().numpy()
    except Exception:
        return None


def softsign(x: torch.Tensor, tau: float) -> Optional[torch.Tensor]:
    if jl is None:
        return None
    arr = _numpy_view(x)
    if arr is None:
        return None
    if arr.dtype == np.float32 and _SOFTSIGN32 is not None:
        dest = np.empty_like(arr)
        _SOFTSIGN32(dest, arr, np.float32(tau))
        return torch.from_numpy(dest.copy())
    if arr.dtype == np.float64 and _SOFTSIGN64 is not None:
        dest = np.empty_like(arr)
        _SOFTSIGN64(dest, arr, np.float64(tau))
        return torch.from_numpy(dest.copy())
    return None


def rms(x: torch.Tensor) -> Optional[float]:
    if jl is None:
        return None
    arr = _numpy_view(x)
    if arr is None:
        return None
    if arr.dtype == np.float32 and _RMS32 is not None:
        return float(_RMS32(arr))
    if arr.dtype == np.float64 and _RMS64 is not None:
        return float(_RMS64(arr))
    return None


def norm(x: torch.Tensor) -> Optional[float]:
    if jl is None:
        return None
    arr = _numpy_view(x)
    if arr is None:
        return None
    if arr.dtype == np.float32 and _NORM32 is not None:
        return float(_NORM32(arr))
    if arr.dtype == np.float64 and _NORM64 is not None:
        return float(_NORM64(arr))
    return None


def is_available() -> bool:
    return jl is not None
