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

import ctypes
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import torch

__all__ = ["softsign", "rms", "norm", "is_available"]

_LOCK = threading.Lock()
_LIB: Optional[ctypes.CDLL] = None
_LOAD_FAILED = False
_FUNCS_BOUND = False


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def _crate_dir() -> Path:
    return _root_dir() / "bench" / "rust_accel"


def _lib_name() -> str:
    if sys.platform.startswith("linux"):
        return "libtiger_accel.so"
    if sys.platform == "darwin":
        return "libtiger_accel.dylib"
    return "tiger_accel.dll"


def _lib_path() -> Path:
    return _crate_dir() / "target" / "release" / _lib_name()


def _try_load() -> Optional[ctypes.CDLL]:
    path = _lib_path()
    if not path.exists():
        return None
    return ctypes.CDLL(str(path))


def _build() -> bool:
    crate = _crate_dir()
    if not crate.exists():
        return False
    if shutil.which("cargo") is None:
        return False
    env = os.environ.copy()
    env.setdefault("CARGO_TERM_COLOR", "never")
    try:
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=str(crate),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        return True
    except Exception:
        return False


def _bind_functions(lib: ctypes.CDLL) -> None:
    global _FUNCS_BOUND
    if _FUNCS_BOUND:
        return
    lib.tiger_softsign_out_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_float]
    lib.tiger_softsign_out_f64.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double]
    lib.tiger_softsign_out_f32.restype = None
    lib.tiger_softsign_out_f64.restype = None

    lib.tiger_rms_f32.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.tiger_rms_f64.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.tiger_rms_f32.restype = ctypes.c_float
    lib.tiger_rms_f64.restype = ctypes.c_double

    lib.tiger_norm_f32.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.tiger_norm_f64.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.tiger_norm_f32.restype = ctypes.c_float
    lib.tiger_norm_f64.restype = ctypes.c_double
    _FUNCS_BOUND = True


def _load() -> Optional[ctypes.CDLL]:
    global _LIB, _LOAD_FAILED
    with _LOCK:
        if _LIB is not None:
            return _LIB
        if _LOAD_FAILED:
            return None
        lib = _try_load()
        if lib is None:
            if not _build():
                _LOAD_FAILED = True
                return None
            lib = _try_load()
        if lib is None:
            _LOAD_FAILED = True
            return None
        _bind_functions(lib)
        _LIB = lib
        return _LIB


def _supports_tensor(x: torch.Tensor) -> bool:
    return x.device.type == "cpu" and x.is_contiguous() and x.dtype in {torch.float32, torch.float64}


def softsign(x: torch.Tensor, tau: float) -> Optional[torch.Tensor]:
    lib = _load()
    if lib is None or not _supports_tensor(x):
        return None
    src = x.contiguous()
    out = torch.empty_like(src)
    length = ctypes.c_size_t(src.numel())
    if src.dtype == torch.float32:
        lib.tiger_softsign_out_f32(ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(src.data_ptr()), length, ctypes.c_float(float(tau)))
    else:
        lib.tiger_softsign_out_f64(ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(src.data_ptr()), length, ctypes.c_double(float(tau)))
    return out


def rms(x: torch.Tensor) -> Optional[float]:
    lib = _load()
    if lib is None or not _supports_tensor(x):
        return None
    src = x.contiguous()
    length = ctypes.c_size_t(src.numel())
    if src.dtype == torch.float32:
        return float(lib.tiger_rms_f32(ctypes.c_void_p(src.data_ptr()), length))
    return float(lib.tiger_rms_f64(ctypes.c_void_p(src.data_ptr()), length))


def norm(x: torch.Tensor) -> Optional[float]:
    lib = _load()
    if lib is None or not _supports_tensor(x):
        return None
    src = x.contiguous()
    length = ctypes.c_size_t(src.numel())
    if src.dtype == torch.float32:
        return float(lib.tiger_norm_f32(ctypes.c_void_p(src.data_ptr()), length))
    return float(lib.tiger_norm_f64(ctypes.c_void_p(src.data_ptr()), length))


def is_available() -> bool:
    if _LIB is not None:
        return True
    if _LOAD_FAILED:
        return False
    if _lib_path().exists():
        return True
    return shutil.which("cargo") is not None and _crate_dir().exists()
