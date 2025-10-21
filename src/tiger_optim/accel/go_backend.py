# =============================================================================
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
# =============================================================================

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


def _module_dir() -> Path:
    return _root_dir() / "bench" / "go_accel"


def _build_dir() -> Path:
    return _module_dir() / "build"


def _lib_basename() -> str:
    if sys.platform.startswith("linux"):
        return "libtiger_go_accel.so"
    if sys.platform == "darwin":
        return "libtiger_go_accel.dylib"
    return "tiger_go_accel.dll"


def _lib_path() -> Path:
    return _build_dir() / _lib_basename()


def _try_load() -> Optional[ctypes.CDLL]:
    path = _lib_path()
    if not path.exists():
        return None
    try:
        return ctypes.CDLL(str(path))
    except OSError:
        return None


def _build() -> bool:
    module_dir = _module_dir()
    if not module_dir.exists():
        return False
    if shutil.which("go") is None:
        return False
    build_dir = _build_dir()
    build_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("CGO_ENABLED", "1")
    try:
        subprocess.run(
            [
                "go",
                "build",
                "-buildmode=c-shared",
                "-o",
                str(_lib_path()),
            ],
            cwd=str(module_dir),
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
    lib.TigerSoftsignOutF32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_float,
    ]
    lib.TigerSoftsignOutF64.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_double,
    ]
    lib.TigerSoftsignOutF32.restype = None
    lib.TigerSoftsignOutF64.restype = None

    lib.TigerRmsF32.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.TigerRmsF64.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.TigerRmsF32.restype = ctypes.c_float
    lib.TigerRmsF64.restype = ctypes.c_double

    lib.TigerNormF32.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.TigerNormF64.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.TigerNormF32.restype = ctypes.c_float
    lib.TigerNormF64.restype = ctypes.c_double
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
        lib.TigerSoftsignOutF32(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(src.data_ptr()),
            length,
            ctypes.c_float(float(tau)),
        )
    else:
        lib.TigerSoftsignOutF64(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(src.data_ptr()),
            length,
            ctypes.c_double(float(tau)),
        )
    return out


def rms(x: torch.Tensor) -> Optional[float]:
    lib = _load()
    if lib is None or not _supports_tensor(x):
        return None
    src = x.contiguous()
    length = ctypes.c_size_t(src.numel())
    if src.dtype == torch.float32:
        return float(lib.TigerRmsF32(ctypes.c_void_p(src.data_ptr()), length))
    return float(lib.TigerRmsF64(ctypes.c_void_p(src.data_ptr()), length))


def norm(x: torch.Tensor) -> Optional[float]:
    lib = _load()
    if lib is None or not _supports_tensor(x):
        return None
    src = x.contiguous()
    length = ctypes.c_size_t(src.numel())
    if src.dtype == torch.float32:
        return float(lib.TigerNormF32(ctypes.c_void_p(src.data_ptr()), length))
    return float(lib.TigerNormF64(ctypes.c_void_p(src.data_ptr()), length))


def is_available() -> bool:
    if _LIB is not None:
        return True
    if _LOAD_FAILED:
        return False
    if _lib_path().exists():
        return True
    return shutil.which("go") is not None and _module_dir().exists()
