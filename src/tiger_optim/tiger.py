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
import logging
import math, time, os, json
from copy import deepcopy
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.optim import Optimizer

from .accel import fast_norm, fast_rms, fast_softsign


_LOG = logging.getLogger(__name__)

_CHECKPOINT_VERSION = 1
_CHECKPOINT_FIELDS = (
    "_global_step", "_last_metrics", "_ffn", "_plateau",
    "_pending_lr_scale", "_pending_group_updates", "_last_reflect",
    "_trust_ema", "_qkv_trust_ema", "_lora_pid", "_lora_cross_state",
    "_lora_bridge", "_qkv_lr_ema", "_qkv_disp_ema", "_qkv_disp_ema_prev",
    "_qkv_spec_ema",
)


def _group_tensors_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _group_tensors_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_group_tensors_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_group_tensors_to_device(item, device) for item in value)
    return value


def _rms(x):
    return fast_rms(x)


def _norm(x):
    return fast_norm(x)


def _softsign(x, tau):
    return fast_softsign(x, tau)


def _nan_to_num_(tensor: torch.Tensor, *, nan: float = 0.0,
                 posinf: Optional[float] = None,
                 neginf: Optional[float] = None) -> torch.Tensor:
    """In-place ``nan_to_num`` that guards for non-floating tensors."""

    if torch.is_floating_point(tensor):
        finfo = torch.finfo(tensor.dtype)
        hi = finfo.max if posinf is None else posinf
        lo = finfo.min if neginf is None else neginf
        torch.nan_to_num_(tensor, nan=nan, posinf=hi, neginf=lo)
    return tensor


def _clamp_finite(x, *, lo: Optional[float] = None, hi: Optional[float] = None):
    """Clamp while ensuring a finite value is returned."""

    if isinstance(x, torch.Tensor):
        if torch.is_floating_point(x):
            _nan_to_num_(x)
            if lo is not None or hi is not None:
                x.clamp_(min=lo if lo is not None else -math.inf,
                         max=hi if hi is not None else math.inf)
        return x

    # scalar path
    if not math.isfinite(float(x)):
        x = float(0.0)
    if lo is not None and x < lo:
        x = float(lo)
    if hi is not None and x > hi:
        x = float(hi)
    return x
def _to_dtype(x: torch.Tensor, dtype: torch.dtype): return x if x.dtype == dtype else x.to(dtype)
def _is_compiling() -> bool:
    try:
        return bool(getattr(torch, "compiler").is_compiling())
    except Exception:
        return False

def _scalar_like(
    reference: Optional[torch.Tensor],
    value: float,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create a scalar tensor aligned with ``reference`` or explicit overrides."""

    if reference is not None:
        target_device = device if device is not None else reference.device
        target_dtype = dtype if dtype is not None else reference.dtype
        return reference.new_tensor(value, dtype=target_dtype, device=target_device)
    target_dtype = dtype if dtype is not None else torch.get_default_dtype()
    target_device = torch.device("cpu") if device is None else device
    return torch.tensor(value, dtype=target_dtype, device=target_device)


def _scalar_tensor(
    value: Union[torch.Tensor, float, int],
    reference: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value
        if out.ndim != 0:
            out = out.reshape(())
        if out.device != reference.device or out.dtype != dtype:
            out = out.to(device=reference.device, dtype=dtype)
        return out
    return _scalar_like(reference, float(value), dtype=dtype, device=reference.device)


def _scalar_to_float(value: Union[torch.Tensor, float, int]) -> float:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            value = value.reshape(())
        return float(value.item())
    return float(value)


def _foreach_all_finite(tensors: Sequence[torch.Tensor]) -> Optional[bool]:
    if not tensors:
        return True
    if not hasattr(torch, "_foreach_norm"):
        return None
    detached = [tensor.detach() for tensor in tensors]
    try:
        norms = torch._foreach_norm(detached)
    except Exception:
        return None
    try:
        if all(isinstance(norm, torch.Tensor) and norm.ndim == 0 for norm in norms):
            stacked = torch.stack(list(norms))
        else:
            reference = detached[0]
            stacked = torch.stack([_scalar_tensor(norm, reference) for norm in norms])
    except Exception:
        return None
    return bool(torch.isfinite(stacked).all().item())


def _median_tensor(
    vals: List[torch.Tensor],
    *,
    reference: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    ref = reference if reference is not None else (vals[0] if vals else None)
    target_dtype = dtype if dtype is not None else (ref.dtype if ref is not None else None)
    target_device = device if device is not None else (ref.device if ref is not None else None)

    if len(vals) == 0:
        base_dtype = target_dtype if target_dtype is not None else torch.float32
        return _scalar_like(ref, 0.0, dtype=base_dtype, device=target_device)

    t = torch.stack(vals)  # (N,)
    k = len(vals)//2
    topk = torch.topk(t, k+1, largest=False).values
    median = topk[-1]

    if target_device is not None and median.device != target_device:
        median = median.to(device=target_device)
    if target_dtype is not None and median.dtype != target_dtype:
        median = median.to(dtype=target_dtype)
    return median


def _reference_tensor(
    reference: Optional[torch.Tensor],
    *candidates: object,
) -> Optional[torch.Tensor]:
    if reference is not None:
        return reference

    def _scan(obj: object) -> Optional[torch.Tensor]:
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, Mapping):
            for value in obj.values():
                found = _scan(value)
                if found is not None:
                    return found
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            for item in obj:
                found = _scan(item)
                if found is not None:
                    return found
        return None

    for candidate in candidates:
        found = _scan(candidate)
        if found is not None:
            return found
    return None


def _spectral_dispersion_tensors(
    x: Optional[torch.Tensor], low_band: float, high_band: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate low/high spectral energy and phase spread without syncing to Python."""

    if x is None:
        zero = torch.zeros((), dtype=torch.float32)
        return zero, zero, zero

    y = x.reshape(-1).detach()
    zero = _scalar_like(y, 0.0, dtype=torch.float32, device=y.device)
    if y.numel() <= 1:
        return zero, zero, zero

    if not y.is_floating_point() or y.dtype != torch.float32:
        y = y.to(torch.float32)

    # remove mean so DC energy does not dominate the dispersion metric
    y = y - y.mean()

    if y.device.type == "mps":
        # MPS's rfft wrapper resizes an internal scalar output and warns on
        # every call unless its known complex64 result shape is supplied.
        spectrum = torch.empty(y.numel() // 2 + 1, dtype=torch.complex64, device=y.device)
        spec = torch.fft.rfft(y, out=spectrum)
    else:
        spec = torch.fft.rfft(y)
    power = spec.abs().pow(2)
    if power.numel() == 0:
        return zero, zero, zero

    # drop the DC component when slicing the spectrum
    side = power[1:] if power.numel() > 1 else power
    if side.numel() == 0:
        val = power.mean().reshape(()) if power.numel() else zero
        return val, val, zero

    n_eff = side.numel()
    # ensure the requested bands are inside (0, 1]
    low_band = float(max(0.0, min(1.0, low_band)))
    high_band = float(max(0.0, min(1.0, high_band)))
    low_len = max(1, min(n_eff, int(math.ceil(low_band * n_eff))))
    high_len = max(1, min(n_eff, int(math.ceil(high_band * n_eff))))

    low_slice = side[:low_len]
    high_slice = side[-high_len:]

    low_energy = low_slice.mean().reshape(()) if low_slice.numel() else zero
    high_energy = high_slice.mean().reshape(()) if high_slice.numel() else zero

    if spec.numel() > 1:
        phase = torch.angle(spec[1:])
        # resultant length indicates phase coherence (1 -> aligned, 0 -> dispersed)
        phase_vector = torch.polar(torch.ones_like(phase), phase)
        phase_coherence = torch.abs(torch.mean(phase_vector))
        # convert to spread in [0, 1]
        phase_spread = (1.0 - phase_coherence.clamp(0.0, 1.0)).reshape(())
    else:
        phase_spread = zero

    return low_energy, high_energy, phase_spread


def _spectral_dispersion_chunks(
    chunks: Sequence[torch.Tensor], low_band: float, high_band: float,
    fused: Optional[torch.Tensor] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Measure equal-length Q/K/V chunks with one FFT; preserve other layouts.

    ``fused`` is the contiguous source of three equal dim-0 chunks, when
    available. Its row view avoids copying the Q/K/V chunks into a stack.
    """

    if (
        len(chunks) != 3
        or any(chunk.numel() <= 1 for chunk in chunks)
        or len({chunk.numel() for chunk in chunks}) != 1
        or len({chunk.device for chunk in chunks}) != 1
    ):
        return [_spectral_dispersion_tensors(chunk, low_band, high_band) for chunk in chunks]

    if (
        fused is not None
        and fused.is_contiguous()
        and fused.numel() == 3 * chunks[0].numel()
        and fused.device == chunks[0].device
        and fused.dtype == chunks[0].dtype
    ):
        rows = fused.detach().view(3, -1).to(torch.float32)
    else:
        rows = torch.stack([chunk.reshape(-1).detach().to(torch.float32) for chunk in chunks])
    rows = rows - rows.mean(dim=-1, keepdim=True)
    if rows.device.type == "mps":
        spectrum = torch.empty(
            (len(chunks), rows.shape[-1] // 2 + 1),
            dtype=torch.complex64,
            device=rows.device,
        )
        spec = torch.fft.rfft(rows, dim=-1, out=spectrum)
    else:
        spec = torch.fft.rfft(rows, dim=-1)

    power = spec.abs().pow(2)
    side = power[:, 1:]
    n_eff = side.shape[-1]
    low_band = float(max(0.0, min(1.0, low_band)))
    high_band = float(max(0.0, min(1.0, high_band)))
    low_len = max(1, min(n_eff, int(math.ceil(low_band * n_eff))))
    high_len = max(1, min(n_eff, int(math.ceil(high_band * n_eff))))
    low_energy = side[:, :low_len].mean(dim=-1)
    high_energy = side[:, -high_len:].mean(dim=-1)
    phase = torch.angle(spec[:, 1:])
    phase_vector = torch.polar(torch.ones_like(phase), phase)
    phase_coherence = torch.abs(phase_vector.mean(dim=-1))
    phase_spread = 1.0 - phase_coherence.clamp(0.0, 1.0)
    return [
        (low_energy[i].reshape(()), high_energy[i].reshape(()), phase_spread[i].reshape(()))
        for i in range(len(chunks))
    ]


def _spectral_dispersion(x: torch.Tensor, low_band: float, high_band: float) -> Tuple[float, float, float]:
    """Estimate mean spectral energy for low/high bands and the phase spread."""

    low_energy, high_energy, phase_spread = _spectral_dispersion_tensors(x, low_band, high_band)
    return (
        _scalar_to_float(low_energy),
        _scalar_to_float(high_energy),
        _scalar_to_float(phase_spread),
    )


def _chunk_scalar_norms(x: torch.Tensor, dim: int, parts: int) -> torch.Tensor:
    """Compute equal-split chunk norms in one pass when possible."""

    if x.ndim == 0 or parts <= 0:
        return torch.stack([_scalar_tensor(_norm(x), x)])

    dim = dim % x.ndim
    size = x.shape[dim]
    if size % parts != 0:
        return torch.stack([_scalar_tensor(_norm(chunk), x) for chunk in torch.chunk(x, parts, dim=dim)])

    moved = x.detach().movedim(dim, 0)
    chunk_size = size // parts
    if moved.numel() == 0:
        return torch.zeros(parts, dtype=torch.float32, device=x.device)
    flat = moved.reshape(parts, chunk_size, -1)
    if not flat.is_floating_point() or flat.dtype != torch.float32:
        flat = flat.to(torch.float32)
    scale = flat.abs().amax(dim=(1, 2), keepdim=True)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale.reshape(parts) * torch.linalg.vector_norm(flat / safe_scale, dim=(1, 2))


def _scalars_to_host(
    values: Sequence[Union[torch.Tensor, float, int]],
    *,
    reference: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
) -> List[float]:
    """Move multiple scalar-like values to Python with a single host transfer."""

    if not values:
        return []
    ref = _reference_tensor(reference, values)
    stacked = torch.stack([
        _scalar_tensor(value, ref, dtype=dtype) if ref is not None else torch.as_tensor(value, dtype=dtype)
        for value in values
    ])
    return [float(item) for item in stacked.detach().cpu().tolist()]

class _Profiler:
    def __init__(self, enabled=False, path="benchmarks/profiles/tiger.jsonl", interval=10, ema_decay=0.9):
        self.enabled = bool(enabled)
        self.path = path
        self.interval = max(1, int(interval))
        self.ema_decay = float(ema_decay)
        self.ema_ms = None
        self._bucket_sizes = []
        self._last_payload = {}

    def _write(self, payload):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def log_step(self, step_ms: float, step_idx: int, payload: Dict):
        if not self.enabled: return
        self.ema_ms = step_ms if self.ema_ms is None else (self.ema_decay*self.ema_ms + (1-self.ema_decay)*step_ms)
        bsz = payload.get("foreach_bucket_size", None)
        if bsz is not None: self._bucket_sizes.append(int(bsz))
        if self._bucket_sizes:
            n = len(self._bucket_sizes)
            mean = sum(self._bucket_sizes)/n
            var  = sum((x-mean)*(x-mean) for x in self._bucket_sizes)/n
            payload["foreach_bucket_mean"] = mean
            payload["foreach_bucket_var"]  = var
        payload.update(step=step_idx, step_ms=step_ms, ema_ms=self.ema_ms)
        self._last_payload = dict(payload)
        if (step_idx % self.interval) == 0:
            self._write(payload)

    def last_payload(self): return dict(self._last_payload)

def _merge_policy(old, new, policy: str):
    if policy in {"replace","set"}:
        return new
    if policy == "merge" and isinstance(old, dict) and isinstance(new, dict):
        out = dict(old); out.update(new); return out
    if policy == "add":
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            return type(old)(old + new)
        return new
    if policy == "mul":
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            return type(old)(old * new)
        return new
    if policy == "pow":
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            return type(old)(float(old) ** float(new))
        return old
    if policy == "exp":
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            return type(old)(float(old) * math.exp(float(new)))
        return old
    if policy == "clip":
        if isinstance(old, (int, float)) and (isinstance(new, (tuple, list)) and len(new)==2):
            lo, hi = float(new[0]), float(new[1])
            return type(old)(min(hi, max(lo, float(old))))
        return old
    if policy == "logit-clip":
        # payload: (delta, lo, hi)
        if isinstance(old, (int, float)) and (isinstance(new, (tuple, list)) and len(new)>=1):
            delta = float(new[0]); lo = float(new[1]) if len(new)>=2 else 0.0; hi = float(new[2]) if len(new)>=3 else 1.0
            x = float(old); x = min(1.0-1e-6, max(1e-6, x))
            z = math.log(x/(1.0-x)); x_new = 1.0/(1.0 + math.exp(-(z + delta)))
            x_new = min(hi, max(lo, x_new))
            return x_new
        return old
    return new

def _apply_field_spec(old_val, spec, default_policy="replace"):
    """Apply an explicit field operation or the staged update's policy."""
    # pipeline
    if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], (list, tuple)) and len(spec[0])==2:
        val = old_val
        for op, payload in spec:
            val = _merge_policy(val, payload, str(op))
        return val
    # single op
    if isinstance(spec, (tuple, list)) and len(spec)==2 and str(spec[0]) in {"add","mul","clip","replace","set","merge","pow","exp","logit-clip"}:
        return _merge_policy(old_val, spec[1], str(spec[0]))
    return _merge_policy(old_val, spec, default_policy)

class Tiger(Optimizer):
    """Tiger v2.1.0
    - LoRA-PID minima & recovery schedule for Ki/Kd (slow recovery, fast decay)
    - Pending arithmetic pipeline: [(op,payload), ...] sequential application
    - QKV step-clip via dispersion acceleration (2nd diff) + bounds
    - Triton experimental fused-apply path (apply updates + bucket stats in one kernel)
    - Keeps v2.0 features: inertia modes, combined stats kernel, pow/exp/logit-clip
    """
    def __init__(self, params,
                 lr=2e-4, betas=(0.9,0.98), eps=1e-8, weight_decay=0.0,
                 factored=True, precond_alpha=1.0,
                 # sign
                 sign_mode="softsign", sign_tau=1e-3, sign_blend=0.2,
                 blend_to=0.6, blend_steps=0, blend_schedule="cosine",
                 # trust
                 use_trust_ratio=True, trust_clip=10.0, trust_space="update",
                 trust_ema_beta=0.0,
                 # AGC
                 agc_clip=0.0, agc_eps=1e-12,
                 # foreach
                 use_foreach=True, use_foreach_update=True, foreach_min_bucket=4,
                 update_buffer_dtype="param",
                 bucket_standardize=False, bucket_standardize_source="global",
                 bucket_scalarless=True, use_triton_bucket_stats=False, use_triton_fused_apply=False,
                 # momentum dtype
                 mom_dtype: str = "fp32",
                 # sparse-ish state pruning
                 state_prune_threshold=0.0, state_prune_interval=100,
                 # nonfinite
                 skip_if_nonfinite=True,
                 # Auto LR & plateau
                 auto_lr=True, lr_decay=0.5, lr_min=1e-6, plateau_patience=200, plateau_tol=1e-4,
                 # Auto Blend
                 auto_blend=True, auto_blend_gain=0.05, auto_blend_bounds=(0.1, 0.8),
                 # Auto FFN Asym
                 auto_ffn_asym=False, ffn_asym_target=1.0, ffn_asym_gain=0.05, ffn_asym_beta=0.9, ffn_asym_interval=10, ffn_lr_min=0.5, ffn_lr_max=1.5,
                 # LoRA EMA + PID + inertia + minima + recovery
                 lora_density_adapt=True, lora_density_k=0.25, lora_density_beta=0.9,
                 lora_sb_bounds=(0.1, 0.5), lora_agc_bounds=(0.01, 0.05),
                 lora_pid_kp=0.6, lora_pid_ki=0.05, lora_pid_kd=0.1, lora_interval=20,
                 lora_int_clip_sb=0.5, lora_int_clip_agc=0.5, lora_aw_reset_on_saturation=True, lora_aw_eps=1e-3,
                 lora_dwell_decay=0.9, lora_dwell_gain=0.05, lora_errflip_damp_sb=0.5, lora_errflip_damp_agc=0.5,
                 lora_pid_mode="auto", lora_inertia_beta=0.9, lora_inertia_strength=1.5, lora_flip_ema_beta=0.8,
                 lora_ki_min=0.01, lora_kd_min=0.005, lora_recover_rate=0.15,
                 lora_cross_adapt=True, lora_cross_beta=0.6, lora_cross_sync=0.25,
                 lora_cross_gain=0.1, lora_cross_attn_tags=("attn_qkv",), lora_cross_ffn_tags=("ffn_up","ffn_down"),
                 lora_bridge=True, lora_bridge_gain=0.25, lora_bridge_bounds=(0.6, 1.4), lora_bridge_beta=0.8,
                 # QKV two-objective + gamma auto-scale + step-clip acceleration shrink
                 qkv_lr_autoadapt=True, qkv_w_rms=0.7, qkv_w_trust=0.3,
                 qkv_lr_gain=0.02, qkv_lr_bounds=(0.7, 1.3), qkv_lr_interval=25, qkv_lr_ema_beta=0.8,
                 qkv_gain_shrink_gamma=1.2, qkv_lr_step_clip=0.08,
                 qkv_disp_ema_beta=0.8, qkv_gamma_rate=0.6, qkv_gamma_min=0.2, qkv_gamma_max=5.0,
                 qkv_clip_shrink_k=0.6, qkv_clip_min=0.02, qkv_clip_max=0.2,
                 qkv_spectral_adapt=True, qkv_spectral_low_band=0.2, qkv_spectral_high_band=0.25,
                 qkv_spectral_beta=0.7, qkv_spectral_eps=1e-9,
                 qkv_gamma_spectral_gain=0.75, qkv_gamma_spectral_clip=(0.5, 1.5),
                 qkv_phase_boost_gain=0.4, qkv_phase_target=0.6, qkv_phase_boost_clip=(0.6, 1.6),
                 # compile staged reflect
                 compile_guard=True, reflect_interval=50,
                 # profiler
                 profiler_enabled=False, profiler_path="benchmarks/profiles/tiger.jsonl", profiler_interval=10, profiler_ema_decay=0.9,
                 qkv_spectral_strength=1.0):
        if not 0.0 <= float(lr_decay) <= 1.0:
            raise ValueError("lr_decay must be in [0, 1]")
        if float(lr_min) < 0.0:
            raise ValueError("lr_min must be nonnegative")
        if int(plateau_patience) < 1:
            raise ValueError("plateau_patience must be positive")
        spectral_strength = float(qkv_spectral_strength)
        if not math.isfinite(spectral_strength) or not 0.0 <= spectral_strength <= 1.0:
            raise ValueError("qkv_spectral_strength must be finite and in [0, 1]")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        factored=factored, precond_alpha=precond_alpha,
                        sign_mode=sign_mode, sign_tau=sign_tau, sign_blend=sign_blend,
                        blend_to=blend_to, blend_steps=blend_steps, blend_schedule=blend_schedule,
                        use_trust_ratio=use_trust_ratio, trust_clip=trust_clip, trust_space=trust_space,
                        trust_ema_beta=float(trust_ema_beta),
                        agc_clip=agc_clip, agc_eps=agc_eps,
                        use_foreach=use_foreach, use_foreach_update=use_foreach_update, foreach_min_bucket=int(foreach_min_bucket),
                        update_buffer_dtype=update_buffer_dtype,
                        bucket_standardize=bucket_standardize, bucket_standardize_source=bucket_standardize_source, bucket_scalarless=bucket_scalarless,
                        use_triton_bucket_stats=use_triton_bucket_stats, use_triton_fused_apply=use_triton_fused_apply,
                        mom_dtype=mom_dtype,
                        state_prune_threshold=state_prune_threshold, state_prune_interval=int(state_prune_interval),
                        skip_if_nonfinite=bool(skip_if_nonfinite),
                        auto_lr=auto_lr, lr_decay=lr_decay, lr_min=lr_min,
                        plateau_patience=plateau_patience, plateau_tol=plateau_tol,
                        auto_blend=auto_blend, auto_blend_gain=auto_blend_gain, auto_blend_bounds=auto_blend_bounds,
                        auto_ffn_asym=auto_ffn_asym, ffn_asym_target=ffn_asym_target, ffn_asym_gain=ffn_asym_gain,
                        ffn_asym_beta=ffn_asym_beta, ffn_asym_interval=ffn_asym_interval, ffn_lr_min=ffn_lr_min, ffn_lr_max=ffn_lr_max,
                        lr_scale=1.0, block_tag="default",
                        rms_clip_threshold=0.0, rms_clip_granularity="param",
                        qkv_rules={}, qkv_trust_split=False, qkv_lr_scales=None,
                        block_trust_chunks=0,
                        # LoRA PID++
                        lora_density_adapt=lora_density_adapt, lora_density_k=float(lora_density_k), lora_density_beta=float(lora_density_beta),
                        lora_sb_bounds=tuple(lora_sb_bounds), lora_agc_bounds=tuple(lora_agc_bounds),
                        lora_pid_kp=float(lora_pid_kp), lora_pid_ki=float(lora_pid_ki), lora_pid_kd=float(lora_pid_kd),
                        lora_interval=int(lora_interval), lora_int_clip_sb=float(lora_int_clip_sb), lora_int_clip_agc=float(lora_int_clip_agc),
                        lora_aw_reset_on_saturation=bool(lora_aw_reset_on_saturation), lora_aw_eps=float(lora_aw_eps),
                        lora_dwell_decay=float(lora_dwell_decay), lora_dwell_gain=float(lora_dwell_gain),
                        lora_errflip_damp_sb=float(lora_errflip_damp_sb), lora_errflip_damp_agc=float(lora_errflip_damp_agc),
                        lora_pid_mode=str(lora_pid_mode), lora_inertia_beta=float(lora_inertia_beta), lora_inertia_strength=float(lora_inertia_strength), lora_flip_ema_beta=float(lora_flip_ema_beta),
                        lora_ki_min=float(lora_ki_min), lora_kd_min=float(lora_kd_min), lora_recover_rate=float(lora_recover_rate),
                        lora_cross_adapt=bool(lora_cross_adapt), lora_cross_beta=float(lora_cross_beta),
                        lora_cross_sync=float(lora_cross_sync), lora_cross_gain=float(lora_cross_gain),
                        lora_cross_attn_tags=tuple(lora_cross_attn_tags), lora_cross_ffn_tags=tuple(lora_cross_ffn_tags),
                        lora_bridge=bool(lora_bridge), lora_bridge_gain=float(lora_bridge_gain),
                        lora_bridge_bounds=tuple(lora_bridge_bounds), lora_bridge_beta=float(lora_bridge_beta),
                        # QKV 2obj + gamma auto-scale + step-clip accel shrink
                        qkv_lr_autoadapt=qkv_lr_autoadapt, qkv_w_rms=float(qkv_w_rms), qkv_w_trust=float(qkv_w_trust),
                        qkv_lr_gain=float(qkv_lr_gain), qkv_lr_bounds=tuple(qkv_lr_bounds), qkv_lr_interval=int(qkv_lr_interval), qkv_lr_ema_beta=float(qkv_lr_ema_beta),
                        qkv_gain_shrink_gamma=float(qkv_gain_shrink_gamma), qkv_lr_step_clip=float(qkv_lr_step_clip),
                        qkv_disp_ema_beta=float(qkv_disp_ema_beta), qkv_gamma_rate=float(qkv_gamma_rate), qkv_gamma_min=float(qkv_gamma_min), qkv_gamma_max=float(qkv_gamma_max),
                        qkv_clip_shrink_k=float(qkv_clip_shrink_k), qkv_clip_min=float(qkv_clip_min), qkv_clip_max=float(qkv_clip_max),
                        qkv_spectral_adapt=bool(qkv_spectral_adapt), qkv_spectral_low_band=float(qkv_spectral_low_band),
                        qkv_spectral_high_band=float(qkv_spectral_high_band), qkv_spectral_beta=float(qkv_spectral_beta),
                        qkv_spectral_eps=float(qkv_spectral_eps),
                        qkv_spectral_strength=spectral_strength,
                        qkv_gamma_spectral_gain=float(qkv_gamma_spectral_gain), qkv_gamma_spectral_clip=tuple(qkv_gamma_spectral_clip),
                        qkv_phase_boost_gain=float(qkv_phase_boost_gain), qkv_phase_target=float(qkv_phase_target),
                        qkv_phase_boost_clip=tuple(qkv_phase_boost_clip),
                        # compile reflect
                        compile_guard=compile_guard, reflect_interval=int(reflect_interval))
        super().__init__(params, defaults)
        self._global_step = 0
        self._last_metrics = {}
        self._ffn = {"ema_up": None, "ema_down": None, "last": 0}
        self._plateau = {"best": None, "streak": 0, "last_loss": None}
        self._mom_storage_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[mom_dtype.lower()]
        self._prof = _Profiler(enabled=profiler_enabled, path=profiler_path, interval=profiler_interval, ema_decay=profiler_ema_decay)

        # pending queues
        self._pending_lr_scale: Dict[int, float] = {}
        self._pending_group_updates: Dict[int, List[Tuple[Dict[str,object], str]]] = {}
        self._last_reflect = 0

        # trust EMA per group id
        self._trust_ema: Dict[Tuple[int, int], Union[float, torch.Tensor]] = {}
        self._qkv_trust_ema: Dict[Tuple[int, int], torch.Tensor] = {}
        # LoRA PID state per group id
        self._lora_pid: Dict[int, Dict[str, float]] = {}  # plus ki_eff/kd_eff for recovery
        self._lora_cross_state: Dict[str, object] = {"global_density": None, "tag_density": {}}
        self._lora_bridge: Dict[int, Dict[str, float]] = {}
        # QKV LR EMA per group & dispersion EMA (and previous for accel)
        self._qkv_lr_ema: Dict[int, Dict[str, float]] = {}
        self._qkv_disp_ema: Dict[int, float] = {}
        self._qkv_disp_ema_prev: Dict[int, float] = {}
        self._qkv_spec_ema: Dict[int, Dict[str, Dict[str, Union[float, torch.Tensor]]]] = {}
        self._group_scalar_cache: Dict[int, Dict[str, Tuple[float, torch.Tensor]]] = {}
        self._flat_cache: Dict[int, "OrderedDict[object, object]"] = {}
        # Flat-entry cap per group: membership changes (e.g. periodic qkv
        # stat capture) would otherwise accumulate F + graph + pool without
        # bound. Entries are LRU-evicted; st["m"]/["v"] views keep an evicted
        # buffer alive until the next membership re-syncs them.
        try:
            self._flat_cache_max = max(
                1, int(os.environ.get("TIGER_FLAT_CACHE_MAX", "4")))
        except ValueError:
            self._flat_cache_max = 4
        # One graph pool per device, shared by every captured flat graph, so
        # transient capture allocations are recycled instead of each graph
        # keeping a private pool.
        self._graph_pools: Dict[torch.device, object] = {}
        self._fused_apply_failures: List[str] = []

    # ---------- Public API ----------
    @property
    def qkv_spectral_strength(self) -> float:
        """Current feedback strength; checkpoints predating this option use one."""
        return float(self.defaults.get("qkv_spectral_strength", 1.0))

    def set_qkv_spectral_strength(self, strength: float) -> None:
        """Blend spectral corrections toward neutral, without changing LR scales.

        Zero skips spectral collection while RMS/trust adaptation continues.
        Spectral EMA history is retained during a pause. This eager control and
        its current value are saved with the optimizer's adaptive state.
        """
        value = float(strength)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("qkv_spectral_strength must be finite and in [0, 1]")
        self.defaults["qkv_spectral_strength"] = value

    def state_dict(self):
        """Save parameter state and the adaptive state needed for exact resume."""

        packed = super().state_dict()
        for group, saved_group in zip(self.param_groups, packed["param_groups"]):
            rules = group.get("qkv_rules") or {}
            if not rules:
                continue
            packed_ids = {id(param): saved_id for param, saved_id in zip(group["params"], saved_group["params"])}
            unknown = set(rules).difference(packed_ids)
            if unknown:
                raise ValueError("qkv_rules contains keys that do not identify parameters in its group")
            saved_group["qkv_rules"] = {packed_ids[key]: tuple(rule) for key, rule in rules.items()}
        packed["tiger_state"] = {
            "version": _CHECKPOINT_VERSION,
            "defaults": deepcopy(self.defaults),
            **{name: deepcopy(getattr(self, name)) for name in _CHECKPOINT_FIELDS},
        }
        return packed

    def load_state_dict(self, state_dict):
        """Restore adaptive state and translate QKV rules to current parameters."""

        tiger_state = state_dict.get("tiger_state")
        if tiger_state is not None and tiger_state.get("version") != _CHECKPOINT_VERSION:
            raise ValueError(f"unsupported Tiger checkpoint version: {tiger_state.get('version')!r}")

        prepared = dict(state_dict)
        saved_groups = []
        for saved_group, current_group in zip(state_dict["param_groups"], self.param_groups):
            saved_group = dict(saved_group)
            rules = saved_group.get("qkv_rules") or {}
            if rules:
                if tiger_state is None:
                    # Old checkpoints stored process-local id(param) keys. The
                    # caller must provide the rules again when constructing Tiger.
                    current_rules = current_group.get("qkv_rules") or {}
                    if not current_rules:
                        raise ValueError("legacy checkpoint has QKV rules; construct Tiger with fresh qkv_rules before loading")
                    saved_group["qkv_rules"] = dict(current_rules)
                else:
                    saved_to_current = dict(zip(saved_group["params"], current_group["params"]))
                    if set(rules).difference(saved_to_current):
                        raise ValueError("checkpoint QKV rules reference parameters outside their group")
                    saved_group["qkv_rules"] = {
                        id(saved_to_current[key]): tuple(rule) for key, rule in rules.items()
                    }
            saved_groups.append(saved_group)
        prepared["param_groups"] = saved_groups
        super().load_state_dict(prepared)
        # Older versions could mark nonfinite moments as sane. Recheck loaded
        # moments once before the first update.
        for param_state in self.state.values():
            if isinstance(param_state, dict):
                param_state["_state_sane"] = False

        if tiger_state is None:
            _LOG.warning("Loaded a legacy Tiger checkpoint without adaptive state; exact continuation is unavailable")
            self._global_step = 0
            self._last_metrics = {}
            self._ffn = {"ema_up": None, "ema_down": None, "last": 0}
            self._plateau = {"best": None, "streak": 0, "last_loss": None}
            self._pending_lr_scale = {}
            self._pending_group_updates = {}
            self._last_reflect = 0
            self._trust_ema = {}
            self._qkv_trust_ema = {}
            self._lora_pid = {}
            self._lora_cross_state = {"global_density": None, "tag_density": {}}
            self._lora_bridge = {}
            self._qkv_lr_ema = {}
            self._qkv_disp_ema = {}
            self._qkv_disp_ema_prev = {}
            self._qkv_spec_ema = {}
        else:
            self.defaults = deepcopy(tiger_state["defaults"])
            for name in _CHECKPOINT_FIELDS:
                setattr(self, name, deepcopy(tiger_state[name]))
            for name in ("_trust_ema", "_qkv_trust_ema", "_qkv_spec_ema"):
                per_group = getattr(self, name)
                for key, value in list(per_group.items()):
                    gi = key[0] if isinstance(key, tuple) else key
                    if 0 <= gi < len(self.param_groups) and self.param_groups[gi]["params"]:
                        device = self.param_groups[gi]["params"][0].device
                        per_group[key] = _group_tensors_to_device(value, device)

        self._mom_storage_dtype = {
            "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16,
        }[self.defaults["mom_dtype"].lower()]
        self._group_scalar_cache = {}
        self._flat_cache = {}
        self._fused_apply_failures = []

    def report_metrics(self, loss: Optional[float] = None, **kwargs):
        if loss is not None:
            loss_value = float(loss)
            self._last_metrics["loss"] = loss_value
            if math.isfinite(loss_value):
                p = self._plateau
                if p["best"] is None or loss_value < p["best"] - self.defaults["plateau_tol"]:
                    p["best"] = loss_value; p["streak"] = 0
                else:
                    p["streak"] += 1
                p["last_loss"] = loss_value
        for k, v in kwargs.items():
            try: self._last_metrics[k] = float(v)
            except Exception: pass

    def reflect_pending(self):
        applied = 0
        if self._pending_lr_scale:
            for gi, mult in list(self._pending_lr_scale.items()):
                if 0 <= gi < len(self.param_groups):
                    g = self.param_groups[gi]
                    g["lr_scale"] = float(g.get("lr_scale", 1.0)) * float(mult)
                    applied += 1
                del self._pending_lr_scale[gi]
        if self._pending_group_updates:
            for gi, items in list(self._pending_group_updates.items()):
                if 0 <= gi < len(self.param_groups):
                    g = self.param_groups[gi]
                    for fields, policy in items:
                        for key, val in fields.items():
                            g[key] = _apply_field_spec(g.get(key), val, policy)
                            applied += 1
                del self._pending_group_updates[gi]
        self._last_reflect = self._global_step
        return applied

    def _group_scalar(
        self,
        gi: int,
        key: str,
        reference: torch.Tensor,
        value: float,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        cache = self._group_scalar_cache.setdefault(gi, {})
        scalar_value = float(value)
        cached = cache.get(key)
        if cached is not None:
            cached_value, tensor = cached
            if cached_value == scalar_value and tensor.device == reference.device and tensor.dtype == dtype:
                return tensor
        tensor = _scalar_like(reference, scalar_value, dtype=dtype, device=reference.device)
        cache[key] = (scalar_value, tensor)
        return tensor

    def _flat_group(
        self,
        gi: int,
        params: List[torch.Tensor],
        reference: torch.Tensor,
        factored_flags: Optional[List[bool]] = None,
        subsegs: Optional[List[int]] = None,
    ) -> Dict[str, object]:
        """Persistent flat fp32 moment buffers + segment metadata for the
        fused Triton fast path. State m entries (and v for non-factored)
        are views into flat_m/flat_v; factored vr/vc are views into
        flat_vr/flat_vc so kernel writes commit in place."""
        if factored_flags is None:
            factored_flags = [False] * len(params)
        if subsegs is None:
            subsegs = [1] * len(params)
        ids = tuple(id(p) for p in params)
        cache = self._flat_cache.setdefault(gi, OrderedDict())
        key = ("flat", ids, tuple(factored_flags), tuple(subsegs),
               reference.device)
        F = cache.get(key)
        if F is not None:
            cache.move_to_end(key)
            # Re-sync params whose moments are no longer views into THIS
            # buffer (membership changed since last use).
            for p, o, n, fct in zip(params, F["offsets"], F["lengths"], F["factored"]):
                st = self.state[p]
                if st.get("m") is None:
                    st["m"] = F["flat_m"][o:o + n].view_as(p).zero_()
                elif st["m"].untyped_storage().data_ptr() != F["flat_m"].untyped_storage().data_ptr():
                    mv = F["flat_m"][o:o + n].view_as(p)
                    mv.copy_(st["m"].to(torch.float32))
                    st["m"] = mv
                if not fct:
                    if st.get("v") is None:
                        st["v"] = F["flat_v"][o:o + n].view_as(p).zero_()
                    elif st["v"].untyped_storage().data_ptr() != F["flat_v"].untyped_storage().data_ptr():
                        vv = F["flat_v"][o:o + n].view_as(p)
                        vv.copy_(st["v"].to(torch.float32))
                        st["v"] = vv
                elif F.get("fact_meta") is not None:
                    meta = F["fact_meta"]
                    fo = meta["off_to_fact"].get(o)
                    if fo is not None:
                        r0 = meta["fact_vroff_l"][fo]; R = meta["fact_R_l"][fo]
                        c0 = meta["fact_vcoff_l"][fo]; C = meta["fact_C_l"][fo]
                        for key_, buf, ln in (("vr", F["flat_vr"], R), ("vc", F["flat_vc"], C)):
                            base = r0 if key_ == "vr" else c0
                            cur = st.get(key_)
                            if cur is None:
                                st[key_] = buf[base:base + ln].zero_()
                            elif cur.untyped_storage().data_ptr() != buf.untyped_storage().data_ptr():
                                nv = buf[base:base + ln]
                                nv.copy_(cur.to(torch.float32).reshape(-1))
                                st[key_] = nv
            return F
        dev = reference.device
        lengths = [int(p.numel()) for p in params]
        total = int(sum(lengths))
        flat_m = torch.empty(total, dtype=torch.float32, device=dev)
        flat_v = torch.empty(total, dtype=torch.float32, device=dev)
        fact_idx = [i for i, f in enumerate(factored_flags) if f]
        rows_l = [int(params[i].shape[0]) for i in fact_idx]
        cols_l = [lengths[i] // r for i, r in zip(fact_idx, rows_l)]
        total_r = int(sum(rows_l)); total_c = int(sum(cols_l))
        flat_vr = torch.empty(total_r, dtype=torch.float32, device=dev) if total_r else None
        flat_vc = torch.empty(total_c, dtype=torch.float32, device=dev) if total_c else None
        offsets: List[int] = []
        seg_starts: List[int] = []
        seg_lens: List[int] = []
        off = 0
        vr_off = 0; vc_off = 0
        fact_meta_lists = {
            "row_start": [], "row_len": [], "row_vr": [],
            "cb_start": [], "cb_c0": [], "cb_R": [], "cb_C": [], "cb_vc": [],
            "pb_start": [], "pb_c0": [], "pb_r0": [], "pb_r1": [],
            "pb_C": [], "pb_R": [],
            "cbf_vc": [], "cbf_p0": [], "cbf_np": [], "cbf_rem": [],
            "rbp_start": [], "rbp_c0": [], "rbp_c1": [], "rbp_n": [],
            "rbf_j": [], "rbf_p0": [], "rbf_np": [],
            "fact_vroff": [], "fact_R": [],
            "elem_pos": [], "elem_fact": [],
            "rok_r": [], "rok_c": [],
        }
        off_to_fact: Dict[int, int] = {}
        CBLOCK = 128
        RBLOCK = 4096  # taller matrices switch to the two-phase col reduce
        # Wider matrices switch to the two-phase row reduce only past this
        # width: the partial fold reorders fp32 accumulation, so the gate
        # stays high to keep bitwise-eager behavior for typical shapes.
        WBLOCK = 16384
        npb = 0
        nrpb = 0
        for i, (p, n, fct) in enumerate(zip(params, lengths, factored_flags)):
            st = self.state[p]
            mv = flat_m[off:off + n].view_as(p)
            if "m" in st:
                mv.copy_(st["m"].to(torch.float32))
            else:
                mv.zero_()
            st["m"] = mv
            if not fct:
                vv = flat_v[off:off + n].view_as(p)
                if "v" in st:
                    vv.copy_(st["v"].to(torch.float32))
                else:
                    vv.zero_()
                st["v"] = vv
            else:
                fo = len(off_to_fact)
                off_to_fact[off] = fo
                R = int(p.shape[0]); C = n // R
                vr_v = flat_vr[vr_off:vr_off + R]
                if "vr" in st:
                    vr_v.copy_(st["vr"].to(torch.float32).reshape(-1))
                else:
                    vr_v.zero_()
                st["vr"] = vr_v
                vc_v = flat_vc[vc_off:vc_off + C]
                if "vc" in st:
                    vc_v.copy_(st["vc"].to(torch.float32).reshape(-1))
                else:
                    vc_v.zero_()
                st["vc"] = vc_v
                fm = fact_meta_lists
                seg0 = len(seg_starts)
                if C <= WBLOCK:
                    for r in range(R):
                        fm["row_start"].append(off + r * C)
                        fm["row_len"].append(C)
                        fm["row_vr"].append(vr_off + r)
                        fm["rok_r"].append(seg0)
                else:
                    # Wide matrices: two-phase row reduce over
                    # (row, col-block) pairs, mirroring the tall col path.
                    for r in range(R):
                        fm["rbf_j"].append(vr_off + r)
                        fm["rbf_p0"].append(nrpb)
                        nrb = (C + RBLOCK - 1) // RBLOCK
                        fm["rbf_np"].append(nrb)
                        for c0 in range(0, C, RBLOCK):
                            fm["rbp_start"].append(off + r * C)
                            fm["rbp_c0"].append(c0)
                            fm["rbp_c1"].append(min(c0 + RBLOCK, C))
                            fm["rbp_n"].append(C)
                            nrpb += 1
                        fm["rok_r"].append(seg0)
                for c0 in range(0, C, CBLOCK):
                    if R <= RBLOCK:
                        fm["cb_start"].append(off)
                        fm["cb_c0"].append(c0)
                        fm["cb_R"].append(R)
                        fm["cb_C"].append(C)
                        fm["cb_vc"].append(vc_off + c0)
                    else:
                        fm["cbf_vc"].append(vc_off + c0)
                        fm["cbf_p0"].append(npb)
                        nrb = (R + RBLOCK - 1) // RBLOCK
                        fm["cbf_np"].append(nrb)
                        fm["cbf_rem"].append(min(CBLOCK, C - c0))
                        for r0 in range(0, R, RBLOCK):
                            fm["pb_start"].append(off)
                            fm["pb_c0"].append(c0)
                            fm["pb_r0"].append(r0)
                            fm["pb_r1"].append(min(r0 + RBLOCK, R))
                            fm["pb_C"].append(C)
                            fm["pb_R"].append(R)
                            npb += 1
                for c in range(C):
                    fm["rok_c"].append(seg0)
                fm["fact_vroff"].append(vr_off)
                fm["fact_R"].append(R)
                ar = torch.arange(n, dtype=torch.long)
                fm["elem_pos"].append(off + ar)
                fm["elem_fact"].append(torch.full((n,), fo, dtype=torch.int32))
                vr_off += R; vc_off += C
            # Sub-segment (chunk) boundaries along dim 0. torch.chunk
            # semantics: the first D%k chunks get one extra element.
            D = int(p.shape[0]) if p.ndim else 1
            per = n // D if D else n
            base, rem = divmod(D, subsegs[i])
            ro = 0
            for j in range(subsegs[i]):
                rows_j = base + (1 if j < rem else 0)
                seg_starts.append(off + ro * per)
                seg_lens.append(rows_j * per)
                ro += rows_j
            offsets.append(off)
            off += n
        lens_t = torch.tensor(seg_lens, device=dev, dtype=torch.long)
        nseg = len(seg_lens)
        seg_idx = torch.repeat_interleave(torch.arange(nseg, device=dev), lens_t)
        starts = torch.tensor(seg_starts, device=dev, dtype=torch.long)
        ends = starts + lens_t
        seg_param = [i for i, k in enumerate(subsegs) for _ in range(k)]
        F = {
            "flat_m": flat_m, "flat_v": flat_v,
            "lengths": lengths, "lens_t": lens_t, "seg_idx": seg_idx,
            "starts": starts, "ends": ends,
            "numel_f": lens_t.to(torch.float32),
            "offsets": offsets, "total": total,
            "factored": factored_flags,
            "seg_param": seg_param,
            "seg_param_t": torch.tensor(seg_param, device=dev, dtype=torch.long),
            "param_nseg": subsegs,
            "seg_starts_l": seg_starts, "seg_lens_l": seg_lens,
        }
        if fact_idx:
            fm = fact_meta_lists
            fact_meta = {
                "row_start": torch.tensor(fm["row_start"], device=dev, dtype=torch.long),
                "row_len": torch.tensor(fm["row_len"], device=dev, dtype=torch.long),
                "row_vr": torch.tensor(fm["row_vr"], device=dev, dtype=torch.long),
                "cb_start": torch.tensor(fm["cb_start"], device=dev, dtype=torch.long),
                "cb_c0": torch.tensor(fm["cb_c0"], device=dev, dtype=torch.long),
                "cb_R": torch.tensor(fm["cb_R"], device=dev, dtype=torch.long),
                "cb_C": torch.tensor(fm["cb_C"], device=dev, dtype=torch.long),
                "cb_vc": torch.tensor(fm["cb_vc"], device=dev, dtype=torch.long),
                "pb_start": torch.tensor(fm["pb_start"], device=dev, dtype=torch.long),
                "pb_c0": torch.tensor(fm["pb_c0"], device=dev, dtype=torch.long),
                "pb_r0": torch.tensor(fm["pb_r0"], device=dev, dtype=torch.long),
                "pb_r1": torch.tensor(fm["pb_r1"], device=dev, dtype=torch.long),
                "pb_C": torch.tensor(fm["pb_C"], device=dev, dtype=torch.long),
                "pb_R": torch.tensor(fm["pb_R"], device=dev, dtype=torch.long),
                "cbf_vc": torch.tensor(fm["cbf_vc"], device=dev, dtype=torch.long),
                "cbf_p0": torch.tensor(fm["cbf_p0"], device=dev, dtype=torch.long),
                "cbf_np": torch.tensor(fm["cbf_np"], device=dev, dtype=torch.long),
                "cbf_rem": torch.tensor(fm["cbf_rem"], device=dev, dtype=torch.long),
                "rbp_start": torch.tensor(fm["rbp_start"], device=dev, dtype=torch.long),
                "rbp_c0": torch.tensor(fm["rbp_c0"], device=dev, dtype=torch.long),
                "rbp_c1": torch.tensor(fm["rbp_c1"], device=dev, dtype=torch.long),
                "rbp_n": torch.tensor(fm["rbp_n"], device=dev, dtype=torch.long),
                "rbf_j": torch.tensor(fm["rbf_j"], device=dev, dtype=torch.long),
                "rbf_p0": torch.tensor(fm["rbf_p0"], device=dev, dtype=torch.long),
                "rbf_np": torch.tensor(fm["rbf_np"], device=dev, dtype=torch.long),
                "fact_vroff": torch.tensor(fm["fact_vroff"], device=dev, dtype=torch.long),
                "fact_R": torch.tensor(fm["fact_R"], device=dev, dtype=torch.long),
                "elem_pos": torch.cat(fm["elem_pos"]).to(dev),
                "elem_fact": torch.cat(fm["elem_fact"]).to(dev),
                # Commit gating index: all row-owner segs, then all col-owner
                # segs — matching the kernel's [NR | NC] split.
                "rok": torch.tensor(fm["rok_r"] + fm["rok_c"], device=dev, dtype=torch.long),
                "fact_start": torch.tensor(
                    [offsets[i] for i in fact_idx], device=dev, dtype=torch.long),
                "fact_C": torch.tensor(cols_l, device=dev, dtype=torch.long),
                "fact_vcoff": torch.tensor(
                    [sum(cols_l[:k]) for k in range(len(cols_l))],
                    device=dev, dtype=torch.long),
                "off_to_fact": off_to_fact,
                "fact_vroff_l": fm["fact_vroff"],
                "fact_R_l": fm["fact_R"],
                "fact_vcoff_l": [sum(cols_l[:k]) for k in range(len(cols_l))],
                "fact_C_l": cols_l,
            }
            F["flat_vr"] = flat_vr; F["flat_vc"] = flat_vc
            F["cand_vr"] = torch.empty(total_r, dtype=torch.float32, device=dev)
            F["cand_vc"] = torch.empty(total_c, dtype=torch.float32, device=dev)
            F["fmean"] = torch.empty(len(fact_idx), dtype=torch.float32, device=dev)
            if npb:
                fact_meta["part"] = torch.empty(npb * CBLOCK, dtype=torch.float32, device=dev)
            if nrpb:
                fact_meta["rpart"] = torch.empty(nrpb, dtype=torch.float32, device=dev)
            F["fact_meta"] = fact_meta
        # Persistent buffers for the CUDA-graphable pre-guard pipeline.
        # Every intermediate lives here so the K1->par_ok sequence can be
        # captured once and replayed; per-step inputs are copied in.
        e = lambda *s: torch.empty(*s, dtype=torch.float32, device=dev)
        B = {
            "g": e(total), "p": e(total),
            "cand_m": e(total), "cand_v": e(total), "pp": e(total),
            "ds": e(total), "d": e(total), "scaled": e(total),
            "pre_f": e(total), "upd": e(total),
            "stats": e(5 * nseg), "stats2": e(5 * nseg),
            "ssq": e(nseg), "cnt": e(nseg),
            "seg_ok": torch.empty(nseg, dtype=torch.bool, device=dev),
            "par_ok": torch.empty(len(params), dtype=torch.bool, device=dev),
            "pn_g": e(nseg), "dn_g": e(nseg), "pre_n": e(nseg),
            "pn_par": e(len(params)), "dn_par": e(len(params)),
            "pre_n_par": e(len(params)), "n_par": e(len(params)),
            "ssq_par": e(len(params)), "am_p_par": e(len(params)),
            "am_d_par": e(len(params)), "am_m_par": e(len(params)),
            "am_pre_par": e(len(params)), "rms_par": e(len(params)),
            "rms_seg": e(nseg), "rescale": e(nseg),
            "emas_t": e(len(params)), "ema_ok": torch.empty(
                len(params), dtype=torch.bool, device=dev),
            "route_lr": e(len(params)),
            "chunked_t": torch.empty(len(params), dtype=torch.bool, device=dev),
            # Per-step dynamic values read by the captured graphs:
            # [sblend, decay_growth, trust_growth, decay_rate, agc_clip,
            #  agc_eps, trust_beta, trust_cap, lora_density_k, lr_eff,
            #  1-trust_beta, 1-decay_rate, |1-decay_rate|]
            # The last three are computed on the host in float64 so the
            # captured graph matches the eager tail's rounding exactly.
            "scalars": e(13),
            "eff": e(nseg), "ok_t": e(nseg), "tval": e(nseg),
            # Apply-graph (graph B) state.
            "qemas": e(nseg), "hasemq": torch.zeros(nseg, device=dev),
            "hasem": torch.zeros(len(params), device=dev),
            "sm": e(len(params)), "qsm": e(nseg),
            "agc_sc": e(len(params)), "qscale": torch.ones(nseg, device=dev),
            "fac": e(len(params)), "acc": torch.zeros(8, device=dev),
            "kzero": torch.zeros((), device=dev),
            "split_t": torch.empty(nseg, dtype=torch.bool, device=dev),
            "ssq2": e(len(params)), "n2": e(len(params)),
        }
        F["B"] = B
        F["graph"] = {"warm": 0, "g": None, "bad": False}
        F["graph2"] = {"warm": 0, "g": None, "bad": False}
        while len(cache) >= self._flat_cache_max:
            evicted = cache.popitem(last=False)[1]
            for egn in ("graph", "graph2"):
                eg = evicted.get(egn)
                if eg is not None:
                    eg["g"] = None
        cache[key] = F
        return F

    def stage_group_update(self, gi: int, fields: Dict[str, object], policy: str = "replace"):
        self._pending_group_updates.setdefault(int(gi), []).append((dict(fields), str(policy)))

    def _qkv_chunk_trust(
        self,
        gi: int,
        pi: int,
        p: torch.Tensor,
        d: torch.Tensor,
        *,
        dim: int,
        parts: int,
        trust_beta: float,
        trust_cap_t: torch.Tensor,
        scalar_one: torch.Tensor,
        scalar_zero: torch.Tensor,
        scalar_tiny: torch.Tensor,
        trust_denominator: Optional[torch.Tensor] = None,
        record_ema: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p_norms = _chunk_scalar_norms(p, dim, parts)
        d_norms = _chunk_scalar_norms(d, dim, parts)
        denom_norms = _chunk_scalar_norms(trust_denominator, dim, parts) if trust_denominator is not None else d_norms
        valid = (p_norms * denom_norms > scalar_zero).to(dtype=p_norms.dtype)
        raw = scalar_one + valid * (p_norms / torch.clamp(denom_norms, min=scalar_tiny) - scalar_one)
        if trust_beta > 0.0:
            key = (gi, pi)
            prev = self._qkv_trust_ema.get(key)
            if prev is not None and prev.numel() == raw.numel():
                prev = prev.to(device=raw.device, dtype=raw.dtype)
                smooth = trust_beta * prev + (1.0 - trust_beta) * raw
            else:
                smooth = raw
            if record_ema:
                self._qkv_trust_ema[key] = smooth.detach()
            return torch.minimum(trust_cap_t, smooth), d_norms
        return torch.minimum(trust_cap_t, raw), d_norms

    # ---------- Bucket stats helpers ----------
    def _bucket_global_rms(self, updates: List[torch.Tensor], source="global", *, reference: Optional[torch.Tensor] = None):
        ref_tensor = reference if reference is not None else (updates[0] if updates else None)
        if not updates:
            ref_device = ref_tensor.device if ref_tensor is not None else None
            return _scalar_like(ref_tensor, 1.0, dtype=torch.float32, device=ref_device)
        # An empty update has no RMS and must not turn the mean/median bucket
        # scale into NaN for its nonempty peers.
        ups32 = [u.to(torch.float32) for u in updates if u.numel()]
        if not ups32:
            return _scalar_like(ref_tensor, 1.0, dtype=torch.float32,
                                device=ref_tensor.device if ref_tensor is not None else None)
        if source == "global":
            ss = torch.stack([torch.sum(u*u) for u in ups32]).sum()
            ne = _scalar_like(ups32[0], float(sum(int(u.numel()) for u in ups32)), dtype=torch.float32)
            rms = torch.sqrt(ss / torch.clamp(ne, min=_scalar_like(ne, 1.0, device=ne.device)))
        elif source == "median":
            rms_list = [torch.sqrt(torch.mean(u*u)) for u in ups32]
            rms = _median_tensor(rms_list, reference=ref_tensor)
        else:
            rms_list = torch.stack([torch.sqrt(torch.mean(u*u)) for u in ups32])
            rms = torch.mean(rms_list)
        return torch.clamp(rms, min=_scalar_like(rms, 1e-12, device=rms.device))

    def _bucket_stats_triton(self, updates: List[torch.Tensor], params: List[torch.Tensor]):
        try:
            from .triton_kernels import bucket_stats_triton
            return bucket_stats_triton(updates, params)  # (ussq, pssq, n)
        except Exception as exc:
            if _LOG.isEnabledFor(logging.DEBUG):
                _LOG.debug("Triton bucket stats fallback", exc_info=exc)
            ups32 = [u.to(torch.float32) for u in updates]
            ps32  = [p.detach().to(torch.float32) for p in params]
            ussq = torch.stack([torch.sum(u*u) for u in ups32]).sum()
            pssq = torch.stack([torch.sum(p*p) for p in ps32]).sum()
            ref_tensor = ups32[0] if ups32 else (ps32[0] if ps32 else None)
            n = _scalar_like(ref_tensor, float(sum(int(u.numel()) for u in ups32)), dtype=torch.float32)
            return ussq, pssq, n

    def _fused_apply_triton(
        self,
        params: List[torch.Tensor],
        updates: List[torch.Tensor],
        diagnostics: Optional[Dict[str, object]] = None,
    ):
        def record(
            status: str,
            *,
            detail: Optional[str] = None,
            exc: Optional[BaseException] = None,
        ) -> None:
            reason_payload: Optional[str] = None
            if exc is not None:
                reason_payload = str(exc)
            elif detail is not None:
                reason_payload = detail
            if diagnostics is not None:
                diagnostics["triton_fused_apply"] = status
                if reason_payload is not None:
                    diagnostics["triton_fused_apply_reason"] = reason_payload
                else:
                    diagnostics.setdefault("triton_fused_apply_reason", status)
            if _LOG.isEnabledFor(logging.DEBUG):
                message = status if detail is None else f"{status}: {detail}"
                if exc is not None:
                    _LOG.debug("Triton fused apply fallback (%s)", message, exc_info=exc)
                else:
                    _LOG.debug("Triton fused apply fallback (%s)", message)
            reason = status if reason_payload is None else f"{status}:{reason_payload}"
            if hasattr(self, "_fused_apply_failures"):
                self._fused_apply_failures.append(reason)

        if not params or not updates:
            record("empty")
            return False

        device = params[0].device
        if device.type != "cuda":
            record("non_cuda_param")
            return False

        if not torch.cuda.is_available():
            record("cuda_unavailable")
            return False

        if len(params) != len(updates):
            record("length_mismatch")
            return False

        if any(p.device != device for p in params):
            record("param_device_mismatch")
            return False

        if any(u.device != device for u in updates):
            record("update_device_mismatch")
            return False

        try:
            from .triton_kernels import fused_apply_updates
        except Exception as exc:
            record("import_error", exc=exc)
            return False

        try:
            kernel_result = fused_apply_updates(params, updates)
        except Exception as exc:
            record("kernel_exception", exc=exc)
            return False
        kernel_ok: bool
        kernel_detail: Optional[str]
        if isinstance(kernel_result, tuple):
            if len(kernel_result) == 0:
                kernel_ok = False
                kernel_detail = "empty_result"
            else:
                kernel_ok = bool(kernel_result[0])
                kernel_detail = str(kernel_result[1]) if len(kernel_result) > 1 else None
        else:
            kernel_ok = bool(kernel_result)
            kernel_detail = None if kernel_ok else "kernel_returned_false"

        if not kernel_ok:
            record("kernel_declined", detail=kernel_detail)
            return False

        if diagnostics is not None:
            diagnostics["triton_fused_apply"] = "ok"
            diagnostics["triton_fused_apply_reason"] = kernel_detail or "ok"
        if _LOG.isEnabledFor(logging.DEBUG):
            if kernel_detail:
                _LOG.debug(
                    "Triton fused apply succeeded on %d tensors (%s)",
                    len(params),
                    kernel_detail,
                )
            else:
                _LOG.debug(
                    "Triton fused apply succeeded on %d tensors",
                    len(params),
                )
        return True

    # ---------- Internal helpers ----------
    def _maybe_plateau_adapt(self):
        d = self.defaults
        p = self._plateau
        if p["best"] is None: return
        if p["streak"] >= d["plateau_patience"]:
            adapted = False
            if d["auto_lr"]:
                lr_changed = False
                for g in self.param_groups:
                    old_lr = float(g["lr"])
                    new_lr = max(float(d["lr_min"]), old_lr * float(d["lr_decay"]))
                    if new_lr < old_lr:
                        g["lr"] = new_lr
                        lr_changed = True
                if lr_changed:
                    self._last_metrics["auto_lr_decay"] = 1.0
                    adapted = True
            if d["auto_blend"] and d["blend_steps"] <= 0:
                _, hi = d["auto_blend_bounds"]
                for g in self.param_groups:
                    sb = float(g.get("sign_blend", d["sign_blend"]))
                    g["sign_blend"] = min(hi, sb + d["auto_blend_gain"])
                self._last_metrics["auto_blend_bump"] = 1.0
                adapted = True
            if adapted:
                p["streak"] = 0

    def _dtype_for_update(self, p: torch.Tensor) -> torch.dtype:
        mode = str(self.defaults.get("update_buffer_dtype","param"))
        if mode == "param": return p.dtype
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(mode, p.dtype)

    # ---------- Main step ----------
    @torch.no_grad()
    def step(self, closure=None):
        t0 = time.perf_counter()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        active_params_by_group: List[List[torch.Tensor]] = [
            [p for p in group["params"] if p.grad is not None]
            for group in self.param_groups
        ]
        if self.defaults["skip_if_nonfinite"]:
            for group, active_params in zip(self.param_groups, active_params_by_group):
                foreach_updates = bool(group.get("use_foreach_update", self.defaults["use_foreach_update"])) and bool(group.get("use_foreach", self.defaults["use_foreach"]))
                min_bucket = int(group.get("foreach_min_bucket", self.defaults["foreach_min_bucket"]))
                if (foreach_updates and bool(group.get("bucket_standardize", self.defaults["bucket_standardize"]))
                        and str(group.get("bucket_standardize_source", self.defaults["bucket_standardize_source"])).lower() == "median"
                        and len(active_params) >= min_bucket
                        and any(p.dtype in (torch.float16, torch.bfloat16) for p in active_params)):
                    raise ValueError("median bucket standardization with low-precision parameters and skip_if_nonfinite is unsupported")

        self._global_step += 1
        self._last_metrics.pop("auto_blend_bump", None)
        self._last_metrics.pop("auto_lr_decay", None)

        # staged reflect cadence
        if (not _is_compiling()) and self.defaults.get("compile_guard", True):
            if (self._pending_lr_scale or self._pending_group_updates) and (self._global_step - self._last_reflect >= int(self.defaults["reflect_interval"])):
                self.reflect_pending()

        # profiler counters
        profiler_enabled = bool(self._prof.enabled)
        profiler_detail_enabled = profiler_enabled and ((self._global_step % self._prof.interval) == 0)
        prof_c = dict(foreach_wd=0, scalar_wd=0, foreach_update=0, foreach_update_tensors=0,
                      foreach_bucket_size=0, foreach_bucket_global_rms=None, foreach_bucket_trust_est=None, foreach_bucket_source=None,
                      triton_fused_apply=None, triton_fused_apply_reason=None,
                      agc_clips=0, nonfinite_skips=0, pruned_params=0, pruned_elems=0,
                      foreach_triton_failures=None,
                      qkv_q_r=None, qkv_k_r=None, qkv_v_r=None, qkv_q_rms=None, qkv_k_rms=None, qkv_v_rms=None,
                      qkv_q_freq=None, qkv_k_freq=None, qkv_v_freq=None,
                      qkv_gamma_eff=None, qkv_disp=None, qkv_disp_ema=None, qkv_step_clip_eff=None, qkv_accel=None,
                      qkv_freq_disp=None, qkv_freq_factor=None, qkv_phase_boost=None)

        self._fused_apply_failures = []

        # sparse state pruning flags
        sprune_thr = float(self.defaults["state_prune_threshold"])
        sprune_int = int(self.defaults["state_prune_interval"])
        do_prune = (sprune_thr > 0.0) and (sprune_int > 0) and (self._global_step % sprune_int == 0)
        if self.defaults["skip_if_nonfinite"]:
            for gi, active_params in enumerate(active_params_by_group):
                if not active_params:
                    continue
                # Squaring a finite FP32 gradient can overflow the second moment.
                # Probe the entire group first so the common path needs one sync.
                probe = _foreach_all_finite([
                    *active_params,
                    *[_to_dtype(p.grad, torch.float32).square() for p in active_params],
                ])
                if probe is True:
                    continue
                valid = []
                for p in active_params:
                    grad_square = _to_dtype(p.grad, torch.float32).square()
                    if bool(torch.isfinite(p).all() and torch.isfinite(grad_square).all()):
                        valid.append(p)
                    else:
                        prof_c["nonfinite_skips"] += 1
                active_params_by_group[gi] = valid

        # Without the skip policy, decay can be batched before the main pass.
        # Under the skip policy it is applied only after candidate moments pass
        # validation so a rejected parameter remains fully unchanged.
        if not self.defaults["skip_if_nonfinite"]:
            for g, active_params in zip(self.param_groups, active_params_by_group):
                wd = float(g["weight_decay"]); lr = float(g["lr"]); lr_scale = float(g.get("lr_scale", 1.0))
                if wd == 0.0 or not active_params:
                    continue
                if g["use_foreach"] and hasattr(torch, "_foreach_add_"):
                    torch._foreach_add_(active_params, active_params, alpha=-(lr * lr_scale * wd))
                    prof_c["foreach_wd"] += len(active_params)
                else:
                    for p in active_params:
                        p.add_(p, alpha=-(lr * lr_scale * wd))

        # ---------- Main pass ----------
        up_s2 = 0.0; up_n = 0
        dn_s2 = 0.0; dn_n = 0
        cross_enabled = bool(self.defaults.get("lora_cross_adapt", False))
        cross_state = self._lora_cross_state
        cross_beta = float(self.defaults.get("lora_cross_beta", 0.0)) if cross_enabled else 0.0
        cross_sync = float(self.defaults.get("lora_cross_sync", 0.0)) if cross_enabled else 0.0
        cross_gain = float(self.defaults.get("lora_cross_gain", 0.0)) if cross_enabled else 0.0
        cross_attn_tags = tuple(self.defaults.get("lora_cross_attn_tags", ())) if cross_enabled else tuple()
        cross_ffn_tags = tuple(self.defaults.get("lora_cross_ffn_tags", ())) if cross_enabled else tuple()
        cross_tag_set = set(cross_attn_tags) | set(cross_ffn_tags)
        pending_lora_blocks: List[Tuple[int, str, Dict[str, float], float]] = []
        ffn_asym_enabled = any(g.get("auto_ffn_asym", self.defaults["auto_ffn_asym"]) for g in self.param_groups)

        for gi, (group, active_params) in enumerate(zip(self.param_groups, active_params_by_group)):
            if not active_params:
                continue
            param_indices = {id(param): index for index, param in enumerate(group["params"])}
            group_ref = active_params[0]
            lr = float(group["lr"]); (b1,b2) = group["betas"]; eps = float(group["eps"])
            factored = bool(group["factored"]); alpha = float(group["precond_alpha"])
            sblend0 = float(group["sign_blend"]); sblendT = int(group["blend_steps"]); sblendTo = float(group["blend_to"]); sched = group["blend_schedule"]
            use_trust = bool(group["use_trust_ratio"]); trust_clip = float(group["trust_clip"]); trust_space = str(group.get("trust_space","update"))
            trust_beta = float(group.get("trust_ema_beta", self.defaults["trust_ema_beta"]))
            agc_clip = float(group["agc_clip"]); agc_eps = float(group["agc_eps"])
            scalar_zero = self._group_scalar(gi, "zero", group_ref, 0.0)
            scalar_one = self._group_scalar(gi, "one", group_ref, 1.0)
            scalar_tiny = self._group_scalar(gi, "tiny", group_ref, 1e-12)
            scalar_vtiny = self._group_scalar(gi, "vtiny", group_ref, 1e-30)
            eps_t = self._group_scalar(gi, "eps", group_ref, eps)
            agc_clip_t = self._group_scalar(gi, "agc_clip", group_ref, agc_clip) if agc_clip and agc_clip > 0.0 else None
            trust_cap_t = self._group_scalar(gi, "trust_clip", group_ref, trust_clip)
            agc_eps_t = self._group_scalar(gi, "agc_eps", group_ref, agc_eps)
            lr_scale = float(group.get("lr_scale", 1.0)); tag = group.get("block_tag","default")
            rms_thr = float(group.get("rms_clip_threshold", 0.0)); rms_gran = group.get("rms_clip_granularity","param")
            rms_thr_t = self._group_scalar(gi, "rms_thr", group_ref, rms_thr) if rms_thr else None
            qkv_rules = group.get("qkv_rules") or {}; qkv_split = bool(group.get("qkv_trust_split", False))
            qkv_lr = group.get("qkv_lr_scales")
            blk_chunks = int(group.get("block_trust_chunks", 0))
            use_foreach_upd = bool(group.get("use_foreach_update", self.defaults["use_foreach_update"])) and bool(group.get("use_foreach", self.defaults["use_foreach"]))
            foreach_min_bucket = int(group.get("foreach_min_bucket", self.defaults["foreach_min_bucket"]))
            bucket_std = bool(group.get("bucket_standardize", self.defaults["bucket_standardize"]))
            bucket_src = str(group.get("bucket_standardize_source", self.defaults["bucket_standardize_source"])).lower()
            bucket_scalarless = bool(group.get("bucket_scalarless", self.defaults["bucket_scalarless"]))
            use_triton_stats = bool(group.get("use_triton_bucket_stats", self.defaults["use_triton_bucket_stats"]))
            use_triton_fused = bool(group.get("use_triton_fused_apply", self.defaults["use_triton_fused_apply"]))
            spec_strength = self.qkv_spectral_strength
            spec_enabled = bool(self.defaults.get("qkv_spectral_adapt", False)) and spec_strength > 0.0
            spec_low_band = float(self.defaults.get("qkv_spectral_low_band", 0.2))
            spec_high_band = float(self.defaults.get("qkv_spectral_high_band", 0.25))
            spec_beta = float(self.defaults.get("qkv_spectral_beta", 0.7))
            spec_eps = float(self.defaults.get("qkv_spectral_eps", 1e-9))
            spec_eps_t = self._group_scalar(gi, "spec_eps", group_ref, spec_eps) if spec_enabled else None
            spec_state = self._qkv_spec_ema.setdefault(gi, {}) if spec_enabled else None
            qkv_adapt_due = (
                tag == "attn_qkv"
                and qkv_lr is not None
                and bool(self.defaults["qkv_lr_autoadapt"])
                and (self._global_step % int(self.defaults["qkv_lr_interval"]) == 0)
            )
            capture_qkv_stats = profiler_detail_enabled or qkv_adapt_due
            collect_qkv_spectral = capture_qkv_stats and spec_enabled and spec_state is not None

            # schedule override for sign_blend
            if sblendT and sblendT>0:
                c = min(self._global_step/float(sblendT), 1.0)
                w = (0.5 - 0.5*math.cos(math.pi*c)) if sched=="cosine" else c
                sblend = (1.0 - w)*sblend0 + w*sblendTo
            else:
                sblend = sblend0

            # foreach buckets
            b_params: List[torch.Tensor] = []
            b_updates: List[torch.Tensor] = []
            b_updates_f32: List[torch.Tensor] = []

            # LoRA EMA/PID accumulators
            lora_dense_sum = 0.0; lora_count = 0
            lora_dense_sum_t: Optional[torch.Tensor] = scalar_zero.clone() if tag in ("lora_a", "lora_b") and bool(self.defaults["lora_density_adapt"]) else None
            cross_density_sum_t: Optional[torch.Tensor] = scalar_zero.clone() if cross_enabled and tag in cross_tag_set else None
            up_s2_t: Optional[torch.Tensor] = scalar_zero.clone() if ffn_asym_enabled and tag == "ffn_up" else None
            dn_s2_t: Optional[torch.Tensor] = scalar_zero.clone() if ffn_asym_enabled and tag == "ffn_down" else None
            agc_clips_t: Optional[torch.Tensor] = scalar_zero.clone() if profiler_detail_enabled and agc_clip and agc_clip > 0.0 else None

            # Collect every fused QKV tensor before adapting the shared group scales.
            # The order of parameters in a group must not select the observation.
            qkv_observations = (
                {label: [] for label in ("q", "k", "v")}
                if capture_qkv_stats else None
            )
            cross_density_sum = 0.0; cross_density_count = 0

            # ---- Fused flat fast path (opt-in via TIGER_FLAT_STEP) ----
            # "Simple" params (fp32 p/g/state, dense v, softsign, alpha>0,
            # no qkv/chunk/rms-clip) run on 5 flat Triton launches + 1 sync.
            skip_checks_g = self.defaults["skip_if_nonfinite"]
            fast_params: List[torch.Tensor] = []
            slow_params: List[torch.Tensor] = list(active_params)
            if (os.environ.get("TIGER_FLAT_STEP", "") not in ("", "0", "false", "False")
                    and group_ref.is_cuda
                    and skip_checks_g
                    and group["sign_mode"] == "softsign"
                    and alpha > 0.0
                    and self._mom_storage_dtype == torch.float32
                    and float(self.defaults["state_prune_threshold"]) == 0.0
                    and not (rms_thr and rms_gran == "param")):
                try:
                    from .triton_kernels import (
                        flat_step_candidates, flat_seg_stats, flat_direction,
                        flat_apply, flat_fused_supported,
                    )
                    if flat_fused_supported():
                        _elig = []
                        _fact_flags = []
                        _subsegs = []
                        _chunked = []
                        for p in active_params:
                            if not (p.dtype == torch.float32
                                    and p.grad is not None and p.grad.dtype == torch.float32
                                    and p.is_contiguous() and p.grad.is_contiguous()
                                    and self.state[p].get("_state_sane")
                                    and "m" in self.state[p]):
                                continue
                            # Chunked routes map to sub-segments along dim 0
                            # (the only dim whose chunks are contiguous). A
                            # qkv rule without an active route needs no split;
                            # stats collection keeps the param on the slow path.
                            rule = qkv_rules.get(id(p))
                            route = rule is not None and (
                                qkv_lr is not None or (qkv_split and use_trust))
                            nseg = 1; ch = False
                            if route:
                                if (capture_qkv_stats or rule[0] != 0
                                        or p.shape[0] % rule[1] != 0):
                                    continue
                                nseg = rule[1]; ch = True
                            if (blk_chunks > 0 and p.ndim == 2
                                    and p.shape[0] % blk_chunks == 0):
                                ch = True
                            if factored and p.ndim >= 2 and p.shape[0] > 0:
                                vr = self.state[p].get("vr"); vc = self.state[p].get("vc")
                                if (vr is None or vc is None
                                        or vr.dtype != torch.float32 or vc.dtype != torch.float32
                                        or not vr.is_contiguous() or not vc.is_contiguous()):
                                    continue
                                _elig.append(p); _fact_flags.append(True)
                            elif "v" in self.state[p]:
                                _elig.append(p); _fact_flags.append(False)
                            else:
                                continue
                            _subsegs.append(nseg); _chunked.append(ch)
                        # Dense params first so flat_v maps cleanly.
                        _order = sorted(range(len(_elig)), key=lambda i: _fact_flags[i])
                        fast_params = [_elig[i] for i in _order]
                        fact_flags = [_fact_flags[i] for i in _order]
                        subseg_counts = [_subsegs[i] for i in _order]
                        chunked_flags = [_chunked[i] for i in _order]
                        slow_params = [p for p in active_params
                                       if not any(p is q for q in fast_params)]
                except Exception:
                    fast_params = []
                    slow_params = list(active_params)

            fast_ok_list: List[bool] = []
            fast_ctx: Optional[Dict[str, object]] = None
            if fast_params:
                from .triton_kernels import (
                    flat_step_candidates, flat_seg_stats, flat_direction, flat_apply,
                    flat_seg_ssq,
                )
                F = self._flat_group(gi, fast_params, group_ref,
                                     factored_flags=fact_flags, subsegs=subseg_counts)
                dev_f = group_ref.device
                Nf = len(fast_params)
                Nseg = F["lens_t"].numel()
                seg = F["seg_idx"]; lens = F["lens_t"]; numel_f = F["numel_f"]
                seg_par = F["seg_param_t"]
                total = F["total"]
                B = F["B"]
                flat_g = B["g"]; flat_p = B["p"]
                cand_m = B["cand_m"]; cand_v = B["cand_v"]
                pp_b = B["pp"]; ds_b = B["ds"]
                # Copy this step's inputs into the persistent buffers (the
                # graph can only replay fixed addresses).
                torch._foreach_copy_(
                    [flat_g[o:o + n] for o, n in zip(F["offsets"], F["lengths"])],
                    [p.grad.detach().reshape(-1) for p in fast_params])
                torch._foreach_copy_(
                    [flat_p[o:o + n] for o, n in zip(F["offsets"], F["lengths"])],
                    [p.detach().reshape(-1) for p in fast_params])
                id_seg = Nseg == Nf
                decay_rate = lr * lr_scale * float(group["weight_decay"])
                decay_growth = abs(1.0 - decay_rate) + (
                    8.0 * torch.finfo(torch.float32).eps * abs(decay_rate))
                trust_growth = max(1.0, trust_clip) if use_trust else 1.0
                # Dynamic scalars -> device buffer (graph reads live values).
                B["scalars"].copy_(torch.tensor(
                    [sblend, decay_growth, trust_growth, decay_rate,
                     agc_clip if agc_clip and agc_clip > 0.0 else 0.0,
                     agc_eps, trust_beta, trust_clip,
                     float(self.defaults["lora_density_k"]), lr * lr_scale,
                     1.0 - trust_beta, 1.0 - decay_rate,
                     abs(1.0 - decay_rate)],
                    dtype=torch.float32))
                has_qkv_lr = any(
                    qkv_rules.get(id(p)) is not None and qkv_lr is not None
                    for p in fast_params)
                if has_qkv_lr:
                    _rls = []
                    for p in fast_params:
                        rls = [abs(lr * lr_scale)]
                        if qkv_rules.get(id(p)) is not None and qkv_lr is not None:
                            rls.extend(abs(lr * lr_scale * float(qkv_lr.get(k, 1.0)))
                                       for k in ("q", "k", "v"))
                        _rls.append(max(rls) if all(math.isfinite(v) for v in rls) else math.inf)
                    B["route_lr"].copy_(torch.tensor(_rls, dtype=torch.float32))
                else:
                    B["route_lr"].fill_(abs(lr * lr_scale))
                B["chunked_t"].copy_(torch.tensor(chunked_flags))
                bucket_route = (
                    use_foreach_upd and bucket_std and hasattr(torch, "_foreach_add_")
                    and len(active_params) >= foreach_min_bucket)
                emas = has_ema_idx = None
                if use_trust and trust_beta > 0.0:
                    _emas = [
                        (None if (chunked_flags[i] and qkv_split
                                  and qkv_rules.get(id(p)) is not None)
                         else self._trust_ema.get((gi, param_indices[id(p)])))
                        for i, p in enumerate(fast_params)]
                    has_ema_idx = [i for i, o in enumerate(_emas) if o is not None]
                    emas = torch.stack([
                        (o.to(torch.float32).reshape(()) if isinstance(o, torch.Tensor)
                         else _scalar_like(group_ref, float(o), dtype=torch.float32, device=dev_f) if o is not None
                         else torch.ones((), device=dev_f)) for o in _emas])
                    B["emas_t"].copy_(emas)
                    _hasem = torch.zeros(Nf, dtype=torch.float32)
                    _hasem[has_ema_idx] = 1.0
                    B["hasem"].copy_(_hasem)
                split_list = [bool(chunked_flags[i] and qkv_split
                                   and qkv_rules.get(id(fast_params[i])) is not None)
                              for i in range(Nf)]
                # Per-seg split flags + per-seg qkv-EMA load for graph B.
                _ss = []
                _qh = []
                s0 = 0
                _qkv_ema_ok = use_trust and trust_beta > 0.0 and any(split_list)
                for i, p in enumerate(fast_params):
                    k = F["param_nseg"][i]
                    _ss.extend([split_list[i]] * k)
                    if _qkv_ema_ok and split_list[i]:
                        prev = self._qkv_trust_ema.get(
                            (gi, param_indices[id(p)]))
                        if prev is not None and prev.numel() == k:
                            B["qemas"][s0:s0 + k].copy_(
                                prev.to(torch.float32).reshape(-1))
                            _qh.extend([1.0] * k)
                        else:
                            _qh.extend([0.0] * k)
                    else:
                        _qh.extend([0.0] * k)
                    s0 += k
                B["split_t"].copy_(torch.tensor(_ss))
                if _qkv_ema_ok:
                    B["hasemq"].copy_(torch.tensor(_qh, dtype=torch.float32))
                if has_qkv_lr:
                    _keys3 = ("q", "k", "v")
                    _qs = []
                    for i, p in enumerate(fast_params):
                        k = F["param_nseg"][i]
                        if qkv_rules.get(id(p)) is not None:
                            _qs.extend(float(qkv_lr.get(_keys3[j], 1.0))
                                       if j < 3 else 1.0 for j in range(k))
                        else:
                            _qs.extend([1.0] * k)
                    B["qscale"].copy_(torch.tensor(_qs, dtype=torch.float32))
                else:
                    B["qscale"].fill_(1.0)

                def _preguard():
                    flat_step_candidates(flat_g, F["flat_m"], F["flat_v"], cand_m, cand_v,
                                         pp_b, ds_b, b1=b1, b2=b2, eps=eps, alpha=alpha,
                                         sign_tau=float(group["sign_tau"]))
                    # Factored preamble (all async): candidate vr/vc by
                    # segmented row/col reduces, then rank-1 vhat + pp/dsign
                    # rewritten into the shared flat buffers so the K2-K5
                    # tail stays uniform. Nonfinite cand_vr/vc propagate into
                    # vhat and are caught by the K2 amax_v guard.
                    if F.get("fact_meta") is not None:
                        from .triton_kernels import flat_fact_moments
                        flat_fact_moments(
                            flat_g, F["flat_vr"], F["flat_vc"], F["cand_vr"],
                            F["cand_vc"], F["fmean"], F["fact_meta"],
                            cand_m, cand_v, pp_b, ds_b,
                            b2=b2, eps=eps, alpha=alpha,
                            sign_tau=float(group["sign_tau"]), vtiny=1e-30)
                    stats = B["stats"]
                    flat_seg_stats(cand_m, cand_v, flat_p, ds_b, F["starts"], F["ends"], stats)
                    st5 = stats.view(Nseg, 5)
                    am_m = st5[:, 0]; am_v = st5[:, 1]; am_p = st5[:, 2]
                    am_ds = st5[:, 3]; ssq_m = st5[:, 4]
                    if id_seg:
                        rms_par = am_m * torch.sqrt(ssq_m / numel_f)
                        rms_seg = rms_par
                        n_par = numel_f
                        am_p_par = am_p
                        am_d_par = None  # filled after stats2
                    else:
                        # Param-level _rms(m): recombine amax-scaled per-seg
                        # sums, rescaled by the param amax so finite norms
                        # cannot overflow.
                        am_m_par = B["am_m_par"].zero_().scatter_reduce_(
                            0, seg_par, am_m, reduce="amax")
                        am_m_safe = torch.where(
                            torch.isfinite(am_m_par) & (am_m_par > 0),
                            am_m_par, torch.ones_like(am_m_par))
                        rescale = (am_m / am_m_safe[seg_par]).square()
                        ssq_par = B["ssq_par"].zero_().index_add_(0, seg_par, ssq_m * rescale)
                        n_par = B["n_par"].zero_().index_add_(0, seg_par, numel_f)
                        rms_par = am_m_safe * torch.sqrt(ssq_par / n_par)
                        rms_seg = rms_par[seg_par]
                        am_p_par = B["am_p_par"].zero_().scatter_reduce_(
                            0, seg_par, am_p, reduce="amax")
                    d_f = B["d"]
                    flat_direction(cand_m, pp_b, ds_b, rms_seg, seg, d_f,
                                   sblend=B["scalars"][0], eps=eps)
                    stats2 = B["stats2"]
                    flat_seg_stats(cand_m, cand_v, flat_p, d_f, F["starts"], F["ends"], stats2)
                    am_d = stats2.view(Nseg, 5)[:, 3]

                    # Finite checks are per-seg; magnitude bounds are
                    # param-level (eager checks the whole tensor).
                    seg_ok = (torch.isfinite(am_m) & torch.isfinite(am_v)
                              & torch.isfinite(am_d) & torch.isfinite(am_ds))
                    if id_seg:
                        am_d_par = am_d
                    else:
                        am_d_par = B["am_d_par"].zero_().scatter_reduce_(
                            0, seg_par, am_d, reduce="amax")
                    raw_bound_par = am_d_par * (B["scalars"][2] * B["route_lr"])
                    if bucket_route:
                        if bucket_src == "median" and not use_triton_stats:
                            upd_par = raw_bound_par * 1e12
                        elif use_triton_stats or bucket_src == "global":
                            upd_par = raw_bound_par.new_full((), math.sqrt(sum(q.numel() for q in active_params)))
                        else:
                            upd_par = raw_bound_par.new_full((), len(active_params) * 1.0) * n_par.sqrt()
                        # Chunked params never enter the foreach bucket.
                        if any(chunked_flags):
                            upd_bound_par = torch.where(B["chunked_t"], raw_bound_par, upd_par)
                        else:
                            upd_bound_par = upd_par
                    else:
                        upd_bound_par = raw_bound_par
                    fp32_limit = torch.finfo(torch.float32).max * (1.0 - 8.0 * torch.finfo(torch.float32).eps)
                    par_ok = (
                        (am_d_par * B["scalars"][2] <= fp32_limit)
                        & (raw_bound_par <= fp32_limit)
                        & (am_p_par * B["scalars"][1] + upd_bound_par <= fp32_limit))

                    pn_par = dn_par = pre_n_par = None
                    if use_trust or (agc_clip and agc_clip > 0.0):
                        ssq_buf = B["ssq"]

                        def _seg_norm(x, am):
                            # Segmented norm via a deterministic per-seg
                            # kernel — a dense indicator GEMV would let one
                            # param's NaN poison every segment (0*NaN=NaN).
                            safe = torch.where(torch.isfinite(am) & (am > 0), am, torch.ones_like(am))
                            # Gather via seg_idx (repeat_interleave would
                            # sync for its output size — illegal in capture).
                            scaled = B["scaled"].copy_(x / safe[seg])
                            flat_seg_ssq(scaled, F["starts"], F["ends"], ssq_buf)
                            return safe * ssq_buf.sqrt()
                        pn_g = _seg_norm(flat_p, am_p)
                        dn_g = _seg_norm(d_f, am_d)
                        if id_seg:
                            pn_par = pn_g; dn_par = dn_g
                        else:
                            def _combine_norm(x):
                                mx = torch.zeros(Nf, device=dev_f).scatter_reduce(
                                    0, seg_par, x, reduce="amax")
                                mx = torch.where(torch.isfinite(mx) & (mx > 0),
                                                 mx, torch.ones_like(mx))
                                s = torch.zeros(Nf, device=dev_f).index_add_(
                                    0, seg_par, (x / mx[seg_par]).square())
                                return mx * s.sqrt()
                            pn_par = _combine_norm(pn_g)
                            dn_par = _combine_norm(dn_g)
                        loose_ok = (am_p_par * (B["scalars"][1] * n_par.sqrt()) <= fp32_limit) & (am_d_par * n_par.sqrt() <= fp32_limit)
                        exact_ok = (pn_par * B["scalars"][1] <= fp32_limit) & (dn_par <= fp32_limit)
                        if use_trust and trust_space == "precond":
                            pre_f = B["pre_f"].copy_(cand_m * pp_b)
                            am_pre = torch.stack(list(torch._foreach_norm(
                                [pre_f[s:s + l] for s, l in zip(F["seg_starts_l"], F["seg_lens_l"])],
                                ord=float("inf"))))
                            pre_n = _seg_norm(pre_f, am_pre)
                            if id_seg:
                                am_pre_par = am_pre; pre_n_par = pre_n
                            else:
                                am_pre_par = torch.zeros(Nf, device=dev_f).scatter_reduce(
                                    0, seg_par, am_pre, reduce="amax")
                                pre_n_par = _combine_norm(pre_n)
                            loose_ok &= am_pre_par * n_par.sqrt() <= fp32_limit
                            exact_ok &= pre_n_par <= fp32_limit
                        par_ok &= loose_ok | exact_ok
                        B["pn_g"].copy_(pn_g); B["dn_g"].copy_(dn_g)
                        B["pn_par"].copy_(pn_par); B["dn_par"].copy_(dn_par)
                        if pre_n_par is not None:
                            B["pre_n"].copy_(pre_n); B["pre_n_par"].copy_(pre_n_par)
                        else:
                            B["pre_n"].zero_(); B["pre_n_par"].zero_()

                    if use_trust and trust_beta > 0.0:
                        par_ok &= torch.isfinite(B["emas_t"]) & (B["emas_t"] >= 0)
                    if id_seg:
                        par_ok &= seg_ok
                    else:
                        seg_ok_par = torch.ones(Nf, dtype=torch.float32, device=dev_f)
                        seg_ok_par.scatter_reduce_(0, seg_par, seg_ok.to(torch.float32), reduce="amin")
                        par_ok &= seg_ok_par > 0.5
                    B["par_ok"].copy_(par_ok)

                # Capture-or-replay: the whole K1->par_ok pipeline runs as one
                # CUDA graph after two eager warmup steps. Any failure falls
                # back to the plain eager launches permanently.
                gs = F["graph"]
                _gkey = (b1, b2, eps, alpha, float(group["sign_tau"]),
                         use_trust, trust_space, trust_beta > 0.0,
                         bool(agc_clip and agc_clip > 0.0),
                         bucket_route, bucket_src, use_triton_stats,
                         id_seg, has_qkv_lr, qkv_split,
                         tuple(chunked_flags), len(active_params),
                         sum(q.numel() for q in active_params))
                if gs["g"] is not None and gs["key"] != _gkey:
                    gs["g"] = None; gs["warm"] = 0
                if gs["g"] is not None:
                    gs["g"].replay()
                elif gs["bad"]:
                    _preguard()
                else:
                    if gs["warm"] >= 2:
                        try:
                            # Sharing one pool across graphs is safe here
                            # because _preguard outputs only into persistent
                            # pre-capture buffers — no capture-pool tensor
                            # escapes — and replays never run concurrently.
                            pool = self._graph_pools.get(group_ref.device)
                            if pool is None:
                                pool = torch.cuda.graph_pool_handle()
                                self._graph_pools[group_ref.device] = pool
                            gcap = torch.cuda.CUDAGraph()
                            with torch.cuda.device(group_ref.device):
                                with torch.cuda.graph(gcap, pool=pool):
                                    _preguard()
                            gs["g"] = gcap; gs["key"] = _gkey
                            gs["g"].replay()
                        except Exception as _ge:
                            gs["bad"] = True; gs["g"] = None; gs["err"] = repr(_ge)
                            _preguard()
                    else:
                        gs["warm"] += 1
                        _preguard()

                # Vector-EMA check for split-trust params (host dict access —
                # kept outside the graph; two small ops).
                par_ok = B["par_ok"]
                if use_trust and trust_beta > 0.0 and qkv_split:
                    qok = []; qidx = []
                    for i, p in enumerate(fast_params):
                        if not (chunked_flags[i]
                                and qkv_rules.get(id(p)) is not None):
                            continue
                        prev = self._qkv_trust_ema.get((gi, param_indices[id(p)]))
                        if prev is None:
                            continue
                        qidx.append(i)
                        prev_f = prev.reshape(-1).to(torch.float32)
                        qok.append((torch.isfinite(prev_f) & (prev_f >= 0)).all())
                    if qok:
                        par_ok = par_ok.clone()
                        par_ok[torch.tensor(qidx, device=dev_f)] &= torch.stack(
                            [q.reshape(()) for q in qok])
                fast_ok_list = par_ok.cpu().tolist()
                _norms = use_trust or (agc_clip and agc_clip > 0.0)
                _precond = use_trust and trust_space == "precond" and _norms
                fast_ctx = {
                    "F": F, "flat_g": flat_g, "flat_p": flat_p, "cand_m": cand_m,
                    "cand_v": cand_v, "pp": pp_b, "ds": ds_b, "d": B["d"],
                    "decay_rate": decay_rate,
                    "pn_g": B["pn_g"] if _norms else None,
                    "dn_g": B["dn_g"] if _norms else None,
                    "pre_n": B["pre_n"] if _precond else None,
                    "pn_par": B["pn_par"] if _norms else None,
                    "dn_par": B["dn_par"] if _norms else None,
                    "pre_n_par": B["pre_n_par"] if _precond else None,
                    "emas": emas, "has_ema_idx": has_ema_idx,
                    "Nf": Nf, "dev": dev_f, "chunked": chunked_flags,
                    "id_seg": id_seg, "par_ok": par_ok,
                }

            # Batched moment update for "simple" slow params: dense fp32
            # m/v with fp32 grads run one foreach pass instead of ~6 kernel
            # launches per param. Rare flags (unsanitized state, prune,
            # low-precision storage) keep their per-param ops so the math is
            # identical. Guard checks and apply stay in the loop below.
            _prebatched: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            if len(slow_params) > 1 and hasattr(torch, "_foreach_mul_"):
                _bp = []
                for p in slow_params:
                    _bs = self.state[p]
                    if ((factored and p.ndim >= 2) or "m" not in _bs
                            or "v" not in _bs
                            or _bs["m"].dtype != torch.float32
                            or _bs["v"].dtype != torch.float32
                            or p.grad is None
                            or p.grad.dtype != torch.float32):
                        continue
                    _bp.append((p, _bs["m"], _bs["v"], p.grad))
                if _bp:
                    _skip_nf = bool(self.defaults["skip_if_nonfinite"])
                    _after = not _skip_nf
                    _thr = (float(self.defaults["state_prune_threshold"])
                            if do_prune else 0.0)
                    mws = [m.clone() if _skip_nf else m for _, m, _, _ in _bp]
                    vws = [v.clone() if _skip_nf else v for _, _, v, _ in _bp]
                    gws = [g for _, _, _, g in _bp]
                    for (p, _, _, _), mw, vw in zip(_bp, mws, vws):
                        if not self.state[p].get("_state_sane"):
                            _nan_to_num_(mw, posinf=0.0, neginf=0.0)
                            _nan_to_num_(vw, posinf=0.0, neginf=0.0)
                            vw.clamp_(min=0.0)
                    torch._foreach_mul_(mws, b1)
                    torch._foreach_add_(mws, gws, alpha=1.0 - b1)
                    torch._foreach_mul_(vws, b2)
                    torch._foreach_addcmul_(vws, gws, gws, value=1.0 - b2)
                    for (p, _, _, _), mw, vw in zip(_bp, mws, vws):
                        if _after:
                            _nan_to_num_(mw)
                            _nan_to_num_(vw, neginf=0.0)
                            vw.clamp_(min=0.0)
                        if _thr > 0.0:
                            mw.masked_fill_(mw.abs() < _thr, 0.0)
                            vw.masked_fill_(vw.abs() < _thr, 0.0)
                    for (p, _, _, g), mw, vw in zip(_bp, mws, vws):
                        _prebatched[id(p)] = (mw, g, vw)

            for p in slow_params:
                _pb = _prebatched.get(id(p))
                pi = param_indices[id(p)]
                g = p.grad
                st = self.state[p]
                dt = self._mom_storage_dtype
                created_keys = []
                if "m" not in st:
                    st["m"] = torch.zeros_like(p, dtype=dt)
                    created_keys.append("m")
                m = st["m"]

                if factored and p.ndim>=2:
                    rows = p.shape[0]; cols = p.numel()//rows
                    if "vr" not in st:
                        st["vr"] = torch.zeros(rows, dtype=dt, device=p.device)
                        st["vc"] = torch.zeros(cols, dtype=dt, device=p.device)
                        created_keys.extend(("vr", "vc"))
                    vr = st["vr"]; vc = st["vc"]
                else:
                    if "v" not in st:
                        st["v"] = torch.zeros_like(p, dtype=dt)
                        created_keys.append("v")
                    v = st["v"]
                state_sane = bool(st.get("_state_sane", False))
                sanitize_loaded_state = not state_sane
                sanitize_after_update = not bool(self.defaults["skip_if_nonfinite"])

                # m update
                if _pb is not None:
                    m32, g32, v32 = _pb
                else:
                    m32 = _to_dtype(m, torch.float32); g32 = _to_dtype(g, torch.float32)
                    if self.defaults["skip_if_nonfinite"] and m32 is m:
                        m32 = m32.clone()
                    if sanitize_loaded_state:
                        _nan_to_num_(m32, posinf=0.0, neginf=0.0)
                    m32.mul_(b1).add_(g32, alpha=1.0-b1)
                    if sanitize_after_update:
                        _nan_to_num_(m32)
                    if do_prune and float(self.defaults["state_prune_threshold"]) > 0.0:
                        thr = float(self.defaults["state_prune_threshold"])
                        m32.masked_fill_(m32.abs() < thr, 0.0)

                # vhat
                if _pb is not None:
                    vhat = v32
                elif factored and p.ndim>=2:
                    g2 = g32.view(p.shape[0], -1).square()
                    vr32 = _to_dtype(vr, torch.float32); vc32 = _to_dtype(vc, torch.float32)
                    if self.defaults["skip_if_nonfinite"]:
                        if vr32 is vr: vr32 = vr32.clone()
                        if vc32 is vc: vc32 = vc32.clone()
                    if sanitize_loaded_state:
                        _nan_to_num_(vr32, posinf=0.0, neginf=0.0)
                        _nan_to_num_(vc32, posinf=0.0, neginf=0.0)
                        vr32.clamp_(min=0.0)
                        vc32.clamp_(min=0.0)
                    # Divide before reducing so a finite mean does not
                    # overflow in the intermediate sum.
                    vr32.mul_(b2).add_((g2 / float(cols)).sum(dim=1), alpha=1.0 - b2)
                    vc32.mul_(b2).add_((g2 / float(rows)).sum(dim=0), alpha=1.0 - b2)
                    if sanitize_after_update:
                        _nan_to_num_(vr32, neginf=0.0)
                        _nan_to_num_(vc32, neginf=0.0)
                        vr32.clamp_(min=0.0)
                        vc32.clamp_(min=0.0)
                    # Normalize first: the unnormalized outer product can
                    # overflow even when every factored moment is finite.
                    vr_mean = (vr32 / float(rows)).sum()
                    vhat = (vr32 / torch.maximum(vr_mean, scalar_vtiny)).unsqueeze(1) * vc32.unsqueeze(0)
                    vhat = vhat.view_as(p)
                else:
                    v = st["v"]
                    v32 = _to_dtype(v, torch.float32)
                    if self.defaults["skip_if_nonfinite"] and v32 is v:
                        v32 = v32.clone()
                    if sanitize_loaded_state:
                        _nan_to_num_(v32, posinf=0.0, neginf=0.0)
                        v32.clamp_(min=0.0)
                    v32.mul_(b2).addcmul_(g32, g32, value=1.0-b2)
                    if sanitize_after_update:
                        _nan_to_num_(v32, neginf=0.0)
                        v32.clamp_(min=0.0)
                    if do_prune and float(self.defaults["state_prune_threshold"]) > 0.0:
                        thr = float(self.defaults["state_prune_threshold"])
                        v32.masked_fill_(v32.abs() < thr, 0.0)
                    vhat = v32

                if self.defaults["skip_if_nonfinite"]:
                    stored_candidates = [m32, vr32, vc32] if factored and p.ndim >= 2 else [m32, v32]
                    # Reduce each nonempty tensor on device before testing finiteness.
                    # An empty tensor passed the old isfinite(...).all() check.
                    stored_magnitudes = [candidate.abs().amax() for candidate in stored_candidates if candidate.numel()]
                    finite_magnitudes = list(stored_magnitudes)
                    if factored and p.ndim >= 2 and vhat.numel():
                        finite_magnitudes.append(vhat.abs().amax())
                    checks = []
                    if dt != torch.float32:
                        storage_max = torch.finfo(dt).max
                        if stored_magnitudes:
                            checks.append(torch.stack(stored_magnitudes).amax() <= storage_max)

                if sanitize_after_update and isinstance(vhat, torch.Tensor):
                    vhat.clamp_(min=1e-30)
                P = (vhat.sqrt().add_(eps_t)).pow(-float(alpha)) if alpha>0 else 1.0
                if isinstance(P, torch.Tensor):
                    P.clamp_(min=1e-8, max=1e8)
                else:
                    P = _clamp_finite(P, lo=1e-8, hi=1e8)

                # Direction
                if group["sign_mode"] == "softsign":
                    d_sign = _softsign(m32, float(group["sign_tau"])) * P
                else:
                    d_sign = torch.sign(m32) * P
                d_mag = (m32 / (_rms(m32)+eps)) * P
                d = (1.0 - sblend) * d_sign + sblend * d_mag
                precond_preview = None
                if self.defaults["skip_if_nonfinite"]:
                    # Keep one host decision for candidate state and direction.
                    # No moment, parameter, or weight-decay update occurs before it.
                    direction_max = d.abs().amax() if d.numel() else scalar_zero
                    if d.numel():
                        finite_magnitudes.append(direction_max)
                    checks.append(
                        torch.isfinite(torch.stack(finite_magnitudes).amax())
                        if finite_magnitudes else torch.ones_like(scalar_zero, dtype=torch.bool)
                    )
                    # Cheap max-based bounds cover ordinary magnitudes. Only
                    # when a loose norm bound fails do we compute the exact
                    # stable norm for a sparse near-limit tensor.
                    norm_checks = []
                    norm_probes = []
                    if p.dtype in (torch.float16, torch.bfloat16):
                        # Preview the final parameter value before committing
                        # moments or weight decay. A finite FP32 direction can
                        # overflow when stored in a low-precision parameter.
                        storage_max = torch.finfo(p.dtype).max
                        preview_p = p.detach().clone()
                        preview_p.add_(preview_p, alpha=-(lr * lr_scale * float(group["weight_decay"])))
                        checks.append(torch.isfinite(preview_p).all())
                        preview_d = d
                        if agc_clip and agc_clip > 0.0:
                            preview_p_norm = _scalar_tensor(_norm(preview_p), preview_p)
                            preview_d_norm = _scalar_tensor(_norm(preview_d), preview_d)
                            agc_scale = torch.minimum(
                                (preview_p_norm + agc_eps_t) * agc_clip_t
                                / (preview_d_norm + scalar_tiny), scalar_one,
                            )
                            preview_d = preview_d * agc_scale.to(dtype=preview_d.dtype)
                        preview_clip = scalar_one
                        if rms_thr and rms_gran == "param":
                            preview_rms = preview_d.to(torch.float32).square().mean().sqrt()
                            preview_clip = torch.minimum(
                                rms_thr_t / (preview_rms + scalar_tiny), scalar_one,
                            )

                        def preview_trust() -> torch.Tensor:
                            if not use_trust:
                                return scalar_one
                            if trust_space == "update":
                                denominator = _scalar_tensor(_norm(preview_d), preview_d)
                            elif trust_space == "precond":
                                denominator = _scalar_tensor(_norm(m32 * P), preview_p)
                            else:
                                return scalar_one
                            numerator = _scalar_tensor(_norm(preview_p), preview_p)
                            valid = (numerator * denominator > scalar_zero).to(dtype=numerator.dtype)
                            raw = scalar_one + valid * (
                                numerator / torch.clamp(denominator, min=scalar_tiny) - scalar_one
                            )
                            if trust_beta > 0.0:
                                old = self._trust_ema.get((gi, pi))
                                if old is not None:
                                    raw = trust_beta * _scalar_tensor(old, group_ref) + (1.0 - trust_beta) * raw
                            return torch.minimum(trust_cap_t, raw)

                        preview_value = preview_p.to(torch.float32).clone()
                        preview_rule = qkv_rules.get(id(p))
                        preview_qkv_route = preview_rule is not None and (
                            qkv_lr is not None or (qkv_split and use_trust)
                        )
                        if preview_qkv_route:
                            dim, parts = preview_rule
                            if use_trust and qkv_split:
                                preview_trusts, _ = self._qkv_chunk_trust(
                                    gi, pi, preview_p, preview_d, dim=dim, parts=parts,
                                    trust_beta=trust_beta, trust_cap_t=trust_cap_t,
                                    scalar_one=scalar_one, scalar_zero=scalar_zero,
                                    scalar_tiny=scalar_tiny,
                                    trust_denominator=m32 * P if trust_space == "precond" else None,
                                    record_ema=False,
                                )
                            else:
                                preview_trusts = preview_trust().expand(parts)
                            for i, (candidate, direction) in enumerate(zip(
                                torch.chunk(preview_value, parts, dim=dim),
                                torch.chunk(preview_d, parts, dim=dim),
                            )):
                                effective_lr = lr * lr_scale
                                if qkv_lr is not None and i < 3:
                                    effective_lr *= float(qkv_lr.get(("q", "k", "v")[i], 1.0))
                                # add_'s FP32 alpha conversion can raise before
                                # the tensor finite checks see an oversized LR.
                                if not math.isfinite(effective_lr) or abs(effective_lr) > torch.finfo(torch.float32).max:
                                    checks.append(torch.zeros_like(scalar_zero, dtype=torch.bool))
                                    continue
                                candidate.add_(direction.to(torch.float32) * (preview_trusts[i] * preview_clip), alpha=-effective_lr)
                        else:
                            unstandardized = preview_d.to(torch.float32) * (preview_trust() * preview_clip) * (lr * lr_scale)
                            # Bucket standardization can replace preview_value
                            # with a finite bound, so validate its input too.
                            checks.append(torch.isfinite(unstandardized).all())
                            if (use_foreach_upd and bucket_std and hasattr(torch, "_foreach_add_")
                                    and len(active_params) >= foreach_min_bucket
                                    and not (blk_chunks and p.ndim == 2 and p.shape[0] % blk_chunks == 0)):
                                # The bucket's other updates are not available
                                # yet. These are sufficient bounds for its
                                # eventual RMS standardization.
                                if use_triton_stats or bucket_src == "global":
                                    bound = math.sqrt(sum(item.numel() for item in active_params))
                                else:
                                    bound = len(active_params) * math.sqrt(p.numel())
                                preview_value = preview_value.abs() + bound
                            else:
                                preview_value.sub_(unstandardized)
                        checks.extend((
                            torch.isfinite(preview_value).all(),
                            (preview_value.abs() <= storage_max).all(),
                        ))
                    elif (p.dtype == torch.float32
                          and rms_thr >= 0.0
                          and (not agc_clip or agc_eps >= 0.0)
                          and (not use_trust or (trust_clip >= 0.0 and 0.0 <= trust_beta <= 1.0))):
                        # Bound all FP32 update routes before committing state.
                        # AGC and RMS clipping only shrink the direction; trust
                        # is capped. Include the unstandardized foreach input,
                        # which must be finite even if bucket RMS later shrinks it.
                        param_max = p.detach().abs().amax() if p.numel() else scalar_zero
                        decay_rate = lr * lr_scale * float(group["weight_decay"])
                        # Decoupled decay is p <- p * (1 - rate), so positive
                        # decay may make a large but finite parameter safer.
                        # Allow for FP32 multiply/add rounding in the bound.
                        decay_growth = abs(1.0 - decay_rate) + (
                            8.0 * torch.finfo(torch.float32).eps * abs(decay_rate)
                        )
                        trust_growth = max(1.0, trust_clip) if use_trust else 1.0
                        route_lrs = [abs(lr * lr_scale)]
                        qkv_route = qkv_rules.get(id(p)) is not None and (qkv_lr is not None or (qkv_split and use_trust))
                        if qkv_route and qkv_lr is not None:
                            route_lrs.extend(abs(lr * lr_scale * float(qkv_lr.get(key, 1.0))) for key in ("q", "k", "v"))
                        route_lr = max(route_lrs) if all(math.isfinite(value) for value in route_lrs) else math.inf
                        scaled_direction_bound = direction_max * trust_growth
                        raw_update_bound = scaled_direction_bound * route_lr
                        bucket_route = (
                            not qkv_route
                            and not (blk_chunks and p.ndim == 2 and p.shape[0] % blk_chunks == 0)
                            and use_foreach_upd and bucket_std and hasattr(torch, "_foreach_add_")
                            and len(active_params) >= foreach_min_bucket
                        )
                        if bucket_route:
                            if bucket_src == "median" and not use_triton_stats:
                                # The median may be at the 1e-12 floor even
                                # when this update is much larger than its peers.
                                update_bound = raw_update_bound * 1e12
                            elif use_triton_stats or bucket_src == "global":
                                update_bound = math.sqrt(sum(item.numel() for item in active_params))
                            else:
                                update_bound = len(active_params) * math.sqrt(p.numel())
                        else:
                            update_bound = raw_update_bound
                        output_bound = param_max * decay_growth + update_bound
                        fp32_limit = torch.finfo(torch.float32).max * (1.0 - 8.0 * torch.finfo(torch.float32).eps)
                        # Compare vector-shaped bounds on MPS without reading
                        # each scalar comparison back to the host.
                        update_bounds = torch.stack((scaled_direction_bound, raw_update_bound, output_bound))
                        checks.append((update_bounds <= torch.full_like(update_bounds, fp32_limit)).all())
                        norm_bounds = []
                        if use_trust or (agc_clip and agc_clip > 0.0):
                            norm_bounds.append(param_max * (decay_growth * math.sqrt(p.numel())))
                            norm_bounds.append(direction_max * math.sqrt(d.numel()))
                            norm_probes.extend(((p.detach(), decay_growth), (d, 1.0)))
                        if use_trust and trust_space == "precond":
                            precond_preview = m32 * P
                            precond_max = precond_preview.abs().amax() if precond_preview.numel() else scalar_zero
                            norm_bounds.append(precond_max * math.sqrt(precond_preview.numel()))
                            norm_probes.append((precond_preview, 1.0))
                        if norm_bounds:
                            norm_values = torch.stack(norm_bounds)
                            norm_checks.append((norm_values <= torch.full_like(norm_values, fp32_limit)).all())
                        if use_trust and trust_beta > 0.0:
                            trust_key = (gi, pi)
                            old_trust = (
                                self._qkv_trust_ema.get(trust_key)
                                if qkv_route and qkv_split else self._trust_ema.get(trust_key)
                            )
                            if old_trust is not None:
                                if isinstance(old_trust, torch.Tensor):
                                    # MPS can read 0-D comparison results back to the host.
                                    old_flat = old_trust.reshape(-1)
                                    checks.append((torch.isfinite(old_flat) &
                                                   (old_flat >= torch.zeros_like(old_flat))).all())
                                else:
                                    checks.append(_scalar_like(p, math.isfinite(float(old_trust)) and float(old_trust) >= 0,
                                                               dtype=torch.bool))
                    guard_ok = bool(torch.stack(checks + norm_checks).all().item())
                    if not guard_ok and norm_probes and bool(torch.stack(checks).all().item()):
                        guard_ok = bool(torch.stack([
                            _scalar_tensor(_norm(tensor), p) * factor <= fp32_limit
                            for tensor, factor in norm_probes
                        ]).all().item())
                    if not guard_ok:
                        prof_c["nonfinite_skips"] += 1
                        for key in created_keys:
                            st.pop(key, None)
                        if not st:
                            self.state.pop(p, None)
                        continue
                if m32 is not m: m.copy_(m32.to(m.dtype))
                if factored and p.ndim >= 2:
                    if vr32 is not vr: vr.copy_(vr32.to(vr.dtype))
                    if vc32 is not vc: vc.copy_(vc32.to(vc.dtype))
                elif v32 is not v:
                    v.copy_(v32.to(v.dtype))
                st["_state_sane"] = True
                if self.defaults["skip_if_nonfinite"] and group["weight_decay"]:
                    p.add_(p, alpha=-(lr * lr_scale * float(group["weight_decay"])))
                    prof_c["scalar_wd"] += 1
                clip_scale_param = scalar_one
                p_norm_t: Optional[torch.Tensor] = None
                d_norm_t: Optional[torch.Tensor] = None

                # AGC
                if agc_clip and agc_clip > 0.0:
                    p_norm_t = _scalar_tensor(_norm(p), p)
                    d_norm_t = _scalar_tensor(_norm(d), d)
                    scale = torch.minimum((p_norm_t + agc_eps_t) * agc_clip_t / (d_norm_t + scalar_tiny), scalar_one)
                    d = d * scale.to(dtype=d.dtype)
                    d_norm_t = d_norm_t * scale
                    if agc_clips_t is not None:
                        agc_clips_t.add_((scale < scalar_one).to(dtype=agc_clips_t.dtype))

                need_d_rms = bool(lora_dense_sum_t is not None or cross_density_sum_t is not None or (rms_thr and rms_gran == "param"))
                need_d_sq = bool(need_d_rms or up_s2_t is not None or dn_s2_t is not None)
                d_sq_sum_t: Optional[torch.Tensor] = None
                d_rms_t: Optional[torch.Tensor] = None
                if need_d_sq:
                    d32 = d if d.dtype == torch.float32 else d.to(torch.float32)
                    d_sq_sum_t = d32.square().sum()
                    if up_s2_t is not None:
                        up_s2_t.add_(d_sq_sum_t)
                        up_n += d.numel()
                    elif dn_s2_t is not None:
                        dn_s2_t.add_(d_sq_sum_t)
                        dn_n += d.numel()
                    if need_d_rms:
                        d_numel_t = _scalar_like(d_sq_sum_t, float(d.numel()), dtype=torch.float32, device=d_sq_sum_t.device)
                        d_rms_t = torch.sqrt(d_sq_sum_t / d_numel_t)

                if lora_dense_sum_t is not None and d_rms_t is not None:
                    thr = d_rms_t * float(self.defaults["lora_density_k"])
                    lora_dense_sum_t.add_((d.abs() > thr.to(dtype=d.dtype)).to(dtype=lora_dense_sum_t.dtype).mean())
                    lora_count += 1

                if cross_density_sum_t is not None and d_rms_t is not None:
                    thr_cross = d_rms_t * float(self.defaults["lora_density_k"])
                    cross_density_sum_t.add_((d.abs() > thr_cross.to(dtype=d.dtype)).to(dtype=cross_density_sum_t.dtype).mean())
                    cross_density_count += 1

                base_lr = lr * lr_scale

                # RMS clip (param)
                if rms_thr and rms_gran == "param":
                    if d_rms_t is None:
                        d_rms_t = _scalar_tensor(_rms(d), d)
                    clip_scale_param = torch.minimum(rms_thr_t / (d_rms_t + scalar_tiny), scalar_one)

                precond_norm_t: Optional[torch.Tensor] = None
                if use_trust and trust_space == "precond":
                    precond_norm_t = _scalar_tensor(_norm(precond_preview if precond_preview is not None else m32 * P), p)

                # trust
                def eff_trust_for(_p, _d):
                    nonlocal p_norm_t, d_norm_t
                    if not use_trust:
                        return scalar_one
                    if trust_space == "update":
                        if _p is p:
                            if p_norm_t is None:
                                p_norm_t = _scalar_tensor(_norm(_p), _p)
                            pn = p_norm_t
                        else:
                            pn = _scalar_tensor(_norm(_p), _p)
                        if _d is d:
                            if d_norm_t is None:
                                d_norm_t = _scalar_tensor(_norm(_d), _d)
                            dn = d_norm_t
                        else:
                            dn = _scalar_tensor(_norm(_d), _d)
                        valid = (pn * dn > scalar_zero).to(dtype=pn.dtype)
                        raw = scalar_one + valid * (pn / torch.clamp(dn, min=scalar_tiny) - scalar_one)
                    elif trust_space == "precond":
                        if _p is p:
                            if p_norm_t is None:
                                p_norm_t = _scalar_tensor(_norm(_p), _p)
                            pn = p_norm_t
                        else:
                            pn = _scalar_tensor(_norm(_p), _p)
                        denom_n = precond_norm_t if precond_norm_t is not None else scalar_one
                        valid = (pn * denom_n > scalar_zero).to(dtype=pn.dtype)
                        raw = scalar_one + valid * (pn / torch.clamp(denom_n, min=scalar_tiny) - scalar_one)
                    else:
                        raw = scalar_one
                    if trust_beta > 0.0:
                        trust_key = (gi, pi)
                        old = self._trust_ema.get(trust_key)
                        if old is None:
                            sm = raw
                        else:
                            old_t = _scalar_tensor(old, group_ref)
                            sm = trust_beta * old_t + (1.0-trust_beta) * raw
                        self._trust_ema[trust_key] = sm.detach()
                        return torch.minimum(trust_cap_t, sm)
                    return torch.minimum(trust_cap_t, raw)

                # chunked paths
                chunked = False
                qkv_rule = qkv_rules.get(id(p))
                qkv_route = qkv_rule is not None and (qkv_lr is not None or (qkv_split and use_trust))
                if (blk_chunks and p.ndim==2 and p.shape[0] % blk_chunks == 0) or qkv_route:
                    chunked = True
                    if qkv_route:
                        dim, parts = qkv_rule
                        p_chunks = torch.chunk(p, parts, dim=dim)
                        d_chunks = torch.chunk(d, parts, dim=dim)
                        keys = ("q","k","v")
                        if use_trust and qkv_split:
                            chunk_trust_t, d_chunk_norms_t = self._qkv_chunk_trust(
                                gi,
                                pi,
                                p,
                                d,
                                dim=dim,
                                parts=parts,
                                trust_beta=trust_beta,
                                trust_cap_t=trust_cap_t,
                                scalar_one=scalar_one,
                                scalar_zero=scalar_zero,
                                scalar_tiny=scalar_tiny,
                                trust_denominator=m32 * P if trust_space == "precond" else None,
                            )
                        else:
                            d_chunk_norms_t = _chunk_scalar_norms(d, dim, parts)
                            trust = eff_trust_for(p, d) if use_trust else scalar_one
                            chunk_trust_t = trust.expand(parts)
                        chunk_rms_t = d_chunk_norms_t / math.sqrt(max(1, d.numel() // parts))
                        spectral_fused = (
                            d if collect_qkv_spectral and dim == 0 and parts == 3 and d.is_contiguous()
                            and d.shape[0] % 3 == 0 else None
                        )
                        spectral_by_chunk = (
                            _spectral_dispersion_chunks(
                                d_chunks[:len(keys)], spec_low_band, spec_high_band,
                                fused=spectral_fused,
                            )
                            if collect_qkv_spectral else None
                        )
                        for i, (pc, dc) in enumerate(zip(p_chunks, d_chunks)):
                            tr = chunk_trust_t[i]
                            eff_lr = base_lr
                            if qkv_lr is not None and i < len(keys):
                                # Group controls and adaptive slice controls compose.
                                eff_lr = base_lr * float(qkv_lr.get(keys[i], 1.0))
                            low_raw_t = high_raw_t = phase_raw_t = None
                            if collect_qkv_spectral and i < len(keys):
                                low_raw_t, high_raw_t, phase_raw_t = spectral_by_chunk[i]
                            if qkv_observations is not None and i < len(keys):
                                qkv_observations[keys[i]].append((
                                    tr.detach(), chunk_rms_t[i].detach(),
                                    low_raw_t, high_raw_t, phase_raw_t,
                                ))
                            scale = (tr * clip_scale_param).to(device=pc.device, dtype=torch.float32)
                            if pc.dtype in (torch.float16, torch.bfloat16):
                                pc.copy_((pc.to(torch.float32) - eff_lr * dc.to(torch.float32) * scale).to(pc.dtype))
                            else:
                                pc.add_(dc.to(torch.float32) * scale, alpha=-eff_lr)
                    else:
                        parts = blk_chunks
                        p_chunks = torch.chunk(p, parts, dim=0)
                        d_chunks = torch.chunk(d, parts, dim=0)
                        block_scale = (eff_trust_for(p, d) * clip_scale_param).to(device=p.device, dtype=torch.float32)
                        for pc, dc in zip(p_chunks, d_chunks):
                            if pc.dtype in (torch.float16, torch.bfloat16):
                                pc.copy_((pc.to(torch.float32) - base_lr * dc.to(torch.float32) * block_scale).to(pc.dtype))
                            else:
                                pc.add_(dc.to(torch.float32) * block_scale, alpha=-base_lr)

                # foreach bucket
                if not chunked:
                    eff = (eff_trust_for(p, d) * clip_scale_param).to(device=d.device, dtype=torch.float32)
                    if use_foreach_upd and hasattr(torch, "_foreach_add_"):
                        requested_dtype = self._dtype_for_update(p)
                        # Apply the scalar scale in FP32 before any optional
                        # low-precision buffer cast. Under the skip policy,
                        # retain FP32 so an otherwise finite step cannot be
                        # spoiled by a narrower scratch buffer.
                        update_dtype = (
                            torch.float32
                            if p.dtype in (torch.float16, torch.bfloat16)
                            or (self.defaults["skip_if_nonfinite"] and requested_dtype in (torch.float16, torch.bfloat16))
                            else requested_dtype
                        )
                        upd = d.to(torch.float32) * eff
                        upd.mul_(-base_lr)
                        if update_dtype != torch.float32:
                            upd = upd.to(update_dtype)
                        b_params.append(p); b_updates.append(upd); 
                        if bucket_std:
                            b_updates_f32.append(upd.to(torch.float32))
                    else:
                        if p.dtype in (torch.float16, torch.bfloat16):
                            p.copy_((p.to(torch.float32) - base_lr * d.to(torch.float32) * eff).to(p.dtype))
                        else:
                            p.add_(d.to(torch.float32) * eff, alpha=-base_lr)

            # flush foreach bucket (combined stats & scale)
            # ---- Fused flat apply: eff, launch K5, enqueue update views ----
            if fast_ctx is not None:
                F = fast_ctx["F"]; d_f = fast_ctx["d"]
                decay_rate = fast_ctx["decay_rate"]
                Nf = fast_ctx["Nf"]; dev_f = fast_ctx["dev"]
                Nseg = F["lens_t"].numel()
                seg_par = F["seg_param_t"]
                param_nseg = F["param_nseg"]
                chunked_fl = fast_ctx["chunked"]
                id_seg = fast_ctx["id_seg"]
                lens = F["lens_t"]; numel_f = F["numel_f"]
                pn_par = fast_ctx["pn_par"]; dn_par = fast_ctx["dn_par"]
                valid_idx = [i for i, b in enumerate(fast_ok_list) if b]
                bad_idx = [i for i, b in enumerate(fast_ok_list) if not b]
                if bad_idx:
                    prof_c["nonfinite_skips"] += len(bad_idx)
                if valid_idx:
                    B2 = F["B"]
                    _agc_on = bool(agc_clip and agc_clip > 0.0)
                    _den_on = (lora_dense_sum_t is not None
                               or cross_density_sum_t is not None
                               or up_s2_t is not None or dn_s2_t is not None)
                    _any_split = any(split_list)
                    seg_g = F["seg_idx"]

                    def _applyg():
                        sc = B2["scalars"]
                        ok_par = B2["par_ok"].to(torch.float32)
                        ok_seg = ok_par if id_seg else ok_par[seg_par]
                        eff = B2["eff"].fill_(1.0)
                        pn_use = dn_use = None
                        if pn_par is not None:
                            pn_use = pn_par * sc[12]
                            dn_use = dn_par
                        agc_scale = None
                        if _agc_on:
                            agc_scale = torch.minimum(
                                (pn_use + sc[5]) * sc[4] / dn_use.clamp(min=1e-12),
                                torch.ones_like(pn_use))
                            eff *= agc_scale if id_seg else agc_scale[seg_par]
                            dn_use = dn_use * agc_scale
                            if agc_clips_t is not None:
                                B2["acc"][7].copy_((agc_scale < 1.0).sum())
                        if _den_on:
                            if agc_scale is None:
                                d_den = d_f
                            else:
                                d_den = d_f * (
                                    agc_scale if id_seg else agc_scale[seg_par])[seg_g]
                            flat_seg_ssq(d_den, F["starts"], F["ends"], B2["ssq"])
                            if id_seg:
                                ssq_par = B2["ssq"]; n_par = numel_f
                            else:
                                ssq_par = B2["ssq2"].zero_().index_add_(
                                    0, seg_par, B2["ssq"])
                                n_par = B2["n2"].zero_().index_add_(
                                    0, seg_par, numel_f)
                            if up_s2_t is not None:
                                B2["acc"][0].copy_(ssq_par.sum())
                            elif dn_s2_t is not None:
                                B2["acc"][1].copy_(ssq_par.sum())
                            if (lora_dense_sum_t is not None
                                    or cross_density_sum_t is not None):
                                drms = (ssq_par / n_par).sqrt()
                                thr = (drms * sc[8] if id_seg
                                       else (drms * sc[8])[seg_par])[seg_g]
                                flat_seg_ssq((d_den.abs() > thr).to(torch.float32),
                                             F["starts"], F["ends"], B2["cnt"])
                                if id_seg:
                                    dens = B2["cnt"] / n_par
                                else:
                                    dens = (B2["ssq2"].zero_().index_add_(
                                        0, seg_par, B2["cnt"]) / n_par)
                                if lora_dense_sum_t is not None:
                                    B2["acc"][2].copy_(dens.sum())
                                if cross_density_sum_t is not None:
                                    B2["acc"][3].copy_(dens.sum())
                        if use_trust:
                            # Norms are post-decay (p) and post-AGC (d), as in
                            # the slow path.
                            pn_g_use = fast_ctx["pn_g"]
                            if pn_g_use is not None:
                                pn_g_use = pn_g_use * sc[12]
                            if trust_space == "update":
                                den_par = dn_use
                                den_seg = fast_ctx["dn_g"]
                                if den_seg is not None and agc_scale is not None:
                                    den_seg = den_seg * (
                                        agc_scale if id_seg else agc_scale[seg_par])
                            elif trust_space == "precond":
                                den_par = fast_ctx["pre_n_par"]
                                den_seg = fast_ctx["pre_n"]
                            else:
                                den_par = None; den_seg = fast_ctx["dn_g"]
                            raw_par = torch.ones(Nf, device=dev_f)
                            if den_par is not None:
                                valid_v = (pn_use * den_par > 0).to(torch.float32)
                                raw_par = 1.0 + valid_v * (
                                    pn_use / den_par.clamp(min=1e-12) - 1.0)
                            raw_seg = None
                            if _any_split and den_seg is not None:
                                valid_s = (pn_g_use * den_seg > 0).to(torch.float32)
                                raw_seg = 1.0 + valid_s * (
                                    pn_g_use / den_seg.clamp(min=1e-12) - 1.0)
                            if trust_beta > 0.0:
                                sm = torch.where(
                                    B2["hasem"] > 0,
                                    sc[6] * B2["emas_t"] + sc[10] * raw_par,
                                    raw_par)
                                B2["sm"].copy_(sm)
                                base = sm[seg_par]
                                if _any_split and raw_seg is not None:
                                    qsm = torch.where(
                                        B2["hasemq"] > 0,
                                        sc[6] * B2["qemas"] + sc[10] * raw_seg,
                                        raw_seg)
                                    B2["qsm"].copy_(qsm)
                                    base = torch.where(B2["split_t"], qsm, base)
                            else:
                                base = raw_par[seg_par]
                                if _any_split and raw_seg is not None:
                                    base = torch.where(B2["split_t"], raw_seg, base)
                            tval = torch.minimum(base, sc[7])
                            eff *= tval
                        eff *= B2["qscale"]
                        eff *= ok_seg
                        # Rejected params keep factor 1.0 — no decay, no update.
                        B2["fac"].copy_(sc[11] * ok_par + (1.0 - ok_par))
                        torch._foreach_mul_(
                            fast_params, [B2["fac"][i] for i in range(Nf)])
                        flat_apply(fast_ctx["flat_p"], F["flat_m"], F["flat_v"],
                                   fast_ctx["cand_m"], fast_ctx["cand_v"], d_f,
                                   eff, seg_g, ok_seg, B2["upd"],
                                   decay=B2["kzero"], lr=sc[9])
                        if F.get("fact_meta") is not None:
                            from .triton_kernels import flat_fact_commit
                            flat_fact_commit(F["flat_vr"], F["flat_vc"],
                                             F["cand_vr"], F["cand_vc"],
                                             F["fact_meta"]["rok"], ok_seg)
                        _views = [B2["upd"][o:o + n].view_as(p)
                                  for p, o, n in zip(fast_params, F["offsets"],
                                                     F["lengths"])]
                        if use_foreach_upd:
                            _dir_p = [p for p, c in zip(fast_params, chunked_fl) if c]
                            _dir_u = [u for u, c in zip(_views, chunked_fl) if c]
                            if _dir_p:
                                torch._foreach_add_(_dir_p, _dir_u)
                        else:
                            torch._foreach_add_(fast_params, _views)

                    # Graph B: post-par_ok apply tail. Rejected params fall
                    # back to the eager tail so bucket membership and EMA
                    # semantics stay identical.
                    gs2 = F["graph2"]
                    _gkey2 = (_gkey, use_foreach_upd,
                              agc_clips_t is not None,
                              lora_dense_sum_t is not None,
                              cross_density_sum_t is not None,
                              up_s2_t is not None, dn_s2_t is not None,
                              _any_split)
                    if gs2["g"] is not None and gs2["key"] != _gkey2:
                        gs2["g"] = None; gs2["warm"] = 0
                    _g2_done = False
                    if not bad_idx and not gs2["bad"]:
                        # Give the graph the host-masked par_ok.
                        if fast_ctx["par_ok"] is not B2["par_ok"]:
                            B2["par_ok"].copy_(fast_ctx["par_ok"])
                        if gs2["g"] is not None:
                            gs2["g"].replay()
                            _g2_done = True
                        elif gs2["warm"] >= 2:
                            try:
                                pool = self._graph_pools.get(group_ref.device)
                                if pool is None:
                                    pool = torch.cuda.graph_pool_handle()
                                    self._graph_pools[group_ref.device] = pool
                                gcap2 = torch.cuda.CUDAGraph()
                                with torch.cuda.device(group_ref.device):
                                    with torch.cuda.graph(gcap2, pool=pool):
                                        _applyg()
                                gs2["g"] = gcap2; gs2["key"] = _gkey2
                                gs2["g"].replay()
                                _g2_done = True
                            except Exception as _ge2:
                                gs2["bad"] = True; gs2["g"] = None
                                gs2["err"] = repr(_ge2)
                        else:
                            gs2["warm"] += 1
                    if _g2_done:
                        # Host bookkeeping the graph cannot do: EMA dicts,
                        # scalar accumulators, and bucket membership.
                        if up_s2_t is not None:
                            up_s2_t.add_(B2["acc"][0])
                            up_n += sum(F["lengths"])
                        elif dn_s2_t is not None:
                            dn_s2_t.add_(B2["acc"][1])
                            dn_n += sum(F["lengths"])
                        if lora_dense_sum_t is not None:
                            lora_dense_sum_t.add_(B2["acc"][2])
                            lora_count += Nf
                        if cross_density_sum_t is not None:
                            cross_density_sum_t.add_(B2["acc"][3])
                            cross_density_count += Nf
                        if agc_clips_t is not None:
                            agc_clips_t.add_(B2["acc"][7])
                        if group["weight_decay"]:
                            prof_c["scalar_wd"] += Nf
                        if use_trust and trust_beta > 0.0:
                            _s0 = 0
                            for i, p in enumerate(fast_params):
                                a = _s0; _s0 += param_nseg[i]
                                tkey = (gi, param_indices[id(p)])
                                if split_list[i]:
                                    self._qkv_trust_ema[tkey] = (
                                        B2["qsm"][a:a + param_nseg[i]].detach())
                                else:
                                    self._trust_ema[tkey] = B2["sm"][i].detach()
                        if use_foreach_upd:
                            for i, p in enumerate(fast_params):
                                if chunked_fl[i]:
                                    continue
                                o = F["offsets"][i]; n = F["lengths"][i]
                                u = B2["upd"][o:o + n].view_as(p)
                                b_params.append(p); b_updates.append(u)
                                if bucket_std:
                                    b_updates_f32.append(u)
                if valid_idx and not _g2_done:
                    eff = torch.ones(Nseg, device=dev_f)
                    agc_scale = None
                    if pn_par is not None and decay_rate != 0.0:
                        pn_par = pn_par * abs(1.0 - decay_rate)
                    if agc_clip and agc_clip > 0.0:
                        agc_scale = torch.minimum(
                            (pn_par + float(agc_eps)) * float(agc_clip) / dn_par.clamp(min=1e-12),
                            torch.ones_like(pn_par))
                        eff = eff * (agc_scale if id_seg else agc_scale[seg_par])
                        dn_par = dn_par * agc_scale  # post-AGC norm, matches slow path
                        if agc_clips_t is not None:
                            _c = (agc_scale < 1.0).to(agc_clips_t.dtype).sum()
                            if bad_idx:
                                _c = _c - (agc_scale[bad_idx] < 1.0).to(agc_clips_t.dtype).sum()
                            agc_clips_t.add_(_c)
                    # Density / FFN accumulators mirror the slow path (post-AGC d).
                    if (lora_dense_sum_t is not None or cross_density_sum_t is not None
                            or up_s2_t is not None or dn_s2_t is not None):
                        d_den = d_f if agc_scale is None else (
                            d_f * torch.repeat_interleave(
                                agc_scale if id_seg else agc_scale[seg_par], lens))
                        ssq_par = torch.empty(Nseg, device=dev_f)
                        flat_seg_ssq(d_den, F["starts"], F["ends"], ssq_par)
                        n_par = numel_f
                        if not id_seg:
                            n_par = torch.zeros(Nf, device=dev_f).index_add_(0, seg_par, numel_f)
                            ssq_par = torch.zeros(Nf, device=dev_f).index_add_(0, seg_par, ssq_par)
                        if bad_idx:
                            ssq_par[bad_idx] = 0.0
                        if up_s2_t is not None:
                            up_s2_t.add_(ssq_par.sum())
                            up_n += sum(F["lengths"][i] for i in valid_idx)
                        elif dn_s2_t is not None:
                            dn_s2_t.add_(ssq_par.sum())
                            dn_n += sum(F["lengths"][i] for i in valid_idx)
                        if lora_dense_sum_t is not None or cross_density_sum_t is not None:
                            k = float(self.defaults["lora_density_k"])
                            drms = (ssq_par / n_par).sqrt()
                            thr = torch.repeat_interleave(
                                drms * k if id_seg else (drms * k)[seg_par], lens)
                            cnt = torch.empty(Nseg, device=dev_f)
                            flat_seg_ssq((d_den.abs() > thr).to(torch.float32),
                                         F["starts"], F["ends"], cnt)
                            dens = cnt / n_par if id_seg else (
                                torch.zeros(Nf, device=dev_f).index_add_(0, seg_par, cnt) / n_par)
                            if bad_idx:
                                dens[bad_idx] = 0.0
                            if lora_dense_sum_t is not None:
                                lora_dense_sum_t.add_(dens.sum())
                                lora_count += len(valid_idx)
                            if cross_density_sum_t is not None:
                                cross_density_sum_t.add_(dens.sum())
                                cross_density_count += len(valid_idx)
                    if use_trust:
                        # Split-trust params use per-chunk norms and a vector
                        # EMA; all others use param-level scalar trust. Norms
                        # are post-decay (p) and post-AGC (d), as in the slow
                        # path.
                        split = [chunked_fl[i] and qkv_split
                                 and qkv_rules.get(id(fast_params[i])) is not None
                                 for i in range(Nf)]
                        pn_g_use = fast_ctx["pn_g"]
                        if pn_g_use is not None and decay_rate != 0.0:
                            pn_g_use = pn_g_use * abs(1.0 - decay_rate)
                        if trust_space == "update":
                            den_par = dn_par
                            den_seg = fast_ctx["dn_g"]
                            if den_seg is not None and agc_scale is not None:
                                den_seg = den_seg * (
                                    agc_scale if id_seg else agc_scale[seg_par])
                        elif trust_space == "precond":
                            den_par = fast_ctx["pre_n_par"]; den_seg = fast_ctx["pre_n"]
                        else:
                            # Unknown space: scalar trust is 1 (eager), but
                            # _qkv_chunk_trust still denominates by d-norms.
                            den_par = None; den_seg = fast_ctx["dn_g"]
                        raw_par = torch.ones(Nf, device=dev_f)
                        if den_par is not None:
                            valid_v = (pn_par * den_par > 0).to(torch.float32)
                            raw_par = 1.0 + valid_v * (pn_par / den_par.clamp(min=1e-12) - 1.0)
                        raw_seg = None
                        if any(split) and den_seg is not None:
                            valid_s = (pn_g_use * den_seg > 0).to(torch.float32)
                            raw_seg = 1.0 + valid_s * (pn_g_use / den_seg.clamp(min=1e-12) - 1.0)
                        tval = torch.ones(Nseg, device=dev_f)
                        s0 = 0
                        seg_of_param: List[int] = []
                        for k in param_nseg:
                            seg_of_param.append(s0)
                            s0 += k
                        if trust_beta > 0.0:
                            sm = trust_beta * fast_ctx["emas"] + (1.0 - trust_beta) * raw_par
                            miss = [i for i in range(Nf) if i not in set(fast_ctx["has_ema_idx"])]
                            if miss:
                                sm[miss] = raw_par[miss]
                            for i in valid_idx:
                                a = seg_of_param[i]; b = a + param_nseg[i]
                                tkey = (gi, param_indices[id(fast_params[i])])
                                if split[i] and raw_seg is not None:
                                    prev = self._qkv_trust_ema.get(tkey)
                                    r = raw_seg[a:b]
                                    sm_i = (trust_beta * prev.to(r.dtype) + (1.0 - trust_beta) * r
                                            if prev is not None and prev.numel() == r.numel() else r)
                                    self._qkv_trust_ema[tkey] = sm_i.detach()
                                    tval[a:b] = torch.minimum(sm_i, trust_cap_t)
                                else:
                                    self._trust_ema[tkey] = sm[i].detach()
                                    tval[a:b] = torch.minimum(sm[i].expand(b - a), trust_cap_t)
                        else:
                            tval = torch.minimum(
                                raw_par if id_seg else raw_par[seg_par], trust_cap_t)
                            for i in valid_idx:
                                if split[i] and raw_seg is not None:
                                    a = seg_of_param[i]; b = a + param_nseg[i]
                                    tval[a:b] = torch.minimum(raw_seg[a:b], trust_cap_t)
                        eff = eff * tval
                    # Per-chunk lr scales ride on the per-seg eff.
                    if qkv_lr is not None and any(
                            qkv_rules.get(id(p)) is not None for p in fast_params):
                        _keys = ("q", "k", "v")
                        qscale = []
                        for i, p in enumerate(fast_params):
                            k = param_nseg[i]
                            if qkv_rules.get(id(p)) is not None:
                                qscale.extend(float(qkv_lr.get(_keys[j], 1.0))
                                              if j < 3 else 1.0 for j in range(k))
                            else:
                                qscale.extend([1.0] * k)
                        if any(s != 1.0 for s in qscale):
                            eff = eff * torch.tensor(qscale, device=dev_f)
                    if bad_idx:
                        ok_par_t = fast_ctx["par_ok"].to(torch.float32)
                        eff = eff * (ok_par_t if id_seg else ok_par_t[seg_par])
                    if group["weight_decay"]:
                        torch._foreach_mul_([fast_params[i] for i in valid_idx], 1.0 - decay_rate)
                        prof_c["scalar_wd"] += len(valid_idx)
                    upd = torch.empty(F["total"], device=dev_f)
                    ok_t = fast_ctx["par_ok"].to(torch.float32)
                    if not id_seg:
                        ok_t = ok_t[seg_par]
                    flat_apply(fast_ctx["flat_p"], F["flat_m"], F["flat_v"],
                               fast_ctx["cand_m"], fast_ctx["cand_v"], d_f, eff,
                               F["seg_idx"], ok_t, upd,
                               decay=B2["kzero"], lr=lr * lr_scale)
                    if F.get("fact_meta") is not None:
                        from .triton_kernels import flat_fact_commit
                        flat_fact_commit(F["flat_vr"], F["flat_vc"],
                                         F["cand_vr"], F["cand_vc"],
                                         F["fact_meta"]["rok"], ok_t)
                    if use_foreach_upd:
                        direct = []
                        direct_upd = []
                        for i in valid_idx:
                            p = fast_params[i]
                            o = F["offsets"][i]; n = p.numel()
                            u = upd[o:o + n].view_as(p)
                            if chunked_fl[i]:
                                direct.append(p); direct_upd.append(u)
                            else:
                                b_params.append(p); b_updates.append(u)
                                if bucket_std:
                                    b_updates_f32.append(u)
                        if direct:
                            torch._foreach_add_(direct, direct_upd)
                    else:
                        torch._foreach_add_(
                            [fast_params[i] for i in valid_idx],
                            [upd[F["offsets"][i]:F["offsets"][i] + fast_params[i].numel()].view_as(fast_params[i])
                             for i in valid_idx])

            if b_params and b_updates and hasattr(torch, "_foreach_add_"):
                prof_c["foreach_bucket_size"] = len(b_params)
                if len(b_params) >= foreach_min_bucket and bucket_std:
                    if use_triton_stats:
                        ussq, pssq, n = self._bucket_stats_triton(b_updates_f32, [p.detach() for p in b_params])
                        grms = torch.sqrt(ussq / torch.clamp(n, min=_scalar_like(n, 1.0)))
                        trust_est = torch.sqrt(pssq) / torch.clamp(torch.sqrt(ussq), min=_scalar_like(ussq, 1e-12))
                    else:
                        grms = self._bucket_global_rms(b_updates_f32, source=bucket_src, reference=b_updates_f32[0])
                        pssq = torch.stack([torch.sum(p.detach().to(torch.float32)**2) for p in b_params]).sum()
                        ussq = torch.stack([torch.sum(u.to(torch.float32)**2) for u in b_updates_f32]).sum()
                        trust_est = torch.sqrt(pssq) / torch.clamp(torch.sqrt(ussq), min=_scalar_like(ussq, 1e-12))
                    sf = (1.0 / torch.clamp(grms, min=1e-12)).to(b_updates[0].dtype)
                    for i in range(len(b_updates)): b_updates[i] = b_updates[i] * sf
                    if profiler_detail_enabled:
                        prof_c["foreach_bucket_global_rms"] = float(grms.detach().cpu())
                        prof_c["foreach_bucket_trust_est"] = float(trust_est.detach().cpu())
                        prof_c["foreach_bucket_source"] = bucket_src

                if use_triton_fused and len(b_params) >= foreach_min_bucket:
                    ok = self._fused_apply_triton(b_params, b_updates, prof_c)
                    if ok:
                        prof_c["foreach_update"] += 1
                        prof_c["foreach_update_tensors"] += len(b_params)
                    else:
                        torch._foreach_add_(b_params, b_updates); prof_c["foreach_update"] += 1; prof_c["foreach_update_tensors"] += len(b_params)
                else:
                    if len(b_params) >= foreach_min_bucket:
                        torch._foreach_add_(b_params, b_updates)
                        prof_c["foreach_update"] += 1
                        prof_c["foreach_update_tensors"] += len(b_params)
                    else:
                        for P, U in zip(b_params, b_updates):
                            P.add_(U)

            scalar_payload_specs = []
            if agc_clips_t is not None:
                scalar_payload_specs.append(("agc_clips", "int", agc_clips_t))
            if up_s2_t is not None:
                scalar_payload_specs.append(("up_s2", "float", up_s2_t))
            if dn_s2_t is not None:
                scalar_payload_specs.append(("dn_s2", "float", dn_s2_t))
            if lora_dense_sum_t is not None:
                scalar_payload_specs.append(("lora_dense_sum", "float", lora_dense_sum_t))
            if cross_density_sum_t is not None:
                scalar_payload_specs.append(("cross_density_sum", "float", cross_density_sum_t))
            if scalar_payload_specs:
                scalar_payload_vals = _scalars_to_host(
                    [value for _, _, value in scalar_payload_specs],
                    reference=group_ref,
                )
                for (name, kind, _), value in zip(scalar_payload_specs, scalar_payload_vals):
                    if name == "agc_clips":
                        prof_c["agc_clips"] += int(value)
                    elif name == "up_s2":
                        up_s2 += value
                    elif name == "dn_s2":
                        dn_s2 += value
                    elif name == "lora_dense_sum":
                        lora_dense_sum = value
                    elif name == "cross_density_sum":
                        cross_density_sum = value

            last_qkv = None
            if qkv_observations is not None:
                last_qkv = {}
                for label, observations in qkv_observations.items():
                    if not observations:
                        last_qkv[label] = None
                        continue
                    trusts = [item[0] for item in observations]
                    rms_values = [item[1] for item in observations]
                    tr_t = trusts[0] if len(trusts) == 1 else torch.stack(trusts).mean()
                    rms_t = rms_values[0] if len(rms_values) == 1 else torch.stack(rms_values).square().mean().sqrt()
                    low_ema_t = high_ema_t = phase_ema_t = freq_disp_t = scalar_zero
                    if collect_qkv_spectral:
                        def mean_observation(index):
                            values = [item[index] for item in observations]
                            return values[0] if len(values) == 1 else torch.stack(values).mean()

                        low_raw_t = mean_observation(2)
                        high_raw_t = mean_observation(3)
                        phase_raw_t = mean_observation(4)
                        prev = spec_state.get(label) if qkv_adapt_due else None
                        if prev is None:
                            low_ema_t, high_ema_t, phase_ema_t = low_raw_t, high_raw_t, phase_raw_t
                        else:
                            prev_low_t = _scalar_tensor(prev.get("low", low_raw_t), group_ref)
                            prev_high_t = _scalar_tensor(prev.get("high", high_raw_t), group_ref)
                            prev_phase_t = _scalar_tensor(prev.get("phase", phase_raw_t), group_ref)
                            low_ema_t = spec_beta * prev_low_t + (1.0 - spec_beta) * low_raw_t
                            high_ema_t = spec_beta * prev_high_t + (1.0 - spec_beta) * high_raw_t
                            phase_ema_t = spec_beta * prev_phase_t + (1.0 - spec_beta) * phase_raw_t
                        if qkv_adapt_due:
                            spec_state[label] = {"low": low_ema_t.detach(), "high": high_ema_t.detach(), "phase": phase_ema_t.detach()}
                        freq_disp_t = torch.log((high_ema_t + spec_eps_t) / torch.clamp(low_ema_t + spec_eps_t, min=spec_eps_t))
                    last_qkv[label] = dict(
                        tr=tr_t, rms=rms_t, freq_low=low_ema_t,
                        freq_high=high_ema_t, freq_phase=phase_ema_t,
                        freq_disp=freq_disp_t,
                    )

            # write QKV stats
            if profiler_detail_enabled and last_qkv is not None:
                qkv_payload = []
                qkv_labels = []
                for label in ("q", "k", "v"):
                    stats = last_qkv.get(label)
                    if stats is None:
                        continue
                    qkv_labels.append(label)
                    qkv_payload.extend([stats["tr"], stats["rms"], stats.get("freq_disp", 0.0)])
                if qkv_payload:
                    qkv_values = _scalars_to_host(qkv_payload, reference=group_ref)
                    for idx, label in enumerate(qkv_labels):
                        base = idx * 3
                        prof_c[f"qkv_{label}_r"] = qkv_values[base]
                        prof_c[f"qkv_{label}_rms"] = qkv_values[base + 1]
                        prof_c[f"qkv_{label}_freq"] = qkv_values[base + 2]

            # LoRA EMA + PID + inertia + minima + recovery
            if tag in ("lora_a","lora_b") and bool(self.defaults["lora_density_adapt"]) and (lora_count > 0) and (self._global_step % int(self.defaults["lora_interval"]) == 0):
                dense_obs = lora_dense_sum / max(1, lora_count)
                st = self._lora_pid.get(gi, {"ema": dense_obs, "vol_ema": 0.0, "last_ema": dense_obs,
                                             "integ_sb": 0.0, "integ_agc": 0.0, "prev_err_sb": 0.0, "prev_err_agc": 0.0,
                                             "dwell_sb":0.0, "dwell_agc":0.0, "flip_ema_sb":0.0, "flip_ema_agc":0.0,
                                             "ki_eff": self.defaults["lora_pid_ki"], "kd_eff": self.defaults["lora_pid_kd"]})
                beta = float(self.defaults["lora_density_beta"])
                st["ema"] = beta*st["ema"] + (1.0-beta)*dense_obs

                if cross_enabled:
                    pending_lora_blocks.append((gi, tag, st, st["ema"]))

                # volatility
                vol = abs(st["ema"] - st["last_ema"])
                st["vol_ema"] = float(self.defaults["lora_inertia_beta"]) * st["vol_ema"] + (1.0 - float(self.defaults["lora_inertia_beta"])) * vol
                st["last_ema"] = st["ema"]

                # targets
                sb_lo, sb_hi = self.defaults["lora_sb_bounds"]; agc_lo, agc_hi = self.defaults["lora_agc_bounds"]
                sb_tgt = max(sb_lo, min(sb_hi, sb_lo + (sb_hi - sb_lo)*st["ema"]))
                agc_tgt = max(agc_lo, min(agc_hi, agc_hi - (agc_hi - agc_lo)*st["ema"]))

                if cross_enabled and cross_gain != 0.0:
                    tag_map = cross_state.get("tag_density", {})
                    attn_vals = [tag_map[t] for t in cross_attn_tags if t in tag_map]
                    ffn_vals = [tag_map[t] for t in cross_ffn_tags if t in tag_map]
                    if attn_vals and ffn_vals:
                        attn_mean = sum(attn_vals)/len(attn_vals)
                        ffn_mean = sum(ffn_vals)/len(ffn_vals)
                        cross_signal = attn_mean - ffn_mean
                        sb_tgt = max(sb_lo, min(sb_hi, sb_tgt + cross_gain * cross_signal))
                        agc_tgt = max(agc_lo, min(agc_hi, agc_tgt - cross_gain * cross_signal))

                # inertia / flip handling for Ki/Kd
                mode = str(self.defaults["lora_pid_mode"]).lower(); strength = float(self.defaults["lora_inertia_strength"])
                flip_beta = float(self.defaults["lora_flip_ema_beta"])

                # sb
                sb_cur = float(group.get("sign_blend", sblend0))
                e_sb = sb_tgt - sb_cur
                flip_sb = 1.0 if e_sb * st["prev_err_sb"] < 0 else 0.0
                st["flip_ema_sb"] = flip_beta * st["flip_ema_sb"] + (1.0 - flip_beta) * flip_sb

                # agc
                agc_cur = float(group.get("agc_clip", agc_lo))
                e_agc = agc_tgt - agc_cur
                flip_agc = 1.0 if e_agc * st["prev_err_agc"] < 0 else 0.0
                st["flip_ema_agc"] = flip_beta * st["flip_ema_agc"] + (1.0 - flip_beta) * flip_agc

                # dynamic Ki/Kd targets
                ki0 = float(self.defaults["lora_pid_ki"]); kd0 = float(self.defaults["lora_pid_kd"])
                scale_vol = (1.2 if mode in {"inertial","auto"} else (0.6 if mode=="stable" else 0.3))
                scale_flip= (1.5 if mode in {"inertial","auto"} else (0.8 if mode=="stable" else 0.4))
                ki_target = ki0 / (1.0 + strength * st["vol_ema"] * scale_vol)
                kd_target = kd0 / (1.0 + strength * st["flip_ema_sb"] * scale_flip)
                ki_min = float(self.defaults["lora_ki_min"]); kd_min = float(self.defaults["lora_kd_min"])
                ki_target = max(ki_min, ki_target); kd_target = max(kd_min, kd_target)

                # recovery (slow up, fast down)
                rec = float(self.defaults["lora_recover_rate"])
                ki_eff = st.get("ki_eff", ki0); kd_eff = st.get("kd_eff", kd0)
                ki_eff = ki_target if ki_target < ki_eff else (ki_eff + rec*(ki_target - ki_eff))
                kd_eff = kd_target if kd_target < kd_eff else (kd_eff + rec*(kd_target - kd_eff))
                st["ki_eff"] = ki_eff; st["kd_eff"] = kd_eff

                # integrate + derivative
                clip_sb = float(self.defaults["lora_int_clip_sb"]); clip_agc = float(self.defaults["lora_int_clip_agc"])
                st["integ_sb"] = max(-clip_sb, min(clip_sb, st["integ_sb"] + e_sb))
                st["integ_agc"] = max(-clip_agc, min(clip_agc, st["integ_agc"] + e_agc))
                d_sb = e_sb - st["prev_err_sb"]; d_agc = e_agc - st["prev_err_agc"]

                kp = float(self.defaults["lora_pid_kp"])
                delta_sb  = kp*e_sb  + ki_eff*st["integ_sb"]  + kd_eff*d_sb
                delta_agc = kp*e_agc + ki_eff*st["integ_agc"] + kd_eff*d_agc

                new_sb = max(sb_lo, min(sb_hi, sb_cur + delta_sb))
                new_agc = max(agc_lo, min(agc_hi, agc_cur + delta_agc))

                # dwell decay & saturation reset
                aw_eps = float(self.defaults["lora_aw_eps"]); dwell_decay = float(self.defaults["lora_dwell_decay"]); dwell_gain = float(self.defaults["lora_dwell_gain"])
                if abs(new_sb - sb_lo) < aw_eps and e_sb < 0: st["dwell_sb"] = st.get("dwell_sb",0.0) + 1.0
                elif abs(new_sb - sb_hi) < aw_eps and e_sb > 0: st["dwell_sb"] = st.get("dwell_sb",0.0) + 1.0
                else: st["dwell_sb"] *= dwell_decay
                if st["dwell_sb"] > 0: st["integ_sb"] *= (1.0 / (1.0 + dwell_gain * st["dwell_sb"]))
                if bool(self.defaults["lora_aw_reset_on_saturation"]) and ((abs(new_sb - sb_lo) < aw_eps and e_sb < 0) or (abs(new_sb - sb_hi) < aw_eps and e_sb > 0)):
                    st["integ_sb"] = 0.0

                if abs(new_agc - agc_lo) < aw_eps and e_agc < 0: st["dwell_agc"] = st.get("dwell_agc",0.0) + 1.0
                elif abs(new_agc - agc_hi) < aw_eps and e_agc > 0: st["dwell_agc"] = st.get("dwell_agc",0.0) + 1.0
                else: st["dwell_agc"] *= dwell_decay
                if st["dwell_agc"] > 0: st["integ_agc"] *= (1.0 / (1.0 + dwell_gain * st["dwell_agc"]))
                if bool(self.defaults["lora_aw_reset_on_saturation"]) and ((abs(new_agc - agc_lo) < aw_eps and e_agc < 0) or (abs(new_agc - agc_hi) < aw_eps and e_agc > 0)):
                    st["integ_agc"] = 0.0

                st["prev_err_sb"] = e_sb; st["prev_err_agc"] = e_agc
                self._lora_pid[gi] = st

                # apply or stage
                if not (_is_compiling() and self.defaults.get("compile_guard", True)):
                    group["sign_blend"] = new_sb
                    group["agc_clip"]   = new_agc
                else:
                    self.stage_group_update(gi, {"sign_blend": new_sb, "agc_clip": new_agc}, policy="replace")
                self._last_metrics.update({"lora_dense_ema": st["ema"], "lora_vol_ema": st["vol_ema"], "lora_flip_ema_sb": st["flip_ema_sb"], "lora_flip_ema_agc": st["flip_ema_agc"],
                                           "lora_sb": new_sb, "lora_agc": new_agc, "lora_ki_eff": ki_eff, "lora_kd_eff": kd_eff})
                if cross_enabled:
                    pending_lora_blocks[-1] = (gi, tag, st, st["ema"])

            # QKV two-objective with gamma auto-scale + step-clip via acceleration
            if qkv_adapt_due and last_qkv is not None:
                if last_qkv["q"] and last_qkv["k"] and last_qkv["v"]:
                    wr = float(self.defaults["qkv_w_rms"]); wt = float(self.defaults["qkv_w_trust"])
                    base_gain = float(self.defaults["qkv_lr_gain"]); gamma0 = float(self.defaults["qkv_gain_shrink_gamma"])
                    b_lo, b_hi = self.defaults["qkv_lr_bounds"]; beta = float(self.defaults["qkv_lr_ema_beta"])
                    base_clip = float(self.defaults["qkv_lr_step_clip"])
                    beta_disp = float(self.defaults["qkv_disp_ema_beta"]); rate = float(self.defaults["qkv_gamma_rate"])
                    gmin = float(self.defaults["qkv_gamma_min"]); gmax = float(self.defaults["qkv_gamma_max"])
                    k_shrink = float(self.defaults["qkv_clip_shrink_k"]); cmin = float(self.defaults["qkv_clip_min"]); cmax=float(self.defaults["qkv_clip_max"])

                    qkv_stat_values = _scalars_to_host(
                        [
                            last_qkv[k]["rms"] for k in ("q", "k", "v")
                        ] + [
                            last_qkv[k]["tr"] for k in ("q", "k", "v")
                        ] + [
                            last_qkv[k].get("freq_disp", 0.0) for k in ("q", "k", "v")
                        ] + [
                            last_qkv[k].get("freq_low", 0.0) for k in ("q", "k", "v")
                        ] + [
                            last_qkv[k].get("freq_high", 0.0) for k in ("q", "k", "v")
                        ] + [
                            last_qkv[k].get("freq_phase", 0.0) for k in ("q", "k", "v")
                        ],
                        reference=group_ref,
                    )
                    rms = [max(1e-12, value) for value in qkv_stat_values[0:3]]
                    trs = [max(1e-12, value) for value in qkv_stat_values[3:6]]
                    freq_disp_vals = qkv_stat_values[6:9]
                    freq_low_vals = qkv_stat_values[9:12]
                    freq_high_vals = qkv_stat_values[12:15]
                    phase_vals = qkv_stat_values[15:18]

                    disp_r = math.log(max(rms)/min(rms)); disp_t = math.log(max(trs)/min(trs))
                    disp = math.sqrt(wr*disp_r*disp_r + wt*disp_t*disp_t)
                    ema_prev = self._qkv_disp_ema.get(gi, disp)
                    ema_prev2 = self._qkv_disp_ema_prev.get(gi, disp)
                    disp_ema = beta_disp*ema_prev + (1.0 - beta_disp)*disp
                    self._qkv_disp_ema_prev[gi] = ema_prev
                    self._qkv_disp_ema[gi] = disp_ema
                    trend = disp - ema_prev
                    accel = disp_ema - 2*ema_prev + ema_prev2  # 2nd diff (EMA-based)

                    # gamma auto-scale
                    gamma_eff = gamma0 * (1.0 + rate * (trend / max(1e-6, disp)))
                    gamma_eff = max(gmin, min(gmax, gamma_eff))
                    freq_disp_mean = sum(freq_disp_vals)/len(freq_disp_vals)
                    freq_factor = 1.0
                    if spec_enabled:
                        freq_gain = float(self.defaults.get("qkv_gamma_spectral_gain", 0.0))
                        clip_lo, clip_hi = self.defaults.get("qkv_gamma_spectral_clip", (0.5, 1.5))
                        if freq_gain != 0.0:
                            freq_factor = math.exp(freq_gain * freq_disp_mean)
                            freq_factor = max(float(clip_lo), min(float(clip_hi), freq_factor))
                            if spec_strength < 1.0:
                                freq_factor = 1.0 + spec_strength * (freq_factor - 1.0)
                            gamma_eff *= freq_factor
                    # step-clip shrink by positive acceleration
                    step_clip_eff = base_clip / (1.0 + k_shrink * max(0.0, accel))
                    phase_mean = sum(phase_vals)/len(phase_vals) if phase_vals else 0.0
                    phase_boost = 1.0
                    if spec_enabled:
                        phase_gain = float(self.defaults.get("qkv_phase_boost_gain", 0.0))
                        phase_target = float(self.defaults.get("qkv_phase_target", 0.0))
                        clip_lo, clip_hi = self.defaults.get("qkv_phase_boost_clip", (0.6, 1.6))
                        if phase_gain != 0.0:
                            phase_boost = 1.0 + phase_gain * (phase_mean - phase_target)
                            phase_boost = max(float(clip_lo), min(float(clip_hi), phase_boost))
                            if spec_strength < 1.0:
                                phase_boost = 1.0 + spec_strength * (phase_boost - 1.0)
                    step_clip_eff = max(cmin, min(cmax, step_clip_eff * phase_boost))

                    med_r = sorted(rms)[1]; med_t = sorted(trs)[1]
                    def adj(cur_r, cur_t):
                        e = wr*math.log(cur_r/med_r) + wt*math.log(cur_t/med_t)
                        delta = math.exp(- (base_gain / (1.0 + gamma_eff * disp)) * e)
                        delta = max(1.0 - step_clip_eff, min(1.0 + step_clip_eff, delta))
                        return delta

                    sc_q = max(b_lo, min(b_hi, float(qkv_lr.get("q",1.0)) * adj(rms[0], trs[0])))
                    sc_k = max(b_lo, min(b_hi, float(qkv_lr.get("k",1.0)) * adj(rms[1], trs[1])))
                    sc_v = max(b_lo, min(b_hi, float(qkv_lr.get("v",1.0)) * adj(rms[2], trs[2])))

                    ema = self._qkv_lr_ema.get(gi, {"q":sc_q,"k":sc_k,"v":sc_v})
                    sc_q = beta*ema["q"] + (1.0-beta)*sc_q
                    sc_k = beta*ema["k"] + (1.0-beta)*sc_k
                    sc_v = beta*ema["v"] + (1.0-beta)*sc_v
                    self._qkv_lr_ema[gi] = {"q":sc_q,"k":sc_k,"v":sc_v}
                    new_map = dict(q=sc_q, k=sc_k, v=sc_v)
                    if not (_is_compiling() and self.defaults.get("compile_guard", True)):
                        group["qkv_lr_scales"] = new_map
                    else:
                        self.stage_group_update(gi, {"qkv_lr_scales": new_map}, policy="replace")
                    self._last_metrics.update({"qkv_lr_q": sc_q, "qkv_lr_k": sc_k, "qkv_lr_v": sc_v,
                                               "qkv_gamma_eff": gamma_eff, "qkv_disp": disp, "qkv_disp_ema": disp_ema,
                                               "qkv_step_clip_eff": step_clip_eff, "qkv_accel": accel,
                                               "qkv_freq_disp": freq_disp_mean, "qkv_freq_factor": freq_factor,
                                               "qkv_phase_boost": phase_boost,
                                               "qkv_spectral_strength": spec_strength,
                                               "qkv_freq_low": sum(freq_low_vals)/len(freq_low_vals) if freq_low_vals else 0.0,
                                               "qkv_freq_high": sum(freq_high_vals)/len(freq_high_vals) if freq_high_vals else 0.0,
                                               "qkv_phase_mean": phase_mean})
                    if profiler_detail_enabled:
                        prof_c.update(qkv_freq_disp=freq_disp_mean, qkv_freq_factor=freq_factor, qkv_phase_boost=phase_boost)

            # end group loop

            if cross_enabled and cross_density_count > 0:
                obs = cross_density_sum / max(1, cross_density_count)
                tag_map = cross_state.setdefault("tag_density", {})
                prev = tag_map.get(tag)
                if prev is None:
                    prev = obs
                tag_map[tag] = cross_beta*prev + (1.0 - cross_beta)*obs

        if cross_enabled:
            tag_map = cross_state.setdefault("tag_density", {})
            if pending_lora_blocks:
                mean_density = sum(item[3] for item in pending_lora_blocks) / len(pending_lora_blocks)
                prev_global = cross_state.get("global_density")
                new_global = mean_density if prev_global is None else cross_beta*prev_global + (1.0 - cross_beta)*mean_density
                cross_state["global_density"] = new_global
                for (_, tag, st, ema_val) in pending_lora_blocks:
                    prev_tag = tag_map.get(tag, ema_val)
                    tag_map[tag] = cross_beta*prev_tag + (1.0 - cross_beta)*ema_val
                if cross_sync > 0.0 and cross_state["global_density"] is not None:
                    global_density = float(cross_state["global_density"])
                    for (gi, tag, st, _) in pending_lora_blocks:
                        st["ema"] = (1.0 - cross_sync) * st["ema"] + cross_sync * global_density
                        self._lora_pid[gi] = st
            global_density_val = cross_state.get("global_density")
            if global_density_val is not None or tag_map:
                attn_avg = sum(tag_map.get(t, 0.0) for t in cross_attn_tags) / max(1, len(cross_attn_tags)) if cross_attn_tags else 0.0
                ffn_avg = sum(tag_map.get(t, 0.0) for t in cross_ffn_tags) / max(1, len(cross_ffn_tags)) if cross_ffn_tags else 0.0
                if global_density_val is not None:
                    self._last_metrics["lora_global_density"] = float(global_density_val)
                self._last_metrics["lora_attn_density"] = float(attn_avg)
                self._last_metrics["lora_ffn_density"] = float(ffn_avg)

        if cross_enabled and bool(self.defaults.get("lora_bridge", False)):
            global_density = cross_state.get("global_density", None)
            if global_density is not None:
                bridge_gain = float(self.defaults.get("lora_bridge_gain", 0.0))
                bridge_beta = float(self.defaults.get("lora_bridge_beta", 0.0))
                b_lo, b_hi = self.defaults.get("lora_bridge_bounds", (0.5, 1.5))
                apply_dict_mut = not (_is_compiling() and self.defaults.get("compile_guard", True))
                tag_map = cross_state.get("tag_density", {})
                target_tags = set(cross_attn_tags) | set(cross_ffn_tags)
                if bridge_gain != 0.0 and target_tags:
                    for gi, group in enumerate(self.param_groups):
                        tag = group.get("block_tag", "default")
                        if tag not in target_tags:
                            continue
                        density_val = tag_map.get(tag)
                        if density_val is None:
                            continue
                        bridge_state = self._lora_bridge.setdefault(gi, {
                            "base": float(group.get("trust_clip", self.defaults.get("trust_clip", 10.0))),
                            "ema": 1.0,
                        })
                        base = bridge_state["base"]
                        target_scale = math.exp(bridge_gain * (density_val - global_density))
                        target_scale = min(b_hi, max(b_lo, target_scale))
                        new_scale = bridge_beta*bridge_state["ema"] + (1.0 - bridge_beta)*target_scale
                        bridge_state["ema"] = new_scale
                        new_trust = base * new_scale
                        if apply_dict_mut:
                            group["trust_clip"] = new_trust
                        else:
                            self.stage_group_update(gi, {"trust_clip": new_trust}, policy="replace")
                    self._last_metrics.update({
                        "lora_bridge_global": float(global_density),
                    })

        # Auto FFN Asym & sign blend
        d = self.defaults
        apply_dict_mut = not (_is_compiling() and d.get("compile_guard", True))
        if any(g.get("auto_ffn_asym", d["auto_ffn_asym"]) for g in self.param_groups):
            beta = float(d["ffn_asym_beta"]); tgt = float(d["ffn_asym_target"]); gain = float(d["ffn_asym_gain"])
            interval = int(d["ffn_asym_interval"]); smin = float(d["ffn_lr_min"]); smax = float(d["ffn_lr_max"])
            st = self._ffn
            if up_n>0:
                up_rms=(up_s2/max(1,up_n))**0.5; st["ema_up"]= beta*(st["ema_up"] if st["ema_up"] is not None else up_rms)+(1-beta)*up_rms
            if dn_n>0:
                dn_rms=(dn_s2/max(1,dn_n))**0.5; st["ema_down"]= beta*(st["ema_down"] if st["ema_down"] is not None else dn_rms)+(1-beta)*dn_rms
            if (self._global_step - st.get("last",0) >= interval) and (st["ema_up"] is not None) and (st["ema_down"] is not None):
                ratio = st["ema_up"]/(st["ema_down"]+1e-12)
                e = math.log(max(1e-12, ratio/tgt)); up_adj=math.exp(-gain*e); dn_adj=math.exp(+gain*e)
                if apply_dict_mut:
                    for gi, g in enumerate(self.param_groups):
                        tg=g.get("block_tag","default")
                        if tg=="ffn_up":
                            g["lr_scale"]= float(min(smax, max(smin, g.get("lr_scale",1.0)*up_adj)))
                        elif tg=="ffn_down":
                            g["lr_scale"]= float(min(smax, max(smin, g.get("lr_scale",1.0)*dn_adj)))
                else:
                    for gi, g in enumerate(self.param_groups):
                        tg=g.get("block_tag","default")
                        if tg=="ffn_up":
                            self._pending_lr_scale[gi] = float(self._pending_lr_scale.get(gi, 1.0) * up_adj)
                        elif tg=="ffn_down":
                            self._pending_lr_scale[gi] = float(self._pending_lr_scale.get(gi, 1.0) * dn_adj)
                st["last"]=self._global_step
                self._last_metrics.update({
                    "ffn_lr_scale_up":next((g.get("lr_scale",1.0) for g in self.param_groups if g.get("block_tag")=="ffn_up"),1.0),
                    "ffn_lr_scale_down":next((g.get("lr_scale",1.0) for g in self.param_groups if g.get("block_tag")=="ffn_down"),1.0),
                    "ffn_ratio": float(ratio)
                })

        if not (_is_compiling() and self.defaults.get("compile_guard", True)):
            self._maybe_plateau_adapt()

        if self._fused_apply_failures:
            prof_c["foreach_triton_failures"] = list(self._fused_apply_failures)
        else:
            prof_c["foreach_triton_failures"] = None

        step_ms = 1000.0 * (time.perf_counter() - t0)
        self._prof.log_step(step_ms, self._global_step, payload=prof_c)

        return loss

    def get_last_metrics(self) -> Dict[str, float]:
        out = dict(self._last_metrics)
        out.update(self._prof.last_payload())
        return out
