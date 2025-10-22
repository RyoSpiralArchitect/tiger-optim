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
import logging
import math, time, os, json
from collections.abc import Mapping, Sequence
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.optim import Optimizer

from .accel import fast_norm, fast_rms, fast_softsign


_LOG = logging.getLogger(__name__)


def _rms(x):
    return fast_rms(x)


def _norm(x):
    return fast_norm(x)


def _softsign(x, tau):
    return fast_softsign(x, tau)
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
    if device is None:
        return torch.tensor(value, dtype=target_dtype)
    return torch.tensor(value, dtype=target_dtype, device=device)


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

class _Profiler:
    def __init__(self, enabled=False, path="bench/profiles/tiger.jsonl", interval=10, ema_decay=0.9):
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

def _apply_field_spec(old_val, spec):
    """Support: ('op', payload) or a pipeline list [(op,payload), ...]."""
    # pipeline
    if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], (list, tuple)) and len(spec[0])==2:
        val = old_val
        for op, payload in spec:
            val = _merge_policy(val, payload, str(op))
        return val
    # single op
    if isinstance(spec, (tuple, list)) and len(spec)==2 and str(spec[0]) in {"add","mul","clip","replace","set","merge","pow","exp","logit-clip"}:
        return _merge_policy(old_val, spec[1], str(spec[0]))
    return spec

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
                 # QKV two-objective + gamma auto-scale + step-clip acceleration shrink
                 qkv_lr_autoadapt=True, qkv_w_rms=0.7, qkv_w_trust=0.3,
                 qkv_lr_gain=0.02, qkv_lr_bounds=(0.7, 1.3), qkv_lr_interval=25, qkv_lr_ema_beta=0.8,
                 qkv_gain_shrink_gamma=1.2, qkv_lr_step_clip=0.08,
                 qkv_disp_ema_beta=0.8, qkv_gamma_rate=0.6, qkv_gamma_min=0.2, qkv_gamma_max=5.0,
                 qkv_clip_shrink_k=0.6, qkv_clip_min=0.02, qkv_clip_max=0.2,
                 # compile staged reflect
                 compile_guard=True, reflect_interval=50,
                 # profiler
                 profiler_enabled=False, profiler_path="bench/profiles/tiger.jsonl", profiler_interval=10, profiler_ema_decay=0.9):
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
                        # QKV 2obj + gamma auto-scale + step-clip accel shrink
                        qkv_lr_autoadapt=qkv_lr_autoadapt, qkv_w_rms=float(qkv_w_rms), qkv_w_trust=float(qkv_w_trust),
                        qkv_lr_gain=float(qkv_lr_gain), qkv_lr_bounds=tuple(qkv_lr_bounds), qkv_lr_interval=int(qkv_lr_interval), qkv_lr_ema_beta=float(qkv_lr_ema_beta),
                        qkv_gain_shrink_gamma=float(qkv_gain_shrink_gamma), qkv_lr_step_clip=float(qkv_lr_step_clip),
                        qkv_disp_ema_beta=float(qkv_disp_ema_beta), qkv_gamma_rate=float(qkv_gamma_rate), qkv_gamma_min=float(qkv_gamma_min), qkv_gamma_max=float(qkv_gamma_max),
                        qkv_clip_shrink_k=float(qkv_clip_shrink_k), qkv_clip_min=float(qkv_clip_min), qkv_clip_max=float(qkv_clip_max),
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
        self._trust_ema: Dict[int, float] = {}
        # LoRA PID state per group id
        self._lora_pid: Dict[int, Dict[str, float]] = {}  # plus ki_eff/kd_eff for recovery
        # QKV LR EMA per group & dispersion EMA (and previous for accel)
        self._qkv_lr_ema: Dict[int, Dict[str, float]] = {}
        self._qkv_disp_ema: Dict[int, float] = {}
        self._qkv_disp_ema_prev: Dict[int, float] = {}
        self._fused_apply_failures: List[str] = []

    # ---------- Public API ----------
    def report_metrics(self, loss: Optional[float] = None, **kwargs):
        if loss is not None:
            self._last_metrics["loss"] = float(loss)
            p = self._plateau
            if p["best"] is None or loss < p["best"] - self.defaults["plateau_tol"]:
                p["best"] = float(loss); p["streak"] = 0
            else:
                p["streak"] += 1
            p["last_loss"] = float(loss)
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
                            g[key] = _apply_field_spec(g.get(key), val)
                            applied += 1
                del self._pending_group_updates[gi]
        self._last_reflect = self._global_step
        return applied

    def stage_group_update(self, gi: int, fields: Dict[str, object], policy: str = "replace"):
        self._pending_group_updates.setdefault(int(gi), []).append((dict(fields), str(policy)))

    # ---------- Bucket stats helpers ----------
    def _bucket_global_rms(self, updates: List[torch.Tensor], source="global", *, reference: Optional[torch.Tensor] = None):
        ref_tensor = reference if reference is not None else (updates[0] if updates else None)
        if not updates:
            ref_device = ref_tensor.device if ref_tensor is not None else None
            return _scalar_like(ref_tensor, 1.0, dtype=torch.float32, device=ref_device)
        ups32 = [u.to(torch.float32) for u in updates]
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

        if not params or not updates:
            record("empty")
            return False

        device = params[0].device
        if device.type != "cuda":
            record("non_cuda_bucket")
            return False

        if not torch.cuda.is_available():
            record("no_cuda")
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
        return True

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
    def _maybe_auto_blend(self):
        d = self.defaults
        if not d["auto_blend"] or d["blend_steps"] > 0: return
        p = self._plateau
        if p["best"] is None: return
        if p["streak"] >= d["plateau_patience"]:
            lo, hi = d["auto_blend_bounds"]
            for g in self.param_groups:
                sb = float(g.get("sign_blend", d["sign_blend"]))
                sb = min(hi, sb + d["auto_blend_gain"])
                g["sign_blend"] = sb
            p["streak"] = 0
            self._last_metrics["auto_blend_bump"] = 1.0

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

        self._global_step += 1
        self._last_metrics.pop("auto_blend_bump", None)

        # staged reflect cadence
        if (not _is_compiling()) and self.defaults.get("compile_guard", True):
            if (self._pending_lr_scale or self._pending_group_updates) and (self._global_step - self._last_reflect >= int(self.defaults["reflect_interval"])):
                self.reflect_pending()

        # profiler counters
        prof_c = dict(foreach_wd=0, foreach_update=0, foreach_update_tensors=0,
                      foreach_bucket_size=0, foreach_bucket_global_rms=None, foreach_bucket_trust_est=None, foreach_bucket_source=None,
                      triton_fused_apply=None, triton_fused_apply_reason=None,
                      agc_clips=0, nonfinite_skips=0, pruned_params=0, pruned_elems=0,
                      foreach_triton_failures=None,
                      qkv_q_r=None, qkv_k_r=None, qkv_v_r=None, qkv_q_rms=None, qkv_k_rms=None, qkv_v_rms=None,
                      qkv_gamma_eff=None, qkv_disp=None, qkv_disp_ema=None, qkv_step_clip_eff=None, qkv_accel=None)

        self._fused_apply_failures = []

        # sparse state pruning flags
        sprune_thr = float(self.defaults["state_prune_threshold"])
        sprune_int = int(self.defaults["state_prune_interval"])
        do_prune = (sprune_thr > 0.0) and (sprune_int > 0) and (self._global_step % sprune_int == 0)

        # ---------- Pre-pass: foreach weight decay ----------
        for g in self.param_groups:
            wd = float(g["weight_decay"]); lr = float(g["lr"]); lr_scale = float(g.get("lr_scale", 1.0))
            if wd == 0.0: continue
            plist = [p for p in g["params"] if p.grad is not None]
            if not plist: continue
            if g["use_foreach"] and hasattr(torch, "_foreach_add_"):
                torch._foreach_add_(plist, plist, alpha=-(lr * lr_scale * wd))
                prof_c["foreach_wd"] += len(plist)
            else:
                for p in plist: p.add_(p, alpha=-(lr * lr_scale * wd))

        # ---------- Main pass ----------
        up_s2 = 0.0; up_n = 0
        dn_s2 = 0.0; dn_n = 0

        for gi, group in enumerate(self.param_groups):
            lr = float(group["lr"]); (b1,b2) = group["betas"]; eps = float(group["eps"])
            factored = bool(group["factored"]); alpha = float(group["precond_alpha"])
            sblend0 = float(group["sign_blend"]); sblendT = int(group["blend_steps"]); sblendTo = float(group["blend_to"]); sched = group["blend_schedule"]
            use_trust = bool(group["use_trust_ratio"]); trust_clip = float(group["trust_clip"]); trust_space = str(group.get("trust_space","update"))
            trust_beta = float(group.get("trust_ema_beta", self.defaults["trust_ema_beta"]))
            agc_clip = float(group["agc_clip"]); agc_eps = float(group["agc_eps"])
            lr_scale = float(group.get("lr_scale", 1.0)); tag = group.get("block_tag","default")
            rms_thr = float(group.get("rms_clip_threshold", 0.0)); rms_gran = group.get("rms_clip_granularity","param")
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

            # QKV slice logging (last seen in this group)
            last_qkv = {"q":None,"k":None,"v":None}

            for p in group["params"]:
                if p.grad is None: continue

                if bool(self.defaults["skip_if_nonfinite"]):
                    if (not torch.isfinite(p).all()) or (not torch.isfinite(p.grad).all()):
                        prof_c["nonfinite_skips"] += 1
                        continue

                g = p.grad
                st = self.state[p]
                dt = self._mom_storage_dtype
                if "m" not in st:
                    st["m"] = torch.zeros_like(p, dtype=dt)
                m = st["m"]

                if factored and p.ndim>=2:
                    rows = p.shape[0]; cols = p.numel()//rows
                    if "vr" not in st:
                        st["vr"] = torch.zeros(rows, dtype=dt, device=p.device)
                        st["vc"] = torch.zeros(cols, dtype=dt, device=p.device)
                    vr = st["vr"]; vc = st["vc"]
                else:
                    if "v" not in st:
                        st["v"] = torch.zeros_like(p, dtype=dt)
                    v = st["v"]

                # m update
                m32 = _to_dtype(m, torch.float32); g32 = _to_dtype(g, torch.float32)
                m32.mul_(b1).add_(g32, alpha=1.0-b1)
                if do_prune and float(self.defaults["state_prune_threshold"]) > 0.0:
                    thr = float(self.defaults["state_prune_threshold"])
                    m32.masked_fill_(m32.abs() < thr, 0.0)
                m.copy_(m32.to(m.dtype))

                # vhat
                if factored and p.ndim>=2:
                    g2 = g32.view(p.shape[0], -1).pow(2)
                    vr32 = _to_dtype(vr, torch.float32); vc32 = _to_dtype(vc, torch.float32)
                    vr32.mul_(b2).add_(g2.mean(dim=1), alpha=1.0-b2)
                    vc32.mul_(b2).add_(g2.mean(dim=0), alpha=1.0-b2)
                    vr.copy_(vr32.to(vr.dtype)); vc.copy_(vc32.to(vc.dtype))
                    vhat = torch.outer(vr32.clamp_min(1e-30), vc32.clamp_min(1e-30)) / vr32.mean().clamp_min(1e-30)
                    vhat = vhat.view_as(p)
                else:
                    if "v" not in st: st["v"] = torch.zeros_like(p, dtype=dt)
                    v = st["v"]
                    v32 = _to_dtype(v, torch.float32)
                    v32.mul_(b2).addcmul_(g32, g32, value=1.0-b2)
                    if do_prune and float(self.defaults["state_prune_threshold"]) > 0.0:
                        thr = float(self.defaults["state_prune_threshold"])
                        v32.masked_fill_(v32.abs() < thr, 0.0)
                    v.copy_(v32.to(v.dtype)); vhat = v32

                P = (vhat.sqrt().add_(eps)).pow(-float(alpha)) if alpha>0 else 1.0

                # Direction
                if group["sign_mode"] == "softsign":
                    d_sign = _softsign(m32, float(group["sign_tau"])) * P
                else:
                    d_sign = torch.sign(m32) * P
                d_mag = (m32 / (_rms(m32)+eps)) * P
                d = (1.0 - sblend) * d_sign + sblend * d_mag

                # AGC
                if agc_clip and agc_clip > 0.0:
                    p_n = float(_norm(p)); d_n = float(_norm(d))
                    max_n = agc_clip * (p_n + agc_eps)
                    if d_n > max_n and d_n > 0.0:
                        d = d * (max_n / d_n); prof_c["agc_clips"] += 1

                if tag=="ffn_up":
                    up_s2 += float((d.float()*d.float()).sum().item()); up_n += d.numel()
                elif tag=="ffn_down":
                    dn_s2 += float((d.float()*d.float()).sum().item()); dn_n += d.numel()

                if tag in ("lora_a","lora_b") and bool(self.defaults["lora_density_adapt"]):
                    thr = float(self.defaults["lora_density_k"]) * float(_rms(d))
                    if thr > 0.0:
                        dense = float((d.abs() > thr).float().mean().item())
                        lora_dense_sum += dense; lora_count += 1

                base_lr = lr * lr_scale

                # RMS clip (param)
                clip_scale_param = 1.0
                if rms_thr and rms_gran == "param":
                    rr = float(_rms(d))
                    clip_scale_param = min(1.0, rms_thr/rr) if rr>0 else 1.0

                # trust
                def eff_trust_for(_p, _d):
                    if not use_trust: return 1.0
                    if trust_space == "update":
                        pn = float(_norm(_p)); dn = float(_norm(_d)); raw = (pn/(dn+1e-12)) if (pn>0 and dn>0) else 1.0
                    elif trust_space == "precond":
                        pn = float(_norm(_p))
                        denom_n = float(_norm(m32 / (vhat.sqrt().add(eps).pow(alpha)))); raw = (pn/(denom_n+1e-12)) if (pn>0 and denom_n>0) else 1.0
                    else:
                        raw = 1.0
                    if trust_beta > 0.0:
                        old = self._trust_ema.get(gi, raw); sm = trust_beta*old + (1.0-trust_beta)*raw; self._trust_ema[gi] = sm; return min(trust_clip, sm)
                    return min(trust_clip, raw)

                # chunked paths
                chunked = False
                if (blk_chunks and p.ndim==2 and p.shape[0] % blk_chunks == 0) or (id(p) in qkv_rules and bool(qkv_split) and use_trust):
                    chunked = True
                    if id(p) in qkv_rules:
                        dim, parts = qkv_rules[id(p)]
                        p_chunks = torch.chunk(p, parts, dim=dim)
                        d_chunks = torch.chunk(d, parts, dim=dim)
                        keys = ("q","k","v")
                        for i, (pc, dc) in enumerate(zip(p_chunks, d_chunks)):
                            tr = eff_trust_for(pc, dc)
                            eff_lr = base_lr
                            if qkv_lr is not None and i < len(keys):
                                eff_lr = lr * float(qkv_lr.get(keys[i], lr_scale))
                            k = keys[i]; last_qkv[k] = dict(tr=float(tr), rms=float(_rms(dc)))
                            pc.add_(dc.to(pc.dtype), alpha=-(eff_lr * tr * clip_scale_param))
                    else:
                        parts = blk_chunks
                        p_chunks = torch.chunk(p, parts, dim=0)
                        d_chunks = torch.chunk(d, parts, dim=0)
                        for pc, dc in zip(p_chunks, d_chunks):
                            tr = eff_trust_for(pc, dc)
                            pc.add_(dc.to(pc.dtype), alpha=-(base_lr * tr * clip_scale_param))

                # foreach bucket
                if not chunked:
                    eff = -(base_lr * eff_trust_for(p, d) * clip_scale_param)
                    if use_foreach_upd and hasattr(torch, "_foreach_add_"):
                        upd = d.to(self._dtype_for_update(p)) * eff
                        b_params.append(p); b_updates.append(upd); 
                        if bucket_std:
                            b_updates_f32.append(upd.to(torch.float32))
                    else:
                        p.add_(d.to(p.dtype), alpha=eff)

            # flush foreach bucket (combined stats & scale)
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

            # write QKV stats
            if last_qkv["q"] is not None:
                prof_c["qkv_q_r"] = last_qkv["q"]["tr"]; prof_c["qkv_q_rms"] = last_qkv["q"]["rms"]
            if last_qkv["k"] is not None:
                prof_c["qkv_k_r"] = last_qkv["k"]["tr"]; prof_c["qkv_k_rms"] = last_qkv["k"]["rms"]
            if last_qkv["v"] is not None:
                prof_c["qkv_v_r"] = last_qkv["v"]["tr"]; prof_c["qkv_v_rms"] = last_qkv["v"]["rms"]

            # LoRA EMA + PID + inertia + minima + recovery
            if tag in ("lora_a","lora_b") and bool(self.defaults["lora_density_adapt"]) and (lora_count > 0) and (self._global_step % int(self.defaults["lora_interval"]) == 0):
                dense_obs = lora_dense_sum / max(1, lora_count)
                st = self._lora_pid.get(gi, {"ema": dense_obs, "vol_ema": 0.0, "last_ema": dense_obs,
                                             "integ_sb": 0.0, "integ_agc": 0.0, "prev_err_sb": 0.0, "prev_err_agc": 0.0,
                                             "dwell_sb":0.0, "dwell_agc":0.0, "flip_ema_sb":0.0, "flip_ema_agc":0.0,
                                             "ki_eff": self.defaults["lora_pid_ki"], "kd_eff": self.defaults["lora_pid_kd"]})
                beta = float(self.defaults["lora_density_beta"])
                st["ema"] = beta*st["ema"] + (1.0-beta)*dense_obs

                # volatility
                vol = abs(st["ema"] - st["last_ema"])
                st["vol_ema"] = float(self.defaults["lora_inertia_beta"]) * st["vol_ema"] + (1.0 - float(self.defaults["lora_inertia_beta"])) * vol
                st["last_ema"] = st["ema"]

                # targets
                sb_lo, sb_hi = self.defaults["lora_sb_bounds"]; agc_lo, agc_hi = self.defaults["lora_agc_bounds"]
                sb_tgt = max(sb_lo, min(sb_hi, sb_lo + (sb_hi - sb_lo)*st["ema"]))
                agc_tgt = max(agc_lo, min(agc_hi, agc_hi - (agc_hi - agc_lo)*st["ema"]))

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

            # QKV two-objective with gamma auto-scale + step-clip via acceleration
            if tag == "attn_qkv" and qkv_lr is not None and bool(self.defaults["qkv_lr_autoadapt"]) and (self._global_step % int(self.defaults["qkv_lr_interval"]) == 0):
                if last_qkv["q"] and last_qkv["k"] and last_qkv["v"]:
                    wr = float(self.defaults["qkv_w_rms"]); wt = float(self.defaults["qkv_w_trust"])
                    base_gain = float(self.defaults["qkv_lr_gain"]); gamma0 = float(self.defaults["qkv_gain_shrink_gamma"])
                    b_lo, b_hi = self.defaults["qkv_lr_bounds"]; beta = float(self.defaults["qkv_lr_ema_beta"])
                    base_clip = float(self.defaults["qkv_lr_step_clip"])
                    beta_disp = float(self.defaults["qkv_disp_ema_beta"]); rate = float(self.defaults["qkv_gamma_rate"])
                    gmin = float(self.defaults["qkv_gamma_min"]); gmax = float(self.defaults["qkv_gamma_max"])
                    k_shrink = float(self.defaults["qkv_clip_shrink_k"]); cmin = float(self.defaults["qkv_clip_min"]); cmax=float(self.defaults["qkv_clip_max"])

                    rms = [max(1e-12, float(last_qkv[k]["rms"])) for k in ("q","k","v")]
                    trs = [max(1e-12, float(last_qkv[k]["tr"])) for k in ("q","k","v")]

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
                    # step-clip shrink by positive acceleration
                    step_clip_eff = base_clip / (1.0 + k_shrink * max(0.0, accel))
                    step_clip_eff = max(cmin, min(cmax, step_clip_eff))

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
                                               "qkv_step_clip_eff": step_clip_eff, "qkv_accel": accel})

            # end group loop

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
            self._maybe_auto_blend()

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
