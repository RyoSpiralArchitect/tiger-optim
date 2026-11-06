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
# Triton kernels: (1) bucket_stats_triton (ussq, pssq, n), (2) fused_apply_updates (experimental)

try:
    import triton
    import triton.language as tl
    import torch
except Exception:
    triton = None

def _concat_views(tensors):
    flats = [t.contiguous().view(-1) for t in tensors]
    if len(flats)==0:
        return None
    return torch.cat(flats, dim=0)

def bucket_stats_triton(updates, params):
    if triton is None:
        raise RuntimeError("Triton not available")
    u = _concat_views(updates)
    p = _concat_views(params)
    if u is None:
        reference = updates[0] if updates else (params[0] if params else None)
        device = reference.device if reference is not None else torch.device("cpu")
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return zero, zero.clone(), zero.clone()

    N = u.numel()
    if p is None or p.numel()==0:
        p = torch.zeros(1, device=u.device, dtype=u.dtype)

    @triton.jit
    def ssq_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc = tl.sum(x * x, axis=0)
        tl.store(out_ptr + pid, acc)

    def _ssq(x):
        BLOCK = 1024
        grid = ( (x.numel() + BLOCK - 1)//BLOCK, )
        out = torch.empty(grid[0], device=x.device, dtype=x.dtype)
        ssq_kernel[grid](x, out, x.numel(), BLOCK=BLOCK)
        return out.sum()

    ussq = _ssq(u)
    pssq = _ssq(p)
    n = torch.tensor(float(N), device=u.device, dtype=torch.float32)
    return ussq.to(torch.float32), pssq.to(torch.float32), n

# Experimental: fused apply updates + per-bucket sums in one pass
def fused_apply_updates(params, updates):
    if not params or not updates:
        return False, "empty"
    if len(params) != len(updates):
        raise ValueError("params and updates must have identical lengths")

    target_device = params[0].device
    target_dtype = params[0].dtype

    if target_device.type != "cuda":
        return False, "non_cuda_bucket"

    apply_inputs = []
    total = 0

    for param, update in zip(params, updates):
        if param.device != target_device:
            return False, "param_device_mismatch"
        if param.dtype != target_dtype:
            return False, "param_dtype_mismatch"
        if update.shape != param.shape:
            return False, "shape_mismatch"
        if update.device != target_device:
            return False, "update_device_mismatch"
        if update.dtype != target_dtype:
            return False, "update_dtype_mismatch"

        n = param.numel()
        if n:
            try:
                flat_param = param.view(-1)
            except RuntimeError:
                return False, "param_non_contiguous"
            try:
                flat_update = update.view(-1)
            except RuntimeError:
                return False, "update_non_contiguous"
            apply_inputs.append((flat_param, flat_update, n))
            total += n

    if total == 0:
        return False, "empty"

    if triton is None:
        raise RuntimeError("Triton not available")

    @triton.jit
    def apply_sum_kernel(p_ptr, u_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        p = tl.load(p_ptr + offs, mask=mask, other=0.0)
        u = tl.load(u_ptr + offs, mask=mask, other=0.0)
        p = p + u
        tl.store(p_ptr + offs, p, mask=mask)

    BLOCK = 1024
    for flat_param, flat_update, n in apply_inputs:
        grid = ((n + BLOCK - 1) // BLOCK,)
        apply_sum_kernel[grid](flat_param, flat_update, n, BLOCK=BLOCK)

    # NOTE: This 'fused' path focuses on in-place apply; bucket-wide stats are recommended
    # via bucket_stats_triton to avoid a second global memory pass.
    return True, "ok"


# ============================================================================
# Flat fused candidate/apply path.
#
# Fuses the per-parameter moment/direction math for "simple" parameters
# (fp32 param/grad/state, dense v, softsign blending, positive alpha) onto
# concatenated flat buffers. Sequence, all async:
#   K1 _flat_cand_kernel : cand_m, cand_v, Pbuf, dsign
#   K2 _flat_seg_stats_kernel : per-param amax_m/amax_v/amax_p/amax_dsign/ssq_m
#   K3 _flat_dir_kernel : d = (1-sb)*dsign + sb*(m/rms)*P
#   K4 _flat_seg_stats_kernel : amax_d
#   K5 _flat_apply_kernel : p = p*(1-decay) - lr*eff*d ; M,V <- cand (ok only)
# The host performs one guard sync per group between K4 and K5.
# ============================================================================

if triton is not None:

    @triton.jit
    def _flat_cand_kernel(g_ptr, m_ptr, v_ptr, cm_ptr, cv_ptr, pp_ptr, ds_ptr,
                          N, b1, b2, eps, alpha, sign_tau, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        m = tl.load(m_ptr + offs, mask=mask, other=0.0)
        v = tl.load(v_ptr + offs, mask=mask, other=0.0)
        cm = m * b1 + g * (1.0 - b1)
        cv = v * b2 + g * g * (1.0 - b2)
        pp = tl.sqrt(cv) + eps
        pp = tl.exp(-alpha * tl.log(tl.maximum(pp, 1e-30)))
        pp = tl.minimum(tl.maximum(pp, 1e-8), 1e8)
        ds = cm / (tl.abs(cm) + sign_tau) * pp
        tl.store(cm_ptr + offs, cm, mask=mask)
        tl.store(cv_ptr + offs, cv, mask=mask)
        tl.store(pp_ptr + offs, pp, mask=mask)
        tl.store(ds_ptr + offs, ds, mask=mask)

    @triton.jit
    def _flat_seg_stats_kernel(cm_ptr, cv_ptr, p_ptr, x_ptr, start, end,
                               out_ptr, BLOCK: tl.constexpr):
        # Program j reduces param j's segment [start_j, end_j) passed via
        # two pointer arrays indexed by program id.
        j = tl.program_id(0)
        s = tl.load(start + j)
        e = tl.load(end + j)
        n = e - s
        am_m = 0.0
        am_v = 0.0
        am_p = 0.0
        am_x = 0.0
        for off in range(0, n, BLOCK):
            idx = s + off + tl.arange(0, BLOCK)
            mask = idx < e
            cm = tl.load(cm_ptr + idx, mask=mask, other=0.0)
            cv = tl.load(cv_ptr + idx, mask=mask, other=0.0)
            pp = tl.load(p_ptr + idx, mask=mask, other=0.0)
            xx = tl.load(x_ptr + idx, mask=mask, other=0.0)
            am_m = tl.maximum(am_m, tl.max(tl.abs(cm), axis=0))
            am_v = tl.maximum(am_v, tl.max(tl.abs(cv), axis=0))
            am_p = tl.maximum(am_p, tl.max(tl.abs(pp), axis=0))
            am_x = tl.maximum(am_x, tl.max(tl.abs(xx), axis=0))
        # Second pass: scaled sum-of-squares so finite moments near the FP32
        # limit do not overflow (norm = am_m * sqrt(ssq_scaled / n)).
        ssq_m = 0.0
        if am_m > 0.0:
            inv = 1.0 / am_m
            for off in range(0, n, BLOCK):
                idx = s + off + tl.arange(0, BLOCK)
                mask = idx < e
                cm = tl.load(cm_ptr + idx, mask=mask, other=0.0)
                cs = cm * inv
                ssq_m += tl.sum(tl.where(mask, cs * cs, 0.0), axis=0)
        tl.store(out_ptr + j * 5 + 0, am_m)
        tl.store(out_ptr + j * 5 + 1, am_v)
        tl.store(out_ptr + j * 5 + 2, am_p)
        tl.store(out_ptr + j * 5 + 3, am_x)
        tl.store(out_ptr + j * 5 + 4, ssq_m)

    @triton.jit
    def _flat_dir_kernel(cm_ptr, pp_ptr, ds_ptr, rms_ptr, seg_ptr, d_ptr,
                         N, sblend, eps, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        seg = tl.load(seg_ptr + offs, mask=mask, other=0)
        rms = tl.load(rms_ptr + seg, mask=mask, other=1.0)
        cm = tl.load(cm_ptr + offs, mask=mask, other=0.0)
        pp = tl.load(pp_ptr + offs, mask=mask, other=0.0)
        ds = tl.load(ds_ptr + offs, mask=mask, other=0.0)
        dmag = cm / (rms + eps) * pp
        d = ds * (1.0 - sblend) + dmag * sblend
        tl.store(d_ptr + offs, d, mask=mask)

    @triton.jit
    def _flat_apply_kernel(p_ptr, m_ptr, v_ptr, cm_ptr, cv_ptr, d_ptr,
                           eff_ptr, seg_ptr, ok_ptr, upd_ptr,
                           N, decay, lr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        seg = tl.load(seg_ptr + offs, mask=mask, other=0)
        ok = tl.load(ok_ptr + seg, mask=mask, other=0.0)
        accepted = mask & (ok > 0.5)
        p = tl.load(p_ptr + offs, mask=mask, other=0.0)
        d = tl.load(d_ptr + offs, mask=mask, other=0.0)
        eff = tl.load(eff_ptr + seg, mask=mask, other=0.0)
        upd = -lr * eff * d
        p = tl.where(accepted, p * (1.0 - decay) + upd, p)
        tl.store(p_ptr + offs, p, mask=mask)
        tl.store(upd_ptr + offs, tl.where(accepted, upd, 0.0), mask=mask)
        cm = tl.load(cm_ptr + offs, mask=mask, other=0.0)
        cv = tl.load(cv_ptr + offs, mask=mask, other=0.0)
        tl.store(m_ptr + offs, cm, mask=accepted)
        tl.store(v_ptr + offs, cv, mask=accepted)


def flat_fused_supported():
    return triton is not None


def _require_triton():
    if triton is None:
        raise RuntimeError("Triton not available")


def _check_flat(*tensors):
    for t in tensors:
        if not t.is_contiguous():
            raise ValueError("flat fused path requires contiguous buffers")
        if t.dtype != torch.float32:
            raise ValueError("flat fused path requires float32 buffers")


def flat_step_candidates(g, m, v, cand_m, cand_v, pp, dsign, *,
                         b1, b2, eps, alpha, sign_tau):
    """K1 launch: fused elementwise candidate moments + preconditioner.

    Requires contiguous fp32 buffers of identical length on one CUDA device.
    """
    _require_triton()
    _check_flat(g, m, v, cand_m, cand_v, pp, dsign)
    if g.device.type != "cuda":
        raise RuntimeError("flat fused path requires CUDA")
    n = g.numel()
    _flat_cand_kernel[((n + 1023) // 1024,)](
        g, m, v, cand_m, cand_v, pp, dsign, n, b1, b2, eps, alpha, sign_tau,
        BLOCK=1024)


def flat_seg_stats(cand_m, cand_v, p_flat, x, starts, ends, out):
    """K2/K4 launch: per-segment [amax_m, amax_v, amax_p, amax_x, ssq_scaled].

    out[j*5+4] holds the *amax-scaled* sum of squares of cand_m, so the
    stable per-segment norm is amax_m * sqrt(ssq_scaled / seg_len) — no
    overflow for finite moments near the FP32 limit.
    """
    _require_triton()
    _check_flat(cand_m, cand_v, p_flat, x, out)
    if starts.numel() != ends.numel():
        raise ValueError("starts/ends length mismatch")
    if out.numel() < 5 * starts.numel():
        raise ValueError("out buffer too small")
    if starts.dtype not in (torch.int32, torch.int64):
        raise ValueError("starts/ends must be integer tensors")
    _flat_seg_stats_kernel[(starts.numel(),)](
        cand_m, cand_v, p_flat, x, starts, ends, out, BLOCK=1024)


def flat_direction(cand_m, pp, dsign, rms, seg_idx, d, *, sblend, eps):
    """K3 launch: d = (1-sblend)*dsign + sblend*(cm/(rms+eps))*P.

    `rms` holds the raw per-segment root-mean-square; eps is added inside
    the kernel to match the eager denominator (_rms(m)+eps).
    """
    _require_triton()
    _check_flat(cand_m, pp, dsign, rms, d)
    n = cand_m.numel()
    _flat_dir_kernel[((n + 1023) // 1024,)](
        cand_m, pp, dsign, rms, seg_idx, d, n, sblend, eps, BLOCK=1024)


def flat_apply(p_flat, m, v, cand_m, cand_v, d, eff, seg_idx, ok, upd, *,
               decay, lr):
    """K5 launch: where ok[seg], p = p*(1-decay) - lr*eff*d and m,v <- cand.

    Rejected segments (ok=0) are left completely untouched — no decay, no
    update, no moment commit; upd receives 0 there.
    """
    _require_triton()
    _check_flat(p_flat, m, v, cand_m, cand_v, d, eff, ok, upd)
    n = p_flat.numel()
    _flat_apply_kernel[((n + 1023) // 1024,)](
        p_flat, m, v, cand_m, cand_v, d, eff, seg_idx, ok, upd, n, decay, lr,
        BLOCK=1024)
