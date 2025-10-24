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

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Optional, Callable

import torch
import torch.nn as nn

__all__ = [
    "ParamGroupSummary",
    "build_tagged_param_groups",
    "collect_param_group_stats",
    "summarize_param_groups",
]

_NORM_TYPES = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm,
               nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)

def _detect_qkv_axis(p: torch.Tensor) -> Optional[int]:
    if p.ndim>=1 and (p.shape[0]%3==0): return 0
    if p.ndim>=2 and (p.shape[1]%3==0): return 1
    return None

def _tag(mn: str, mod: nn.Module, pn: str) -> str:
    n = (mn + "." + pn).lower().strip(".")
    if any(k in n for k in ["lora_a","lora_down","lora_in","lora_l","lora_left"]): return "lora_a"
    if any(k in n for k in ["lora_b","lora_up","lora_out","lora_r","lora_right"]): return "lora_b"
    if any(k in n for k in ["attn_gate","gate_proj","g_proj","gating"]): return "attn_gate"
    if any(k in n for k in ["ffn_up","up_proj","fc1","w1","mlp_up","expand"]): return "ffn_up"
    if any(k in n for k in ["ffn_down","down_proj","fc2","w2","mlp_down","reduce"]): return "ffn_down"
    if any(k in n for k in ["rope","rotary","ntk","pos_rot"]): return "rope"
    if any(k in n for k in ["scale","scaling","gain"]) and not n.endswith("scale_factor"): return "scaling"
    if pn.endswith("bias"):
        if "in_proj_bias" in n or "qkv" in n: return "attn_qkv"
        return "bias"
    if isinstance(mod, nn.Embedding) or any(k in n for k in ["embed","token"]): return "embed"
    if isinstance(mod, _NORM_TYPES): return "norm"
    if isinstance(mod, _CONV_TYPES): return "conv"
    if "in_proj_weight" in n or "qkv." in n or ".qkv" in n or "to_qkv" in n: return "attn_qkv"
    if any(k in n for k in ["q_proj",".to_q",".query",".q."]): return "attn_q"
    if any(k in n for k in ["k_proj",".to_k",".key",".k."]): return "attn_k"
    if any(k in n for k in ["v_proj",".to_v",".value",".v."]): return "attn_v"
    if any(k in n for k in ["o_proj","out_proj",".to_out",".o."]): return "attn_o"
    if isinstance(mod, nn.Linear): return "mlp"
    return "other"

def build_tagged_param_groups(model: nn.Module, *, base_lr: float=3e-4, base_wd: float=0.01, enable_qkv_slicing: bool=True,
                              tag_overrides: Optional[Dict[str, Dict]] = None,
                              lora_overrides: Optional[Dict] = None) -> List[dict]:
    lr_scales = {"embed":0.5,"norm":0.5,"bias":0.5,"scaling":0.5,"rope":0.5,
                 "attn_q":0.9,"attn_k":0.8,"attn_v":1.1,"attn_o":1.0,"attn_qkv":1.0,
                 "ffn_up":1.1,"ffn_down":0.9,"attn_gate":0.9,"mlp":1.0,"conv":1.0,"other":1.0,
                 "lora_a":0.8,"lora_b":0.8}
    zero_wd_tags = {"bias","norm","embed","rope","scaling"}
    tag_overrides = tag_overrides or {}
    lo = lora_overrides or {}

    param_to_mod = {}
    for mn, mod in model.named_modules():
        for pn, p in mod.named_parameters(recurse=False):
            param_to_mod[id(p)] = (mn, mod, pn)
    groups = {}
    names = {}
    for full, p in model.named_parameters():
        if not p.requires_grad: continue
        mn, mod, pn = param_to_mod.get(id(p), ("", None, ""))
        tg = _tag(mn, mod, pn) if mod is not None else "other"
        groups.setdefault(tg, []).append(p); names[id(p)] = full

    qkv_rules = {}
    for p in groups.get("attn_qkv", []):
        if enable_qkv_slicing:
            ax = _detect_qkv_axis(p)
            if ax is not None: qkv_rules[id(p)] = (ax, 3)

    param_groups = []
    for tg, ps in groups.items():
        g = dict(params=ps, lr=base_lr, weight_decay=(0.0 if tg in zero_wd_tags else base_wd),
                 lr_scale=float(lr_scales.get(tg, 1.0)), block_tag=tg,
                 rms_clip_threshold=0.0, rms_clip_granularity="param",
                 agc_clip=0.0,
                 block_trust_chunks=0,
                 use_foreach=True, use_foreach_update=True, foreach_min_bucket=4,
                 bucket_standardize=False, bucket_standardize_source="global",
                 bucket_scalarless=True, use_triton_bucket_stats=False, use_triton_fused_apply=False,
                 )
        if tg=="attn_qkv" and qkv_rules:
            g["qkv_rules"] = qkv_rules; g["qkv_trust_split"] = True
            g["qkv_lr_scales"] = {"q": float(lr_scales.get("attn_q", 1.0)),
                                  "k": float(lr_scales.get("attn_k", 1.0)),
                                  "v": float(lr_scales.get("attn_v", 1.0))}
        if tg in {"ffn_up", "ffn_down"}:
            g["auto_ffn_asym"] = True
        if tg in {"lora_a","lora_b"}:
            g["sign_blend"] = 0.15 if "sign_blend" not in lo else float(lo["sign_blend"])
            g["agc_clip"] = 0.02 if "agc_clip" not in lo else float(lo["agc_clip"])
            g["weight_decay"] = 0.0 if "weight_decay" not in lo else float(lo["weight_decay"])
            if "lr_scale" in lo: g["lr_scale"] = float(lo["lr_scale"])
            if "trust_clip" in lo: g["trust_clip"] = float(lo["trust_clip"])
        if tg in tag_overrides:
            for k,v in tag_overrides[tg].items(): g[k] = v
        param_groups.append(g)
    return param_groups


@dataclass(frozen=True)
class ParamGroupSummary:
    """Lightweight container describing the composition of a parameter group."""

    idx: int
    tag: str
    lr: float
    weight_decay: float
    lr_scale: float
    n_tensors: int
    n_params: int
    param_ratio: float


def collect_param_group_stats(param_groups: Iterable[dict]) -> List[ParamGroupSummary]:
    """Return structured statistics for each parameter group.

    Args:
        param_groups: Iterable of parameter group dictionaries.

    Returns:
        A list of :class:`ParamGroupSummary` entries mirroring the original order.
    """

    pg_list = list(param_groups)
    staged: List[Tuple[int, dict, int, int]] = []
    total_params = 0

    for idx, group in enumerate(pg_list):
        params = [p for p in group.get("params", []) if isinstance(p, torch.Tensor)]
        n_tensors = len(params)
        n_params = int(sum(int(p.numel()) for p in params))
        staged.append((idx, group, n_tensors, n_params))
        total_params += n_params

    if not staged:
        return []

    summaries: List[ParamGroupSummary] = []
    denom = float(total_params) if total_params else 0.0

    for idx, group, n_tensors, n_params in staged:
        ratio = (n_params / denom) if denom else 0.0
        summaries.append(
            ParamGroupSummary(
                idx=idx,
                tag=str(group.get("block_tag", "default")),
                lr=float(group.get("lr", 0.0)),
                weight_decay=float(group.get("weight_decay", 0.0)),
                lr_scale=float(group.get("lr_scale", 1.0)),
                n_tensors=n_tensors,
                n_params=n_params,
                param_ratio=ratio,
            )
        )

    return summaries


def summarize_param_groups(param_groups: Iterable[dict], *, precision: int = 4,
                           sort_by: str = "tag",
                           include_share: bool = False,
                           descending: Optional[bool] = None) -> str:
    """Return a human-friendly table describing tagged parameter groups.

    The function operates purely on the optimizer param group dictionaries and
    therefore works for both the output of :func:`build_tagged_param_groups`
    and live ``optimizer.param_groups`` instances.

    Args:
        param_groups: Iterable of parameter group dictionaries.
        precision: Number of decimal places for floating point columns.
        sort_by: Sort order – ``"tag"`` (default), ``"index"``, ``"n_params"`` and
            additional numeric columns such as ``"tensors"``, ``"lr"``,
            ``"weight_decay"``/``"wd"``, ``"lr_scale"`` and ``"share"``.
        include_share: When ``True`` append a column with the percentage of
            parameters captured by each group.
        descending: When ``True`` force a descending sort, ``False`` forces
            ascending order. When ``None`` the function applies a sensible
            default (descending for numeric metrics such as ``"n_params"``,
            ``"lr"``, etc.).

    Returns:
        A multi-line string with a compact summary table.
    """
    rows = collect_param_group_stats(param_groups)
    if not rows:
        return "(no parameter groups)"

    total_params = sum(row.n_params for row in rows)

    key = str(sort_by).lower()
    aliases = {
        "idx": "index",
        "params": "n_params",
        "wd": "weight_decay",
    }
    key = aliases.get(key, key)

    sort_options: Dict[str, Tuple[Callable[[ParamGroupSummary], Any], bool]] = {
        "tag": (lambda r: r.tag, False),
        "index": (lambda r: r.idx, False),
        "n_params": (lambda r: r.n_params, True),
        "tensors": (lambda r: r.n_tensors, True),
        "lr": (lambda r: r.lr, True),
        "weight_decay": (lambda r: r.weight_decay, True),
        "lr_scale": (lambda r: r.lr_scale, True),
        "share": (lambda r: r.param_ratio, True),
    }

    sort_key, default_desc = sort_options.get(key, sort_options["index"])

    if descending is None:
        descending = default_desc
    descending = bool(descending)

    rows.sort(key=lambda r: r.idx)
    rows.sort(key=sort_key, reverse=descending)

    def fmt_float(val: float) -> str:
        if abs(val) >= 1e4 or (0 < abs(val) < 1e-3):
            return f"{val:.{precision}e}"
        return f"{val:.{precision}f}"

    def fmt_percent(value: float) -> str:
        return f"{value * 100:.{precision}f}%"

    headers = ["idx", "tag", "tensors", "params", "lr", "wd", "lr_scale"]
    if include_share:
        headers.append("share")

    table = []
    for row in rows:
        cols = [
            str(row.idx),
            row.tag,
            str(row.n_tensors),
            f"{row.n_params:,}",
            fmt_float(row.lr),
            fmt_float(row.weight_decay),
            fmt_float(row.lr_scale),
        ]
        if include_share:
            cols.append(fmt_percent(row.param_ratio))
        table.append(cols)

    widths = [max(len(h), *(len(row[i]) for row in table)) for i, h in enumerate(headers)]

    def fmt_row(cols):
        return " ".join(col.ljust(widths[i]) for i, col in enumerate(cols))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines.extend(fmt_row(row) for row in table)
    lines.append("Total params: {:,}".format(total_params))
    return "\n".join(lines)
