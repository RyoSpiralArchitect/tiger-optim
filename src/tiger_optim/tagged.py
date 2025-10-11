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
import torch, torch.nn as nn
from typing import Dict, List, Tuple, Optional

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
