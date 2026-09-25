#!/usr/bin/env python3
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

# Minimal local import (assumes 'pip install -e .' or PYTHONPATH set to src/)
from tiger_optim import Tiger, build_tagged_param_groups

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class TinyMix(torch.nn.Module):
    def __init__(self, d=256, ff=512, heads=4):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(d, heads, batch_first=True)
        self.up = torch.nn.Linear(d, ff)
        self.down = torch.nn.Linear(ff, d)
        self.ln = torch.nn.LayerNorm(d)
    def forward(self, x):
        h,_ = self.mha(x,x,x)
        h = torch.nn.functional.gelu(self.up(h))
        h = self.down(h)
        return self.ln(h)

def sync(dev):
    if dev.type=="cuda" and torch.cuda.is_available(): torch.cuda.synchronize()
    elif dev.type=="mps" and hasattr(torch,"mps") and torch.backends.mps.is_available():
        try: torch.mps.synchronize()
        except: pass


def _git_identity():
    def git(*args):
        try:
            result = subprocess.run(
                ["git", "-C", str(PROJECT_ROOT), *args],
                capture_output=True, text=True, timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=normal")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def _device_name(dev):
    if dev.type == "cuda":
        return torch.cuda.get_device_name(dev.index if dev.index is not None else torch.cuda.current_device())
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return platform.processor() or platform.machine() or dev.type


def _json_config(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_config(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_config(item) for key, item in value.items()}
    return str(value)


def _optimizer_config(opt, model):
    names = {id(param): name for name, param in model.named_parameters()}
    groups = []
    for group in opt.param_groups:
        options = {}
        for key, value in group.items():
            if key == "params":
                continue
            if key == "qkv_rules":
                options[key] = {names.get(param_id, str(param_id)): _json_config(rule)
                                for param_id, rule in value.items()}
            else:
                options[key] = _json_config(value)
        groups.append({
            "parameters": [
                {"name": names.get(id(param), "<unnamed>"),
                 "shape": list(param.shape), "dtype": str(param.dtype)}
                for param in group["params"]
            ],
            "options": options,
        })
    return {"class": type(opt).__module__ + "." + type(opt).__qualname__,
            "defaults": _json_config(opt.defaults), "param_groups": groups}


def _iteration(model, opt, loss_fn, x, y, dev):
    # Two timed compute sections keep loss.item() and Tiger's report_metrics()
    # outside both timers. AdamW performs the same scalar extraction.
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()
    sync(dev)
    t1 = time.perf_counter()

    loss_value = loss.detach().item()
    if hasattr(opt, "report_metrics"):
        opt.report_metrics(loss_value)

    t2 = time.perf_counter()
    opt.step()
    sync(dev)
    t3 = time.perf_counter()
    forward_backward_ms = 1000.0 * (t1 - t0)
    optimizer_ms = 1000.0 * (t3 - t2)
    return loss_value, forward_backward_ms, optimizer_ms

def build_optimizer(model, mode: str, device: torch.device):
    mode = mode.lower()
    if mode == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.999), weight_decay=0.01)
    # Tiger modes
    groups = build_tagged_param_groups(model, base_lr=3e-4, base_wd=0.01, enable_qkv_slicing=True)
    common = dict(factored=True, precond_alpha=1.0, trust_space="precond", trust_ema_beta=0.9,
                  use_foreach_update=True, bucket_standardize=True, bucket_scalarless=True,
                  profiler_enabled=False)
    if mode == "tiger_v21_full":
        return Tiger(groups, **common,
            lora_density_adapt=True, lora_pid_mode="auto",
            qkv_lr_autoadapt=True, qkv_w_rms=0.7, qkv_w_trust=0.3,
            qkv_lr_gain=0.02, qkv_gain_shrink_gamma=1.2, qkv_disp_ema_beta=0.8, qkv_gamma_rate=0.6)
    if mode == "tiger_v21_no_qkv":
        for g in groups:
            if g.get("block_tag")=="attn_qkv":
                g["qkv_lr_scales"] = None; g["qkv_trust_split"] = False
        return Tiger(groups, **common, qkv_lr_autoadapt=False)
    if mode == "tiger_v21_no_ffn":
        for g in groups:
            if g.get("block_tag") in ("ffn_up","ffn_down"):
                g["auto_ffn_asym"] = False
        return Tiger(groups, **common, qkv_lr_autoadapt=True)
    if mode == "tiger_v13_compat":
        # Approximate older Tiger by disabling new features and inertia
        for g in groups:
            g["auto_ffn_asym"] = False
            if g.get("block_tag")=="attn_qkv":
                g["qkv_lr_scales"] = None; g["qkv_trust_split"] = False
        compat = dict(common)
        compat["trust_ema_beta"] = 0.0
        return Tiger(groups, **compat, lora_density_adapt=False, qkv_lr_autoadapt=False)
    raise ValueError(f"Unknown mode: {mode}")

def run(mode: str, device_str: str, steps=100, warmup=20, seed=0, batch=16, T=32, D=256,
        cli_config=None, comparison_id=None):
    if steps <= 0 or warmup < 0:
        raise ValueError("steps must be positive and warmup must be nonnegative")
    dev = torch.device(device_str if torch.device(device_str).type in {"cpu","cuda","mps"} else "cpu")
    torch.manual_seed(seed)
    model = TinyMix(d=D, ff=2*D, heads=4).to(dev)
    x = torch.randn(batch, T, D, device=dev)
    y = torch.randn(batch, T, D, device=dev)
    loss_fn = torch.nn.MSELoss()

    opt = build_optimizer(model, mode, dev)
    initial_optimizer_config = _optimizer_config(opt, model)
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = uuid.uuid4().hex
    comparison_id = comparison_id or run_id
    run_config = {
        "run_id": run_id,
        "comparison_id": comparison_id,
        "timestamp_utc": timestamp,
        "git": _git_identity(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "pytorch_cuda_build": torch.version.cuda,
        "os": platform.platform(),
        "device": {"requested": device_str, "resolved": str(dev), "name": _device_name(dev)},
        "runtime": {
            "torch_default_device": (str(torch.get_default_device())
                                     if hasattr(torch, "get_default_device") else None),
            "torch_default_dtype": str(torch.get_default_dtype()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "TIGER_ACCEL_DISABLE": os.environ.get("TIGER_ACCEL_DISABLE"),
            "TIGER_ACCEL_PREFER": os.environ.get("TIGER_ACCEL_PREFER"),
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        },
        "cli": _json_config(cli_config) if cli_config is not None else None,
        "effective_arguments": {
            "mode": mode, "device": str(dev), "steps": steps, "warmup": warmup,
            "seed": seed, "batch": batch, "sequence_length": T,
            "model_width": D, "ff_width": 2 * D, "attention_heads": 4,
        },
        "optimizer_config_initial": initial_optimizer_config,
        "timing_scope": {
            "ms": "zero_grad + forward + loss + backward + opt.step + synchronization",
            "ms_forward_backward": "zero_grad + forward + loss + backward + synchronization",
            "ms_optimizer": "opt.step + synchronization",
            "excluded": "loss.item + report_metrics between timed sections",
        },
    }

    # warmup
    for _ in range(warmup):
        _iteration(model, opt, loss_fn, x, y, dev)

    ms, forward_backward_ms, optimizer_ms, losses = [], [], [], []
    for _ in range(steps):
        loss_value, fb_ms, opt_ms = _iteration(model, opt, loss_fn, x, y, dev)
        forward_backward_ms.append(fb_ms)
        optimizer_ms.append(opt_ms)
        ms.append(fb_ms + opt_ms)
        losses.append(loss_value)

    summary = dict(mode=mode, device=dev.type, steps=steps, warmup=warmup,
                   ms_median=statistics.median(ms), ms_mean=sum(ms)/len(ms),
                   ms_forward_backward_median=statistics.median(forward_backward_ms),
                   ms_forward_backward_mean=sum(forward_backward_ms)/len(forward_backward_ms),
                   ms_optimizer_median=statistics.median(optimizer_ms),
                   ms_optimizer_mean=sum(optimizer_ms)/len(optimizer_ms),
                   loss_last=losses[-1], loss_min=min(losses), loss_mean=sum(losses)/len(losses))

    run_config["optimizer_config_final"] = _optimizer_config(opt, model)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("benchmarks", "results"); os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"compare-{mode}-{dev.type}-{run_id[:8]}-{ts}.json")
    with open(out_path, "w") as f:
        json.dump(dict(summary=summary, run=run_config,
                       series=dict(ms=ms, ms_forward_backward=forward_backward_ms,
                                   ms_optimizer=optimizer_ms, loss=losses)), f, indent=2)
    print(json.dumps(summary, indent=2))
    print("Saved:", out_path)
    return out_path

def main():
    ap = argparse.ArgumentParser("AdamW vs Tiger compare")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", nargs="+", default=["adamw","tiger_v13_compat","tiger_v21_no_qkv","tiger_v21_no_ffn","tiger_v21_full"])
    args = ap.parse_args()

    comparison_id = uuid.uuid4().hex
    cli_config = {"argv": sys.argv[1:], "parsed": vars(args)}
    for m in args.modes:
        run(m, args.device, steps=args.steps, warmup=args.warmup, seed=args.seed,
            cli_config=cli_config, comparison_id=comparison_id)

if __name__ == "__main__":
    main()
