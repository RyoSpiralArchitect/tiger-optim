#!/usr/bin/env python3
"""Bounded MPS diagnostic for Tiger's first QKV adaptation step.

Run each variant in a fresh process.  This script records every global step so
the local 24/25/26 comparison is visible without a cross-process speed claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_compare_optim import TinyMix, build_optimizer, sync  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def qkv_scales(opt: torch.optim.Optimizer) -> dict | None:
    for group in opt.param_groups:
        if group.get("block_tag") == "attn_qkv":
            scales = group.get("qkv_lr_scales")
            return dict(scales) if scales is not None else None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=("full", "no_adapt", "no_spectral", "interval_1000", "interval_7"))
    parser.add_argument("--steps", type=int, default=29)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 26:
        parser.error("--steps must include global steps 24, 25, and 26")
    if not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")

    tiger_source = ROOT / "src" / "tiger_optim" / "tiger.py"
    bench_source = ROOT / "benchmarks" / "bench_compare_optim.py"
    source_before = sha256(tiger_source)
    bench_before = sha256(bench_source)
    torch.manual_seed(0)
    dev = torch.device("mps")
    model = TinyMix(d=256, ff=512, heads=4).to(dev)
    x = torch.randn(16, 32, 256, device=dev)
    y = torch.randn(16, 32, 256, device=dev)
    loss_fn = torch.nn.MSELoss()
    opt = build_optimizer(model, "tiger_v21_full", dev)
    if args.variant == "no_adapt":
        opt.defaults["qkv_lr_autoadapt"] = False
    elif args.variant == "no_spectral":
        opt.defaults["qkv_spectral_adapt"] = False
    elif args.variant == "interval_1000":
        opt.defaults["qkv_lr_interval"] = 1000
    elif args.variant == "interval_7":
        opt.defaults["qkv_lr_interval"] = 7

    rows = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(args.steps):
            global_step_before = int(opt._global_step)
            scales_before = qkv_scales(opt)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            sync(dev)
            loss_value = float(loss.detach().item())
            opt.report_metrics(loss_value)
            t0 = time.perf_counter()
            opt.step()
            sync(dev)
            elapsed_ms = 1000.0 * (time.perf_counter() - t0)
            rows.append({
                "global_step": int(opt._global_step),
                "global_step_before": global_step_before,
                "optimizer_ms": elapsed_ms,
                "loss": loss_value,
                "qkv_scales_before": scales_before,
                "qkv_scales_after": qkv_scales(opt),
            })
        warning_rows = [{
            "category": item.category.__name__, "message": str(item.message),
            "filename": item.filename, "lineno": item.lineno,
        } for item in caught]

    source_after = sha256(tiger_source)
    bench_after = sha256(bench_source)
    payload = {
        "schema": 1,
        "variant": args.variant,
        "steps": args.steps,
        "git_head": git_head(),
        "machine": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "device": str(dev),
        "default_device": str(torch.get_default_device()),
        "pid": os.getpid(),
        "tiger_source_sha256_before": source_before,
        "tiger_source_sha256_after": source_after,
        "bench_source_sha256_before": bench_before,
        "bench_source_sha256_after": bench_after,
        "source_stable_during_run": source_before == source_after and bench_before == bench_after,
        "optimizer_defaults": {
            "qkv_lr_autoadapt": opt.defaults["qkv_lr_autoadapt"],
            "qkv_spectral_adapt": opt.defaults["qkv_spectral_adapt"],
            "qkv_lr_interval": opt.defaults["qkv_lr_interval"],
            "profiler_enabled": opt._prof.enabled,
        },
        "rows": rows,
        "warnings": warning_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "variant": args.variant,
        "stable": payload["source_stable_during_run"],
        "step24_ms": rows[23]["optimizer_ms"],
        "step25_ms": rows[24]["optimizer_ms"],
        "step26_ms": rows[25]["optimizer_ms"],
        "warning_count": len(warning_rows),
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
