#!/usr/bin/env python3
"""Capture CPU operator trace around MPS optimizer global steps 24–26."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
import warnings
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_compare_optim import TinyMix, build_optimizer, sync  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_sources() -> dict[str, str]:
    names = ("src/tiger_optim/tiger.py", "src/tiger_optim/accel/__init__.py",
             "src/tiger_optim/accel/torch_backend.py", "benchmarks/bench_compare_optim.py")
    return {name: sha256(ROOT / name) for name in names}


def iteration(model, opt, loss_fn, x, y, dev, *, global_step: int, profile: bool) -> dict:
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()
    sync(dev)
    opt.report_metrics(float(loss.detach().item()))
    t0 = time.perf_counter()
    if profile:
        with torch.autograd.profiler.record_function(f"diagnose/optimizer_step_{global_step}"):
            opt.step()
            sync(dev)
    else:
        opt.step()
        sync(dev)
    return {"global_step": int(opt._global_step),
            "optimizer_ms": 1000.0 * (time.perf_counter() - t0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("full", "no_spectral"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_before = sha_sources()
    torch.manual_seed(0)
    dev = torch.device("mps")
    model = TinyMix(d=256, ff=512, heads=4).to(dev)
    x = torch.randn(16, 32, 256, device=dev)
    y = torch.randn(16, 32, 256, device=dev)
    loss_fn = torch.nn.MSELoss()
    opt = build_optimizer(model, "tiger_v21_full", dev)
    if args.variant == "no_spectral":
        opt.defaults["qkv_spectral_adapt"] = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for global_step in range(1, 24):
            iteration(model, opt, loss_fn, x, y, dev, global_step=global_step, profile=False)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            rows = [iteration(model, opt, loss_fn, x, y, dev, global_step=step, profile=True)
                    for step in (24, 25, 26)]
        warning_rows = [{"category": w.category.__name__, "message": str(w.message),
                         "filename": w.filename, "lineno": w.lineno} for w in caught]

    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
    (args.output_dir / "table.txt").write_text(table + "\n")
    counts = {item.key: int(item.count) for item in prof.key_averages()}
    with tempfile.NamedTemporaryFile(prefix="tiger-step25-", suffix=".json", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        prof.export_chrome_trace(str(temp_path))
        trace_sha256_uncompressed = sha256(temp_path)
        with temp_path.open("rb") as src, (args.output_dir / "trace.json.gz").open("wb") as dest:
            with gzip.GzipFile(filename="", mode="wb", fileobj=dest, mtime=0, compresslevel=9) as gz:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    gz.write(chunk)
    finally:
        temp_path.unlink(missing_ok=True)
    source_after = sha_sources()
    payload = {
        "schema": 1,
        "variant": args.variant,
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "source_stable_during_run": source_before == source_after,
        "global_steps": rows,
        "warnings": warning_rows,
        "operator_counts": counts,
        "trace_sha256_uncompressed": trace_sha256_uncompressed,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"variant": args.variant,
                      "stable": payload["source_stable_during_run"],
                      "steps": rows,
                      "warnings": len(warning_rows),
                      "fft_rfft": counts.get("aten::fft_rfft", 0),
                      "local_scalar_dense": counts.get("aten::_local_scalar_dense", 0)}, sort_keys=True))


if __name__ == "__main__":
    main()
