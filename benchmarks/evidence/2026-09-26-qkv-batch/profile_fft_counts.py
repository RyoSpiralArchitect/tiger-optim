#!/usr/bin/env python3
"""Profile Tiger MPS optimizer steps 24–26 from one selected checkout."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    names = ("src/tiger_optim/tiger.py", "src/tiger_optim/accel/__init__.py",
             "src/tiger_optim/accel/torch_backend.py", "benchmarks/bench_compare_optim.py")
    return {name: sha256(root / name) for name in names}


def trace_counts(trace_path: Path) -> list[dict]:
    events = json.loads(trace_path.read_text())["traceEvents"]
    markers = [event for event in events if event.get("ph") == "X"
               and event.get("name", "").startswith("diagnose/optimizer_step_")]
    rows = []
    for marker in markers:
        start, end = marker["ts"], marker["ts"] + marker["dur"]
        counts = Counter(event.get("name") for event in events
                         if event.get("ph") == "X" and event.get("tid") == marker["tid"]
                         and start <= event.get("ts", -1) < end)
        rows.append({"step": int(marker["name"].rsplit("_", 1)[1]),
                     "fft_rfft": counts["aten::fft_rfft"],
                     "fft_r2c": counts["aten::_fft_r2c"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before = source_hashes(root)
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "benchmarks"))
    from bench_compare_optim import TinyMix, build_optimizer, sync

    if not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    torch.manual_seed(0)
    dev = torch.device("mps")
    model = TinyMix(d=256, ff=512, heads=4).to(dev)
    x = torch.randn(16, 32, 256, device=dev)
    y = torch.randn(16, 32, 256, device=dev)
    opt = build_optimizer(model, "tiger_v21_full", dev)
    loss_fn = torch.nn.MSELoss()

    def iteration(global_step: int, profile: bool) -> None:
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        sync(dev)
        opt.report_metrics(float(loss.detach().item()))
        if profile:
            with torch.autograd.profiler.record_function(f"diagnose/optimizer_step_{global_step}"):
                opt.step()
                sync(dev)
        else:
            opt.step()
            sync(dev)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for step in range(1, 24):
            iteration(step, False)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            for step in (24, 25, 26):
                iteration(step, True)
        warning_rows = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        temp = Path(handle.name)
    try:
        prof.export_chrome_trace(str(temp))
        rows = trace_counts(temp)
        trace_hash = sha256(temp)
        with temp.open("rb") as src, (args.output_dir / "trace.json.gz").open("wb") as dest:
            with gzip.GzipFile(filename="", fileobj=dest, mode="wb", mtime=0, compresslevel=9) as gz:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    gz.write(chunk)
    finally:
        temp.unlink(missing_ok=True)
    after = source_hashes(root)
    summary = {"schema": 1, "source_root": str(root), "source_sha256_before": before,
               "source_sha256_after": after, "source_stable_during_run": before == after,
               "torch": torch.__version__, "python": sys.version,
               "platform": platform.platform(), "trace_sha256_uncompressed": trace_hash,
               "steps": rows, "warnings": warning_rows}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stable": summary["source_stable_during_run"], "steps": rows,
                      "resize_warnings": sum("was resized" in w["message"] for w in warning_rows)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
