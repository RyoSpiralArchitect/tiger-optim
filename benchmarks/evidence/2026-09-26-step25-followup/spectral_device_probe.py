#!/usr/bin/env python3
"""Bounded MPS/CPU spectral-helper timing probe; no optimizer changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from tiger_optim.tiger import _spectral_dispersion_chunks


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(chunks: tuple[torch.Tensor, ...], device: str) -> tuple[list[float], torch.Tensor]:
    before = time.perf_counter()
    if device == "cpu":
        local = tuple(chunk.to("cpu") for chunk in chunks)
        values = _spectral_dispersion_chunks(local, 0.2, 0.25)
        result = torch.stack([torch.stack(row) for row in values]).to("mps")
    else:
        values = _spectral_dispersion_chunks(chunks, 0.2, 0.25)
        result = torch.stack([torch.stack(row) for row in values])
    torch.mps.synchronize()
    return [1000.0 * (time.perf_counter() - before)], result.to("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", choices=("cpu", "mps"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    torch.manual_seed(0)
    tensors = {
        "weight": torch.randn(768, 256, device="mps"),
        "bias": torch.randn(768, device="mps"),
    }
    source = ROOT / "src/tiger_optim/tiger.py"
    before = sha256(source)
    order = (args.first, "mps" if args.first == "cpu" else "cpu")
    out: dict = {"first": args.first, "source_sha256_before": before, "cases": {}}
    for name, tensor in tensors.items():
        chunks = tuple(torch.chunk(tensor, 3, dim=0))
        data = {}
        for device in order:
            times = []
            reference = None
            for _ in range(6):
                sample, result = measure(chunks, device)
                times.extend(sample)
                reference = result
            data[device] = {"first_ms": times[0], "warm_median_ms": statistics.median(times[1:]),
                            "times_ms": times, "values": reference.tolist()}
        out["cases"][name] = data
    out["source_sha256_after"] = sha256(source)
    out["source_stable"] = before == out["source_sha256_after"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"source_stable": out["source_stable"], "cases": {
        name: {d: {"first_ms": row["first_ms"], "warm_median_ms": row["warm_median_ms"]}
               for d, row in result.items()}
        for name, result in out["cases"].items()}}))


if __name__ == "__main__":
    main()
