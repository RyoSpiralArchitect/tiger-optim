#!/usr/bin/env python3
"""Compare six scalar QKV FFT calls with two batched calls on MPS."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch


def spectral_metrics(spec: torch.Tensor, low_band: float = 0.2,
                     high_band: float = 0.25) -> torch.Tensor:
    """Return per-row low energy, high energy, phase spread as one MPS tensor."""
    power = spec.abs().pow(2)
    side = power[..., 1:]
    n_eff = side.shape[-1]
    low_len = max(1, min(n_eff, math.ceil(low_band * n_eff)))
    high_len = max(1, min(n_eff, math.ceil(high_band * n_eff)))
    low = side[..., :low_len].mean(dim=-1)
    high = side[..., -high_len:].mean(dim=-1)
    phase = torch.angle(spec[..., 1:])
    phase_vector = torch.polar(torch.ones_like(phase), phase)
    spread = 1.0 - torch.abs(phase_vector.mean(dim=-1)).clamp(0.0, 1.0)
    return torch.stack((low, high, spread), dim=-1)


def sequential(chunks: list[torch.Tensor]) -> torch.Tensor:
    outputs = []
    for chunk in chunks:
        y = chunk.reshape(-1).detach().float()
        y = y - y.mean()
        out = torch.empty(y.numel() // 2 + 1, dtype=torch.complex64, device=y.device)
        outputs.append(spectral_metrics(torch.fft.rfft(y, out=out)))
    return torch.stack(outputs)


def batched(chunks: list[torch.Tensor]) -> torch.Tensor:
    rows = torch.stack([chunk.reshape(-1).detach().float() for chunk in chunks])
    rows = rows - rows.mean(dim=-1, keepdim=True)
    out = torch.empty((len(chunks), rows.shape[-1] // 2 + 1),
                      dtype=torch.complex64, device=rows.device)
    return spectral_metrics(torch.fft.rfft(rows, dim=-1, out=out))


def measure(method, chunks: list[torch.Tensor], repeats: int) -> list[float]:
    times = []
    for _ in range(repeats):
        torch.mps.synchronize()
        start = time.perf_counter()
        method(chunks)
        torch.mps.synchronize()
        times.append(1000.0 * (time.perf_counter() - start))
    return times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    torch.manual_seed(0)
    rows = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for shape in ((768, 256), (768,)):
            source = torch.randn(shape, device="mps")
            chunks = list(source.chunk(3, dim=0))
            reference = sequential(chunks)
            candidate = batched(chunks)
            torch.mps.synchronize()
            abs_err = (reference - candidate).abs()
            scale = reference.abs().clamp_min(1e-12)
            max_abs_err = float(abs_err.max().item())
            max_rel_err = float((abs_err / scale).max().item())
            for _ in range(args.warmup):
                sequential(chunks)
                batched(chunks)
            torch.mps.synchronize()
            sequential_ms = measure(sequential, chunks, args.repeats)
            batched_ms = measure(batched, chunks, args.repeats)
            rows.append({
                "input_shape": list(shape),
                "chunk_shape": list(chunks[0].shape),
                "sequential_ms": sequential_ms,
                "batched_ms": batched_ms,
                "sequential_median_ms": statistics.median(sequential_ms),
                "batched_median_ms": statistics.median(batched_ms),
                "max_abs_metric_difference": max_abs_err,
                "max_rel_metric_difference": max_rel_err,
                "reference_metrics": reference.cpu().tolist(),
                "batched_metrics": candidate.cpu().tolist(),
            })
        warning_rows = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
    payload = {"schema": 1, "torch": torch.__version__, "python": sys.version,
               "platform": platform.platform(), "warmup": args.warmup,
               "repeats": args.repeats, "rows": rows, "warnings": warning_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for row in rows:
        print(row["input_shape"], "sequential_ms=", round(row["sequential_median_ms"], 3),
              "batched_ms=", round(row["batched_median_ms"], 3),
              "max_abs_metric_difference=", row["max_abs_metric_difference"])
    print("warnings=", len(warning_rows))


if __name__ == "__main__":
    main()
