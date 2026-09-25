#!/usr/bin/env python3
"""Compare cold and warm MPS row reductions on QKV spectral shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def reduce_rows(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "mean":
        return x.mean(dim=-1)
    return x.sum(dim=-1) / x.shape[-1]


def measure(rows: list[torch.Tensor], kind: str) -> tuple[list[float], list[list[float]]]:
    times = []
    values = []
    for _ in range(6):
        torch.mps.synchronize()
        start = time.perf_counter()
        results = [reduce_rows(x, kind) for x in rows]
        torch.mps.synchronize()
        times.append(1000.0 * (time.perf_counter() - start))
        values = [result.to("cpu").tolist() for result in results]
    return times, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", choices=("mean", "sum"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    torch.manual_seed(0)
    lengths = (65536, 6554, 8192, 32768, 256, 26, 32, 128)
    rows = [torch.randn(3, n, device="mps") for n in lengths]
    order = (args.first, "sum" if args.first == "mean" else "mean")
    result = {"first": args.first, "lengths": lengths, "runs": {}}
    for kind in order:
        times, values = measure(rows, kind)
        result["runs"][kind] = {"first_ms": times[0], "warm_median_ms": statistics.median(times[1:]),
                                "times_ms": times, "values": values}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({kind: {"first_ms": data["first_ms"], "warm_median_ms": data["warm_median_ms"]}
                      for kind, data in result["runs"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
