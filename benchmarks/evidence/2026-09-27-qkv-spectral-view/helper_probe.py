#!/usr/bin/env python3
"""Compare stacked and contiguous-view QKV spectral inputs on MPS."""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from tiger_optim.tiger import _spectral_dispersion_chunks


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "src/tiger_optim/tiger.py"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(chunks, fused, route):
    start = time.perf_counter()
    result = _spectral_dispersion_chunks(
        chunks, 0.2, 0.25, fused=fused if route == "view" else None
    )
    torch.mps.synchronize()
    return 1000.0 * (time.perf_counter() - start), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS unavailable")

    source_before = sha256(SOURCE)
    torch.manual_seed(0)
    rows = []
    with torch.no_grad():
        for length in (256, 65536, 262144, 1048576, 4194304):
            fused = torch.randn((3, length), dtype=torch.float32, device="mps")
            chunks = torch.chunk(fused, 3, dim=0)
            torch.mps.synchronize()
            stacked_rows = torch.stack([chunk.reshape(-1).detach() for chunk in chunks])
            viewed_rows = fused.detach().view(3, -1)
            torch.testing.assert_close(stacked_rows, viewed_rows, rtol=0.0, atol=0.0)
            assert viewed_rows.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()
            assert stacked_rows.untyped_storage().data_ptr() != fused.untyped_storage().data_ptr()
            del stacked_rows, viewed_rows

            old = measure(chunks, fused, "stack")[1]
            new = measure(chunks, fused, "view")[1]
            for actual_chunk, expected_chunk in zip(new, old):
                for actual, expected in zip(actual_chunk, expected_chunk):
                    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)

            for _ in range(2):
                measure(chunks, fused, "stack")
                measure(chunks, fused, "view")
            timings = {"stack": [], "view": []}
            order = []
            for pair in range(8):
                routes = ("stack", "view") if pair % 2 == 0 else ("view", "stack")
                for route in routes:
                    elapsed, _ = measure(chunks, fused, route)
                    timings[route].append(elapsed)
                    order.append(route)
            rows.append({
                "chunk_length": length,
                "shape": [3, length],
                "source_bytes": fused.numel() * fused.element_size(),
                "view_shares_source_storage": True,
                "stack_allocates_separate_storage": True,
                "metrics_exact_equal": True,
                "timing_order": order,
                "stack_ms": timings["stack"],
                "view_ms": timings["view"],
                "stack_median_ms": statistics.median(timings["stack"]),
                "view_median_ms": statistics.median(timings["view"]),
            })

    result = {
        "schema": 1,
        "device": "mps",
        "platform": platform.platform(),
        "torch": torch.__version__,
        "source_sha256_before": source_before,
        "source_sha256_after": sha256(SOURCE),
        "source_stable_during_run": source_before == sha256(SOURCE),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"source_stable": result["source_stable_during_run"], "rows": [
        {"length": row["chunk_length"], "stack_median_ms": row["stack_median_ms"],
         "view_median_ms": row["view_median_ms"]} for row in rows
    ]}, sort_keys=True))


if __name__ == "__main__":
    main()
