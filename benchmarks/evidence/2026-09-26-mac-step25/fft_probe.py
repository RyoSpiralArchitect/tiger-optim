#!/usr/bin/env python3
"""Isolate the MPS rfft output-resize warning from Tiger's optimizer step."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    torch.manual_seed(0)
    rows = []
    for length in (256, 65536):
        cpu_input = torch.randn(length, dtype=torch.float32)
        expected = torch.fft.rfft(cpu_input)
        mps_input = cpu_input.to("mps")
        plain_mps_result = None
        for method in ("rfft", "rfft_out", "fft_slice", "cpu_rfft"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                start = time.perf_counter()
                if method == "rfft":
                    actual = torch.fft.rfft(mps_input)
                elif method == "rfft_out":
                    out = torch.empty(length // 2 + 1, dtype=torch.complex64, device="mps")
                    actual = torch.fft.rfft(mps_input, out=out)
                elif method == "fft_slice":
                    actual = torch.fft.fft(mps_input)[: length // 2 + 1]
                else:
                    actual = torch.fft.rfft(cpu_input)
                if actual.device.type == "mps":
                    torch.mps.synchronize()
                elapsed_ms = 1000.0 * (time.perf_counter() - start)
                actual_cpu = actual.cpu()
                if method == "rfft":
                    plain_mps_result = actual_cpu
                mps_difference = (
                    float((actual_cpu - plain_mps_result).abs().max().item())
                    if method == "rfft_out" else None
                )
                max_abs_error = float((actual_cpu - expected).abs().max().item())
                rows.append({
                    "length": length,
                    "method": method,
                    "output_shape": list(actual.shape),
                    "output_dtype": str(actual.dtype),
                    "ms_cold": elapsed_ms,
                    "max_abs_error_vs_cpu_rfft": max_abs_error,
                    "max_abs_error_vs_plain_mps_rfft": mps_difference,
                    "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
                })
    payload = {"schema": 1, "torch": torch.__version__, "python": sys.version,
               "platform": platform.platform(), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for row in rows:
        print(row["length"], row["method"], "warnings=", len(row["warnings"]),
              "ms_cold=", round(row["ms_cold"], 3),
              "max_abs_error=", round(row["max_abs_error_vs_cpu_rfft"], 6))


if __name__ == "__main__":
    main()
