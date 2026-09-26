#!/usr/bin/env python3
"""Count CPU profiler events inside each marked optimizer step."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.trace, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    markers = [event for event in events
               if event.get("ph") == "X" and event.get("name", "").startswith("diagnose/optimizer_step_")]
    rows = []
    for marker in markers:
        start = marker["ts"]
        end = start + marker["dur"]
        counts = Counter(event.get("name") for event in events
                         if event.get("ph") == "X"
                         and event.get("tid") == marker["tid"]
                         and start <= event.get("ts", -1) < end)
        rows.append({
            "step": int(marker["name"].rsplit("_", 1)[1]),
            "marker_duration_us": marker["dur"],
            "fft_rfft": counts["aten::fft_rfft"],
            "fft_r2c": counts["aten::_fft_r2c"],
            "local_scalar_dense": counts["aten::_local_scalar_dense"],
            "isfinite": counts["aten::isfinite"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": 1, "rows": rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
