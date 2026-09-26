#!/usr/bin/env python3
"""Recheck QKV view-path output and MPS operator evidence against merged main."""

import gzip
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def trace_counts(path):
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    events = payload["traceEvents"] if isinstance(payload, dict) else payload
    markers = [
        event for event in events
        if event.get("ph") == "X" and event.get("name", "").startswith("diagnose/optimizer_step_")
    ]
    rows = []
    for marker in markers:
        start, end = marker["ts"], marker["ts"] + marker["dur"]
        names = [
            event.get("name") for event in events
            if event.get("ph") == "X" and start <= event.get("ts", -1) < end
        ]
        rows.append({
            "step": int(marker["name"].rsplit("_", 1)[1]),
            "stack": names.count("aten::stack"),
            "mean": names.count("aten::mean"),
            "fft_rfft": names.count("aten::fft_rfft"),
            "local_scalar_dense": names.count("aten::_local_scalar_dense"),
        })
    return sorted(rows, key=lambda row: row["step"])


def main():
    base = read(HERE / "raw/baseline.json")
    view = read(HERE / "raw/view.json")
    base_profile = read(HERE / "raw/baseline-profile/summary.json")
    view_profile = read(HERE / "raw/view-profile/summary.json")
    assert base["steps"] == view["steps"] == 29
    assert base["source_stable_during_run"] and view["source_stable_during_run"]
    assert base_profile["source_stable_during_run"] and view_profile["source_stable_during_run"]
    source_path = "src/tiger_optim/tiger.py"
    assert base["tiger_source_sha256_before"] == base_profile["source_sha256_before"][source_path]
    assert view["tiger_source_sha256_before"] == view_profile["source_sha256_before"][source_path]

    max_loss_diff = 0.0
    max_scale_diff = 0.0
    for left, right in zip(base["rows"], view["rows"]):
        assert left["global_step"] == right["global_step"]
        max_loss_diff = max(max_loss_diff, abs(left["loss"] - right["loss"]))
        for field in ("qkv_scales_before", "qkv_scales_after"):
            lhs, rhs = left[field], right[field]
            if lhs is None:
                assert rhs is None
            else:
                assert lhs.keys() == rhs.keys()
                for key in lhs:
                    max_scale_diff = max(max_scale_diff, abs(lhs[key] - rhs[key]))

    helper_rows = []
    for run in (1, 2):
        probe = read(HERE / f"raw/helper-probe-r{run}.json")
        assert probe["source_stable_during_run"]
        assert probe["source_sha256_before"] == view["tiger_source_sha256_before"]
        for row in probe["rows"]:
            assert row["metrics_exact_equal"]
            assert row["view_shares_source_storage"] and row["stack_allocates_separate_storage"]
            helper_rows.append({
                "run": run,
                "chunk_length": row["chunk_length"],
                "source_bytes": row["source_bytes"],
                "stack_median_ms": row["stack_median_ms"],
                "view_median_ms": row["view_median_ms"],
            })

    result = {
        "schema": 1,
        "baseline_source_sha256": base["tiger_source_sha256_before"],
        "view_source_sha256": view["tiger_source_sha256_before"],
        "steps": base["steps"],
        "max_loss_abs_diff": max_loss_diff,
        "max_qkv_scale_abs_diff": max_scale_diff,
        "baseline_warnings": len(base["warnings"]),
        "view_warnings": len(view["warnings"]),
        "baseline_trace": trace_counts(HERE / "raw/baseline-profile/trace.json.gz"),
        "view_trace": trace_counts(HERE / "raw/view-profile/trace.json.gz"),
        "helper_runs": helper_rows,
    }
    (HERE / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "max_loss_abs_diff": max_loss_diff,
        "max_qkv_scale_abs_diff": max_scale_diff,
        "baseline_trace": result["baseline_trace"],
        "view_trace": result["view_trace"],
        "large_helper_runs": [row for row in helper_rows if row["chunk_length"] >= 1048576],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
