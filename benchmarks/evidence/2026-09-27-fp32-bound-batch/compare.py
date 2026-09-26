#!/usr/bin/env python3
"""Recheck batched FP32 bounds and 29-step outputs against PR #52."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "benchmarks/evidence/2026-09-27-trust-guard/raw"


def read(path):
    return json.loads(path.read_text())


def main():
    baseline = read(BASE / "post-r1.json")
    patched = read(HERE / "raw/post-r1.json")
    base_trace = read(BASE / "post-profile/trace-summary.json")["rows"]
    post_trace = read(HERE / "raw/post-profile/trace-summary.json")["rows"]
    assert baseline["steps"] == patched["steps"] == 29
    assert [row["step"] for row in base_trace] == [row["step"] for row in post_trace]
    assert baseline["source_stable_during_run"] and patched["source_stable_during_run"]

    loss_diff = 0.0
    scale_diff = 0.0
    for before, after in zip(baseline["rows"], patched["rows"]):
        assert before["global_step"] == after["global_step"]
        loss_diff = max(loss_diff, abs(before["loss"] - after["loss"]))
        for field in ("qkv_scales_before", "qkv_scales_after"):
            left, right = before[field], after[field]
            if left is None:
                assert right is None
            else:
                assert set(left) == set(right)
                for key in left:
                    scale_diff = max(scale_diff, abs(left[key] - right[key]))

    result = {
        "schema": 1,
        "baseline_source_sha256": baseline["tiger_source_sha256_before"],
        "patched_source_sha256": patched["tiger_source_sha256_before"],
        "steps": baseline["steps"],
        "max_loss_abs_diff": loss_diff,
        "max_qkv_scale_abs_diff": scale_diff,
        "baseline_warnings": len(baseline["warnings"]),
        "patched_warnings": len(patched["warnings"]),
        "profile_steps": [row["step"] for row in base_trace],
        "baseline_local_scalar_dense": [row["local_scalar_dense"] for row in base_trace],
        "patched_local_scalar_dense": [row["local_scalar_dense"] for row in post_trace],
        "baseline_isfinite": [row["isfinite"] for row in base_trace],
        "patched_isfinite": [row["isfinite"] for row in post_trace],
    }
    (HERE / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
