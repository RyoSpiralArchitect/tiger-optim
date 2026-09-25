"""One bounded end-to-end contract for the toy CPU learning smoke."""

import json
import math
import subprocess
import sys
from pathlib import Path


def test_quality_smoke_tiger_learns_on_seed_11(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "benchmarks/bench_quality_smoke.py"
    output = tmp_path / "quality-smoke.json"
    completed = subprocess.run(
        [sys.executable, str(script), "--seeds", "11", "--steps", "200",
         "--output", str(output)],
        cwd=root, capture_output=True, text=True, timeout=120, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["environment"]["device"] == "cpu"
    assert result["task"]["steps"] == 200
    assert len(result["seeds"]) == 1
    runs = {run["optimizer"]: run for run in result["seeds"][0]["runs"]}
    assert set(runs) == {"AdamW", "Tiger"}
    assert all(run["status"] == "complete" for run in runs.values())
    assert [point["step"] for point in runs["Tiger"]["trace"]] == [0, 50, 100, 150, 200]
    initial = runs["Tiger"]["trace"][0]["heldout_mse"]
    final = runs["Tiger"]["trace"][-1]["heldout_mse"]
    assert math.isclose(initial, runs["AdamW"]["trace"][0]["heldout_mse"], rel_tol=0, abs_tol=1e-7)
    assert math.isfinite(final)
    # The fixed-rate run regressed above 0.2 on this seed; keep the schedule's
    # meaningful held-out improvement as a regression gate.
    assert final < 0.2
