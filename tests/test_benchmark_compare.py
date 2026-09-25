import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "benchmarks" / "bench_compare_optim.py"


def test_comparison_records_paired_identity_and_separate_compute_times(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, str(SCRIPT), "--device", "cpu", "--steps", "1",
         "--warmup", "0", "--seed", "17", "--modes", "adamw", "tiger_v21_full"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )

    paths = sorted((tmp_path / "benchmarks" / "results").glob("compare-*.json"))
    assert len(paths) == 2
    payloads = {payload["summary"]["mode"]: payload
                for payload in (json.loads(path.read_text()) for path in paths)}
    assert set(payloads) == {"adamw", "tiger_v21_full"}
    assert len({payload["run"]["comparison_id"] for payload in payloads.values()}) == 1
    assert len({payload["run"]["run_id"] for payload in payloads.values()}) == 2

    for mode, payload in payloads.items():
        summary, run, series = payload["summary"], payload["run"], payload["series"]
        assert run["git"]["commit"] and len(run["git"]["commit"]) == 40
        assert isinstance(run["git"]["dirty"], bool)
        assert run["timestamp_utc"].endswith("+00:00")
        assert run["python"] and run["pytorch"] and run["os"]
        assert run["device"]["name"] and run["device"]["resolved"] == "cpu"
        assert run["cli"]["parsed"]["seed"] == 17
        assert run["cli"]["parsed"]["modes"] == ["adamw", "tiger_v21_full"]
        assert run["effective_arguments"]["mode"] == mode
        assert run["optimizer_config_initial"]["defaults"]
        assert run["optimizer_config_initial"]["param_groups"]
        assert run["optimizer_config_final"]["param_groups"]

        assert len(series["ms"]) == len(series["ms_forward_backward"]) == len(series["ms_optimizer"]) == len(series["loss"]) == 1
        assert series["ms"][0] == pytest.approx(series["ms_forward_backward"][0] + series["ms_optimizer"][0])
        assert summary["ms_median"] == pytest.approx(series["ms"][0])
        assert summary["ms_optimizer_median"] == pytest.approx(series["ms_optimizer"][0])
        assert series["ms_optimizer"][0] > 0

    assert payloads["adamw"]["run"]["optimizer_config_initial"]["param_groups"][0]["options"]["betas"] == [0.9, 0.999]
    assert payloads["tiger_v21_full"]["run"]["optimizer_config_initial"]["defaults"]["betas"] == [0.9, 0.98]
