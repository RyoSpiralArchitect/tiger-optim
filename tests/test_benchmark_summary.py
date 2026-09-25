import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "benchmarks" / "summarize_results.py"


def test_benchmark_summary_script_renders_markdown(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    baseline_old = {
        "summary": {
            "device": "cpu",
            "mode": "adamw",
            "ms_median": 9.0,
            "ms_mean": 9.5,
            "loss_last": 0.6,
            "loss_min": 0.6,
            "loss_mean": 0.6,
        }
    }
    baseline_new = {
        "summary": {
            "device": "cpu",
            "mode": "adamw",
            "ms_median": 10.0,
            "ms_mean": 10.5,
            "loss_last": 0.5,
            "loss_min": 0.5,
            "loss_mean": 0.5,
        }
    }
    sample = {
        "summary": {
            "device": "cpu",
            "mode": "tiger_v21_full",
            "ms_median": 12.5,
            "ms_mean": 13.0,
            "loss_last": 0.123,
            "loss_min": 0.12,
            "loss_mean": 0.2,
        }
    }
    (results_dir / "compare-adamw-cpu-20240101-000000.json").write_text(
        json.dumps(baseline_old),
        encoding="utf-8",
    )
    (results_dir / "compare-adamw-cpu-20240101-000001.json").write_text(
        json.dumps(baseline_new),
        encoding="utf-8",
    )
    (results_dir / "compare-tiger_v21_full-cpu-20240101-000002.json").write_text(
        json.dumps(sample),
        encoding="utf-8",
    )

    markdown_path = results_dir / "summary.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pattern",
            str(results_dir / "compare-*.json"),
            "--markdown-out",
            str(markdown_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "| Device | Mode | Median ms |" in proc.stdout
    assert "Delta vs AdamW" in proc.stdout
    assert proc.stdout.count("| cpu | adamw |") == 1
    assert "| cpu | adamw | 10.000 | 10.500 | 0.500000 | +0.0% |" in proc.stdout
    assert "| cpu | tiger_v21_full | 12.500 | 13.000 | 0.123000 | +25.0% |" in proc.stdout
    assert markdown_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "tiger_v21_full" in markdown
    assert "12.500" in markdown
