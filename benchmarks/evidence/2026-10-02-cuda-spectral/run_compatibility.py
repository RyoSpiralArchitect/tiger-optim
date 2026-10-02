"""Replay the final default implementation against the frozen development run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan_bytes = (HERE / "compatibility_plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    reference_bytes = (args.output_dir / plan["reference_file"]).read_bytes()
    assert hashlib.sha256(reference_bytes).hexdigest() == plan["reference_sha256"]
    reference = json.loads(reference_bytes)
    assert reference["status"] == "complete" and reference["config"]["development"]
    command = [sys.executable, str(ROOT / "benchmarks/bench_associative_recall.py"),
               "--development", "--measure-qkv-updates", "--output", str(args.output_dir / plan["output_file"])]
    keys = ("mode", "device", "precision", "cpu_threads", "eval_interval", "eval_size", "test_size",
            "seed", "steps", "batch_size", "lr", "symbols", "pairs", "gap", "queries", "width", "layers", "heads")
    for key in keys:
        command.extend(["--" + key.replace("_", "-"), str(reference["config"][key])])
    subprocess.run(command, cwd=ROOT, check=True)
    raw = (args.output_dir / plan["output_file"]).read_bytes()
    r = json.loads(raw)
    assert r["status"] == "complete" and "test" not in r
    for key in ("initial_state_sha256", "final_state_sha256", "data_sha256", "binding_check"):
        assert r[key] == reference[key]
    assert [{k: p[k] for k in ("step", "loss", "lr", "grad_norm", "qkv_update")} for p in r["steps"]] == [
        {k: p[k] for k in ("step", "loss", "lr", "grad_norm", "qkv_update")} for p in reference["steps"]]
    assert [{k: p[k] for k in ("step", "validation", "qkv_scales", "qkv_feedback")} for p in r["evaluations"]] == [
        {k: p[k] for k in ("step", "validation", "qkv_scales", "qkv_feedback")} for p in reference["evaluations"]]
    receipt = {"plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "source_commit": r["git"]["head"],
               "reference_sha256": plan["reference_sha256"], "result_sha256": hashlib.sha256(raw).hexdigest(),
               "all_numeric_traces_equal": True, "final_weights_bitwise_equal": True}
    destination = args.output_dir / "final_compatibility.json"
    assert not destination.exists()
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(destination.read_text(), flush=True)


if __name__ == "__main__":
    main()
