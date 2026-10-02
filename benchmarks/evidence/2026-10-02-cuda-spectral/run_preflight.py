"""Run every predeclared CUDA learnability check using validation only."""
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
    plan_bytes = (HERE / "preflight_plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = args.output_dir / "preflight_selection.json"
    if selection.exists():
        parser.error("selection already exists")
    records = {}
    for task, config in plan["tasks"].items():
        records[task] = {}
        for mode, rates in plan["learning_rates"].items():
            records[task][mode] = []
            for lr in rates:
                label = f"preflight-{task}-{mode}-{lr}"
                output = args.output_dir / (label + ".json")
                command = [sys.executable, str(ROOT / "benchmarks/bench_associative_recall.py"),
                           "--mode", mode, "--lr", str(lr), "--development",
                           "--measure-qkv-updates", "--output", str(output)]
                options = {key: plan[key] for key in ("device", "precision", "cpu_threads", "eval_interval", "eval_size", "test_size")}
                options.update({key: config[key] for key in ("seed", "steps", "batch_size")})
                options.update(config["model"])
                for key, value in options.items():
                    command.extend(["--" + key.replace("_", "-"), str(value)])
                print("START " + label, flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
                r = json.loads(output.read_text())
                assert r["status"] == "complete" and "test" not in r
                gate = plan["binding_gate"]
                original = r["evaluations"][-1]["validation"]
                passed = (original["accuracy"] >= gate["original_min"] and
                          r["binding_check"]["rebound"]["accuracy"] >= gate["rebound_min"] and
                          r["binding_check"]["old_targets"]["accuracy"] <= gate["obsolete_max"])
                records[task][mode].append({"file": output.name, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                                            "lr": lr, "validation_ce": original["ce"], "binding_gate_passed": passed})
    chosen = None
    for task in ("medium", "short"):
        eligible = {mode: [r for r in candidates if r["binding_gate_passed"]]
                    for mode, candidates in records[task].items()}
        if all(eligible.values()):
            chosen = {"task": task, "learning_rates": {mode: min(runs, key=lambda r: r["validation_ce"])["lr"]
                                                      for mode, runs in eligible.items()}}
            break
    selection.write_text(json.dumps({"plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                                     "candidates": records, "selected": chosen}, indent=2) + "\n")
    print(selection.read_text(), flush=True)


if __name__ == "__main__":
    main()
