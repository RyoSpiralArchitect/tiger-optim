"""Run all frozen paired CUDA spectral controls; retain every outcome."""
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
    plan_bytes = (HERE / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "confirmation_index.json"
    if index_path.exists():
        parser.error("confirmation index already exists")
    records = []
    for precision, seeds in plan["precisions"].items():
        for i, seed in enumerate(seeds):
            order = plan["modes"][i % len(plan["modes"]):] + plan["modes"][:i % len(plan["modes"])]
            for mode in order:
                label = f"confirm-{precision}-{mode}-{seed}"
                destination = args.output_dir / (label + ".json")
                command = [sys.executable, str(ROOT / "benchmarks/bench_associative_recall.py"),
                           "--mode", mode, "--seed", str(seed), "--precision", precision,
                           "--measure-qkv-updates", "--output", str(destination)]
                options = {key: plan[key] for key in ("device", "cpu_threads", "lr", "steps", "batch_size", "eval_interval", "eval_size", "test_size")}
                options.update(plan["model"])
                for key, value in options.items():
                    command.extend(["--" + key.replace("_", "-"), str(value)])
                print("START " + label, flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
                record = json.loads(destination.read_text())
                assert record["status"] == "complete" and len(record["steps"]) == plan["steps"]
                records.append({"file": destination.name, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
                index_path.write_text(json.dumps({"plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                                                  "complete": len(records) == sum(len(s) for s in plan["precisions"].values()) * len(plan["modes"]),
                                                  "results": records}, indent=2) + "\n")
    print(index_path.read_text(), flush=True)


if __name__ == "__main__":
    main()
