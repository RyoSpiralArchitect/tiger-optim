"""Run the frozen 2x2 QKV scale study without tuning or dropping failures."""
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
    parser.add_argument("--phase", choices=("cpu", "mps"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan_bytes = (HERE / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable]
    if sys.flags.no_site:
        command.append("-S")
    command.append(str(ROOT / "benchmarks/bench_associative_recall.py"))
    for key in ("batch_size", "steps", "eval_interval", "eval_size", "test_size", "precision", "cpu_threads", "lr"):
        command.extend(["--" + key.replace("_", "-"), str(plan[key])])
    for key, value in plan["model"].items():
        command.extend(["--" + key, str(value)])
    command.extend(["--device", args.phase, "--measure-qkv-updates"])
    seeds = plan["seeds"] if args.phase == "cpu" else [plan["mps_replay"]["seed"]]
    modes = list(plan["cells"]) if args.phase == "cpu" else plan["mps_replay"]["modes"]
    run_index = {"plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "phase": args.phase, "results": []}
    index_path = args.output_dir / (args.phase + "-index.json")
    if index_path.exists():
        parser.error("run index already exists; use a fresh output directory")
    for index, seed in enumerate(seeds):
        shift = index % len(modes)
        for mode in modes[shift:] + modes[:shift]:
            output = args.output_dir / f"{args.phase}-{mode}-{seed}.json"
            print("START " + output.name, flush=True)
            subprocess.run(command + ["--mode", mode, "--seed", str(seed), "--output", str(output)], cwd=ROOT, check=True)
            raw = output.read_bytes()
            record = json.loads(raw)
            assert record["status"] == "complete"
            run_index["results"].append({"file": output.name, "sha256": hashlib.sha256(raw).hexdigest()})
    index_path.write_text(json.dumps(run_index, indent=2) + "\n")


if __name__ == "__main__":
    main()
