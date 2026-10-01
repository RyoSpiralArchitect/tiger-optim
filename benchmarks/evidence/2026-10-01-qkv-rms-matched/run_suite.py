"""Run new paired seeds, fixing each asymmetric QKV step to reference RMS."""
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
    index_path = args.output_dir / "index.json"
    if index_path.exists():
        parser.error("run index already exists; use a fresh output directory")
    command = [sys.executable]
    if sys.flags.no_site:
        command.append("-S")
    command.append(str(ROOT / "benchmarks/bench_associative_recall.py"))
    for key in ("batch_size", "steps", "eval_interval", "eval_size", "test_size", "device", "precision", "cpu_threads", "lr"):
        command.extend(["--" + key.replace("_", "-"), str(plan[key])])
    for key, value in plan["model"].items():
        command.extend(["--" + key, str(value)])
    command.append("--measure-qkv-updates")
    index = {"plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "results": []}
    for seed in plan["seeds"]:
        reference = args.output_dir / f"reference-uniform-{seed}.json"
        for label, recipe in plan["recipes"].items():
            output = args.output_dir / f"{label}-{seed}.json"
            cmd = command + ["--mode", recipe["mode"], "--seed", str(seed), "--output", str(output)]
            if recipe["rms_match"]:
                cmd.extend(["--qkv-rms-reference", str(reference)])
            print("START " + output.name, flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
            raw = output.read_bytes()
            assert json.loads(raw)["status"] == "complete"
            index["results"].append({"file": output.name, "sha256": hashlib.sha256(raw).hexdigest()})
    index_path.write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
