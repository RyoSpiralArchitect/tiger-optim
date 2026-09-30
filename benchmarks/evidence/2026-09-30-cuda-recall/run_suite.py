"""Run the predeclared development search, then freeze rates for confirmation."""
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
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    plan = json.loads((HERE / "plan.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = [sys.executable, str(ROOT / "benchmarks/bench_associative_recall.py"),
              "--device", "cuda", "--precision", "bf16", "--cpu-threads", "2",
              "--batch-size", str(plan["batch_size"])]
    for key, value in plan["model"].items():
        common.extend(["--" + key, str(value)])

    def invoke(mode, lr, seed, steps, label, development=False):
        destination = args.output_dir / (label + ".json")
        command = common + ["--mode", mode, "--lr", str(lr), "--seed", str(seed),
                            "--steps", str(steps), "--output", str(destination)]
        if development:
            command.append("--development")
        print("START " + label, flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        record = json.loads(destination.read_text())
        assert record["status"] == "complete" and len(record["steps"]) == steps
        return record

    selection_path = args.output_dir / "selection.json"
    if args.phase == "development":
        if selection_path.exists():
            parser.error("selection already exists")
        chosen = {}
        dev = plan["development"]
        for mode, rates in dev["learning_rates"].items():
            candidates = []
            for rate in rates:
                record = invoke(mode, rate, dev["seed"], dev["steps"], "dev-" + mode + "-" + str(rate), True)
                candidates.append({"lr": rate, "validation_ce": record["evaluations"][-1]["validation"]["ce"]})
            chosen[mode] = {"candidates": candidates, "lr": min(candidates, key=lambda r: r["validation_ce"])["lr"]}
        selection_path.write_text(json.dumps({"plan_sha256": hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest(),
                                               "selected": chosen}, indent=2) + "\n")
        print(selection_path.read_text(), flush=True)
    else:
        selection = json.loads(selection_path.read_text())
        assert selection["plan_sha256"] == hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest()
        confirm = plan["confirmation"]
        for i, seed in enumerate(confirm["seeds"]):
            modes = confirm["modes"][i:] + confirm["modes"][:i]
            for mode in modes:
                rate = selection["selected"]["adamw" if mode == "adamw" else "tiger-full"]["lr"]
                invoke(mode, rate, seed, confirm["steps"], "confirm-" + mode + "-" + str(seed))


if __name__ == "__main__":
    main()
