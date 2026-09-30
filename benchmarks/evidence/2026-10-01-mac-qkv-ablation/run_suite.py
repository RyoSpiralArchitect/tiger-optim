"""Select rates on development data, then run frozen Mac component ablations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def qualifies(record):
    return (record["evaluations"][-1]["validation"]["accuracy"] >= 0.9
            and record["binding_check"]["rebound"]["accuracy"] >= 0.9
            and record["binding_check"]["old_targets"]["accuracy"] <= 0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan_bytes = (HERE / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable]
    if sys.flags.no_site:
        command.append("-S")
    command.append(str(ROOT / "benchmarks/bench_associative_recall.py"))
    for key in ("batch_size", "steps", "eval_interval", "eval_size", "test_size", "device", "precision", "cpu_threads"):
        command.extend(["--" + key.replace("_", "-"), str(plan[key])])
    for key, value in plan["model"].items():
        command.extend(["--" + key, str(value)])

    def invoke(mode, lr, seed, prefix, development=False):
        suffix = str(lr) if development else str(seed)
        output = args.output_dir / (prefix + "-" + mode + "-" + suffix + ".json")
        cmd = command + ["--mode", mode, "--lr", str(lr), "--seed", str(seed), "--output", str(output)]
        if development:
            cmd.append("--development")
        print("START " + output.name, flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        record = json.loads(output.read_text())
        assert record["status"] == "complete"
        return record

    selection_path = args.output_dir / "selection.json"
    if args.phase == "development":
        if selection_path.exists():
            parser.error("selection already exists")
        selection = {}
        for mode, rates in plan["development"]["rates"].items():
            candidates = []
            for rate in rates:
                record = invoke(mode, rate, plan["development"]["seed"], "dev", True)
                candidates.append({"lr": rate, "ce": record["evaluations"][-1]["validation"]["ce"],
                                   "qualified": qualifies(record)})
            selected = min(candidates, key=lambda row: row["ce"])
            selection[mode] = {"candidates": candidates, "lr": selected["lr"], "qualified": selected["qualified"]}
        selection_path.write_text(json.dumps({"plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                                               "selected": selection}, indent=2) + "\n")
        print(selection_path.read_text(), flush=True)
    else:
        selection = json.loads(selection_path.read_text())
        assert selection["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest()
        if not all(row["qualified"] for row in selection["selected"].values()):
            parser.error("a selected development recipe failed the binding-sensitive learning gate")
        for index, seed in enumerate(plan["confirmation"]["seeds"]):
            modes = plan["confirmation"]["modes"]
            order = modes[index:] + modes[:index]
            for mode in order:
                family = "adamw" if mode == "adamw" else "tiger-full"
                invoke(mode, selection["selected"][family]["lr"], seed, "confirm")


if __name__ == "__main__":
    main()
