"""Replay original input hashes and retain rare equal-score key ordering."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    benchmark = root / "benchmarks/bench_associative_recall.py"
    plan = json.loads((root / "benchmarks/evidence/2026-10-02-cuda-spectral/plan.json").read_text())
    spec = importlib.util.spec_from_file_location("recall", benchmark)
    recall = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recall)
    torch.set_default_device("cpu")
    torch.set_num_threads(2)
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip() == ""
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               "benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
               "replayer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "environment": {"torch": torch.__version__, "architecture": platform.machine(), "device": "cpu"},
               "seeds": {}}
    config = {k: plan["model"][k] for k in ("symbols", "pairs", "gap", "queries")}
    for seed in plan["seeds"]:
        reference = json.loads((args.raw_dir / f"confirm-bf16-adamw-{seed}.json").read_text())
        assert reference["source_sha256"]["benchmarks/bench_associative_recall.py"] == receipt["benchmark_sha256"]
        arrays = {}
        splits = {}
        for name, offset, count in (("train", 10000, plan["steps"] * plan["batch_size"]),
                                     ("validation", 20000, plan["eval_size"]), ("test", 30000, plan["test_size"])):
            data = recall.corpus(seed + offset, count, **config)
            digest = recall.tensor_hash(zip(("tokens", "targets"), data))
            assert digest == reference["data_sha256"][name], (seed, name)
            rng = torch.Generator(device="cpu").manual_seed(seed + offset)
            scores = torch.rand(count, config["symbols"], generator=rng, device="cpu")
            sorted_scores = scores.sort(dim=1).values
            rows = (sorted_scores[:, 1:] == sorted_scores[:, :-1]).any(dim=1).nonzero().flatten()
            arrays[name + "_rows"] = rows.numpy()
            arrays[name + "_keys"] = data[0][rows, :2 * config["pairs"]:2].numpy()
            splits[name] = {"sha256": digest, "native_replay_bitwise_equal": True, "tie_row_count": len(rows)}
        destination = args.output_dir / f"key-ties-{seed}.npz"
        np.savez_compressed(destination, **arrays)
        receipt["seeds"][str(seed)] = {"file": destination.name, "file_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                                     "splits": splits}
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Original-runtime input hashes reproduced: 21 of 21")


if __name__ == "__main__":
    main()
