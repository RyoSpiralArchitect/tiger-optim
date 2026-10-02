"""Recreate initial hashes on the original runtime and retain embedding bits.

Run outside the measured checkout, keeping its Git status clean. CPU normal
draws can differ across architectures; these small fixtures restore the exact
embedding values while all other parameters still come from seeded generation.
"""
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
               "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "architecture": platform.machine(), "device": "cpu", "cpu_threads": 2},
               "seeds": {}}
    for seed in plan["seeds"]:
        reference = json.loads((args.raw_dir / f"confirm-bf16-adamw-{seed}.json").read_text())
        assert reference["source_sha256"]["benchmarks/bench_associative_recall.py"] == receipt["benchmark_sha256"]
        torch.manual_seed(seed)
        model = recall.RecallTransformer(**plan["model"])
        initial_hash = recall.tensor_hash(model.state_dict().items())
        assert initial_hash == reference["initial_state_sha256"], seed
        destination = args.output_dir / f"embeddings-{seed}.npz"
        np.savez_compressed(destination, token=model.token.weight.detach().numpy(),
                            position=model.position.weight.detach().numpy())
        receipt["seeds"][str(seed)] = {"initial_state_sha256": initial_hash, "native_replay_bitwise_equal": True,
                                     "embedding_file": destination.name,
                                     "embedding_file_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Original-runtime initialization hashes reproduced:", len(receipt["seeds"]), "of 7")


if __name__ == "__main__":
    main()
