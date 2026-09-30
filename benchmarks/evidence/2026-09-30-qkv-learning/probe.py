"""Bounded exploration of QKV adaptation on the existing causal toy task."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import torch

spec = importlib.util.spec_from_file_location(
    "toy", ROOT / "benchmarks/bench_toy_transformer_learning.py"
)
toy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toy)

CONFIGS = {
    "gain-zero": {"qkv_lr_gain": 0.0, "qkv_lr_interval": 25},
    "baseline": {"qkv_lr_gain": 0.02, "qkv_lr_interval": 25},
    "fast": {"qkv_lr_gain": 0.2, "qkv_lr_interval": 5},
    "fast-spectral": {"qkv_lr_gain": 0.2, "qkv_lr_interval": 5},
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    torch.set_default_device("cpu")
    torch.set_num_threads(1)
    original = toy.Tiger
    def configured_tiger(*positional, **kwargs):
        kwargs.update(CONFIGS[args.config])
        return original(*positional, **kwargs)
    toy.Tiger = configured_tiger
    mode = "tiger-full" if args.config.endswith("spectral") else "tiger-no-spectral"
    result = {
        "git": toy._git_info(), "config": args.config,
        "overrides": CONFIGS[args.config], "mode": mode,
        "python": sys.version, "torch": torch.__version__,
        "platform": platform.platform(), "device": "cpu",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT / "benchmarks/bench_toy_transformer_learning.py",
                      ROOT / "src/tiger_optim/tiger.py")},
        "runs": [toy._run_seed(seed, 100, mode, 0.01, "tail-cosine", 0.001,
                               torch.device("cpu")) for seed in args.seeds],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"config": args.config, "runs": [
        {"seed": run["seed"], "status": run["status"],
         "heldout": run["evaluations"][-1]["heldout"],
         "scales": run["steps"][-1]["qkv_scales"]} for run in result["runs"]
    ]}), flush=True)

if __name__ == "__main__":
    main()
