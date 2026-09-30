"""Replay a pinned Tiger implementation to verify QKV group controls in training."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import torch

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--control", choices=("group", "lr"), required=True)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if not 0 <= args.factor <= 1:
        parser.error("factor must be in [0, 1]")
    torch.set_default_device("cpu")
    if args.device == "cpu":
        torch.set_num_threads(1)
    commit = subprocess.check_output(
        ["git", "rev-parse", args.source_ref + "^{commit}"], cwd=ROOT, text=True,
    ).strip()
    source = subprocess.check_output(
        ["git", "show", commit + ":src/tiger_optim/tiger.py"], cwd=ROOT,
    )
    spec = importlib.util.spec_from_file_location(
        "toy", ROOT / "benchmarks/bench_toy_transformer_learning.py",
    )
    toy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy)
    module = types.ModuleType("tiger_optim._probe_tiger")
    module.__package__ = "tiger_optim"
    sys.modules[module.__name__] = module
    exec(compile(source, commit + ":src/tiger_optim/tiger.py", "exec"), module.__dict__)
    toy.Tiger = module.Tiger
    original_optimizer = toy._optimizer
    captured = {}
    def optimizer(model, mode, lr):
        opt, group = original_optimizer(model, mode, lr)
        group["lr_scale" if args.control == "group" else "lr"] *= args.factor
        captured["model"] = model
        captured["qkv_initial"] = model.attn.in_proj_weight.detach().clone()
        return opt, group
    toy._optimizer = optimizer
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit, "source_sha256": hashlib.sha256(source).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "toy_sha256": hashlib.sha256((ROOT / "benchmarks/bench_toy_transformer_learning.py").read_bytes()).hexdigest(),
        "git": toy._git_info(), "device": args.device,
        "control": args.control, "factor": args.factor,
        "python": sys.version, "torch": torch.__version__,
        "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
        "recipe": "tiger-full, lr .01, 100 updates, tail-cosine toward .001",
        "runs": [],
    }
    for seed in args.seeds:
        run = toy._run_seed(seed, 100, "tiger-full", 0.01, "tail-cosine", 0.001,
                            torch.device(args.device))
        run["final_state_sha256"] = toy._tensor_hash(captured["model"].state_dict().items())
        run["qkv_weight_max_abs_change"] = float((
            captured["model"].attn.in_proj_weight.detach() - captured["qkv_initial"]
        ).abs().max().item())
        result["runs"].append(run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"source": commit[:7], "control": args.control, "factor": args.factor,
        "device": args.device, "runs": [{"seed": r["seed"], "status": r["status"],
        "heldout": r["evaluations"][-1]["heldout"]} for r in result["runs"]]}), flush=True)

if __name__ == "__main__":
    main()
