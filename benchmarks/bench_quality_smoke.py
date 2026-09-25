#!/usr/bin/env python3
"""CPU-only toy held-out learning smoke; not a general optimizer ranking.

Both optimizers receive the same initialized model, synthetic data, minibatch
indices, and step budget for each seed. AdamW is a fixed-LR reference. Tiger
uses the no-AGC cosine recipe that learned this task in a bounded local probe.
The different schedules make this a learning smoke, not a controlled claim of
optimizer superiority.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tiger_optim import Tiger  # noqa: E402

DEFAULT_SEEDS = (11, 23, 37)
TRAIN_N = 768
HELDOUT_N = 256
BATCH_SIZE = 64
TRAIN_LABEL_NOISE_STD = 0.05
CPU = torch.device("cpu")


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def _teacher(x):
    return (
        torch.sin(1.5 * x[:, :1])
        + 0.4 * x[:, 1:2].square()
        + 0.5 * x[:, :1] * x[:, 1:2]
    )


def _seed_data(seed, steps):
    data_rng = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.rand((TRAIN_N + HELDOUT_N, 2), generator=data_rng, device=CPU) * 4 - 2
    clean_y = _teacher(x)
    noise = TRAIN_LABEL_NOISE_STD * torch.randn(
        (TRAIN_N, 1), generator=data_rng, device=CPU)
    batch_rng = torch.Generator(device="cpu").manual_seed(seed + 9000)
    batches = torch.randint(
        0, TRAIN_N, (steps, BATCH_SIZE), generator=batch_rng, device=CPU)
    return x[:TRAIN_N], clean_y[:TRAIN_N] + noise, x[TRAIN_N:], clean_y[TRAIN_N:], batches


def _evaluation_steps(steps):
    return sorted(set(range(0, steps + 1, 50)) | {steps})


def _check_cpu(model, tensors):
    if any(param.device.type != "cpu" for param in model.parameters()):
        raise RuntimeError("quality smoke requires CPU model parameters")
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise RuntimeError("quality smoke requires CPU data and minibatches")


def _optimizer(model, name, steps):
    if name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        config = {"lr": 1e-3, "weight_decay": 0.01,
                  "betas": [0.9, 0.999], "eps": 1e-8,
                  "scheduler": None}
        return optimizer, None, config
    if name == "Tiger":
        group = {"params": list(model.parameters()),
                 "rms_clip_threshold": 1.0,
                 "rms_clip_granularity": "param"}
        optimizer = Tiger(
            [group], lr=1e-2, weight_decay=0.01,
            factored=True, precond_alpha=1.0, trust_space="precond",
            update_buffer_dtype="fp32", agc_clip=0.0, trust_clip=5.0,
            auto_lr=False, auto_blend=False,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=1e-4)
        config = {
            "lr": 1e-2, "weight_decay": 0.01, "factored": True,
            "precond_alpha": 1.0, "trust_space": "precond",
            "update_buffer_dtype": "fp32", "agc_clip": 0.0,
            "trust_clip": 5.0, "rms_clip_threshold": 1.0,
            "rms_clip_granularity": "param", "auto_lr": False,
            "auto_blend": False,
            "scheduler": {"name": "CosineAnnealingLR", "T_max": steps,
                          "eta_min": 1e-4,
                          "order": "scheduler.step() after optimizer.step()"},
        }
        return optimizer, scheduler, config
    raise ValueError("unknown optimizer: " + name)


def _train_one(name, initial_state, data, steps, eval_steps):
    train_x, train_y, heldout_x, heldout_y, batches = data
    model = MLP().to(CPU)
    model.load_state_dict(initial_state)
    _check_cpu(model, data)
    optimizer, scheduler, config = _optimizer(model, name, steps)
    trace = []
    status = "complete"
    stopped_at_step = None

    for step in range(steps + 1):
        if step in eval_steps:
            model.eval()
            with torch.no_grad():
                train_mse = torch.nn.functional.mse_loss(model(train_x), train_y).item()
                heldout_mse = torch.nn.functional.mse_loss(model(heldout_x), heldout_y).item()
            if not (math.isfinite(train_mse) and math.isfinite(heldout_mse)):
                status = "nonfinite_evaluation"
                stopped_at_step = step
                break
            trace.append({"step": step, "train_mse": train_mse,
                          "heldout_mse": heldout_mse,
                          "lr": float(optimizer.param_groups[0]["lr"])})
        if step == steps:
            break
        model.train()
        batch = batches[step]
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(train_x[batch]), train_y[batch])
        if not torch.isfinite(loss).item():
            status = "nonfinite_batch_loss"
            stopped_at_step = step
            break
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if any(not torch.isfinite(param).all().item() for param in model.parameters()):
            status = "nonfinite_parameter"
            stopped_at_step = step + 1
            break

    return {"optimizer": name, "config": config, "status": status,
            "stopped_at_step": stopped_at_step, "trace": trace}


def _git_metadata():
    def git(*args):
        try:
            result = subprocess.run(
                ["git", "-C", str(PROJECT_ROOT), *args],
                capture_output=True, text=True, timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=normal")
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(status) if status is not None else None,
            "tiger_py_sha256": hashlib.sha256(
                (PROJECT_ROOT / "src/tiger_optim/tiger.py").read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def run_quality_smoke(seeds=DEFAULT_SEEDS, steps=200):
    """Return a source-backed toy learning trace for each requested seed."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if not seeds:
        raise ValueError("at least one seed is required")
    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    eval_steps = _evaluation_steps(steps)
    result = {
        "label": "CPU-only toy held-out quality smoke; not a general optimizer ranking",
        "limitations": [
            "Single synthetic nonlinear regression task with three default seeds.",
            "AdamW uses a fixed learning rate while Tiger uses a cosine schedule.",
            "The Tiger recipe followed bounded local exploration and is not a broad tuning result.",
        ],
        "git": _git_metadata(),
        "environment": {"python": sys.version.split()[0],
                        "torch": torch.__version__,
                        "platform": platform.platform(),
                        "device": "cpu", "threads": torch.get_num_threads(),
                        "deterministic_algorithms": True,
                        "torch_default_device": str(torch.get_default_device())
                        if hasattr(torch, "get_default_device") else None},
        "task": {"kind": "2D nonlinear regression",
                 "teacher": "sin(1.5*x0) + 0.4*x1^2 + 0.5*x0*x1",
                 "input_distribution": "uniform[-2, 2]^2",
                 "train_n": TRAIN_N, "heldout_n": HELDOUT_N,
                 "train_label_noise_std": TRAIN_LABEL_NOISE_STD,
                 "heldout_labels": "noise-free", "model": "2-64-64-1 SiLU MLP",
                 "steps": steps, "batch_size": BATCH_SIZE,
                 "evaluation_steps": eval_steps,
                 "minibatches": "same precomputed indices for both optimizers per seed"},
        "seeds": [],
    }
    for seed in seeds:
        data = _seed_data(seed, steps)
        torch.manual_seed(seed + 1000)
        initial = copy.deepcopy(MLP().to(CPU).state_dict())
        runs = [_train_one(name, initial, data, steps, eval_steps)
                for name in ("AdamW", "Tiger")]
        result["seeds"].append({"seed": int(seed), "runs": runs})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
                        help="integer seeds (default: 11 23 37)")
    parser.add_argument("--steps", type=int, default=200,
                        help="equal training steps per optimizer (default: 200)")
    parser.add_argument("--output", type=Path,
                        help="JSON output path (default: timestamped benchmarks/results path)")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = PROJECT_ROOT / "benchmarks/results" / ("quality-smoke-" + stamp + ".json")
    if output.exists():
        parser.error("output already exists: " + str(output))
    result = run_quality_smoke(args.seeds, args.steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    print("wrote " + str(output))
    for item in result["seeds"]:
        finals = {run["optimizer"]: run["trace"][-1]["heldout_mse"]
                  if run["status"] == "complete" else None for run in item["runs"]}
        print("seed=" + str(item["seed"]) + " final_heldout_mse=" + json.dumps(finals, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
