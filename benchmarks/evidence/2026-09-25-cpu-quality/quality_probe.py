"""Bounded CPU-only Tiger vs AdamW learning-quality probe.

Run with: python3 -S quality_probe.py
The explicit site-packages path bypasses this host's sitecustomize, which
otherwise changes the PyTorch default device to MPS.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SITE_PACKAGES = "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages"
REPO = Path("/Users/ryospiralarchitect/🌀SpiralReality🌀/tiger-optim")
OUT = Path("/tmp/tiger-optim-20260925-quality")
sys.path.insert(0, SITE_PACKAGES)
sys.path.insert(0, str(REPO / "src"))

import torch
from tiger_optim import Tiger


SEEDS = (11, 23, 37)
STEPS = 200
BATCH_SIZE = 64
TRAIN_N = 768
HELDOUT_N = 256
EVAL_STEPS = (0, 50, 100, 150, 200)


class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def teacher(x: torch.Tensor) -> torch.Tensor:
    return (
        torch.sin(1.5 * x[:, :1])
        + 0.4 * x[:, 1:2].square()
        + 0.5 * x[:, :1] * x[:, 1:2]
    )


def data_for_seed(seed: int):
    data_rng = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.rand((TRAIN_N + HELDOUT_N, 2), generator=data_rng, device="cpu") * 4 - 2
    clean = teacher(x)
    noise = 0.05 * torch.randn((TRAIN_N, 1), generator=data_rng, device="cpu")
    train_x, heldout_x = x[:TRAIN_N], x[TRAIN_N:]
    train_y, heldout_y = clean[:TRAIN_N] + noise, clean[TRAIN_N:]
    batch_rng = torch.Generator(device="cpu").manual_seed(seed + 9000)
    batches = torch.randint(0, TRAIN_N, (STEPS, BATCH_SIZE), generator=batch_rng, device="cpu")
    return train_x, train_y, heldout_x, heldout_y, batches


def check_cpu(model, *tensors):
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all(tensor.device.type == "cpu" for tensor in tensors)


def train(seed: int, optimizer_name: str, initial_state, data):
    train_x, train_y, heldout_x, heldout_y, batches = data
    model = MLP().to("cpu")
    model.load_state_dict(initial_state)
    check_cpu(model, train_x, train_y, heldout_x, heldout_y, batches)
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        config = {"lr": 1e-3, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8}
    elif optimizer_name == "Tiger":
        # README stability settings, with RMS clipping set on the param group
        # because Tiger.__init__ does not accept rms_clip_threshold directly.
        group = {"params": list(model.parameters()),
                 "rms_clip_threshold": 1.0,
                 "rms_clip_granularity": "param"}
        optimizer = Tiger(
            [group], lr=2e-4, weight_decay=0.01,
            factored=True, precond_alpha=1.0, trust_space="precond",
            update_buffer_dtype="fp32", agc_clip=0.02, trust_clip=5.0,
            auto_lr=False, auto_blend=False,
        )
        config = {"lr": 2e-4, "weight_decay": 0.01, "factored": True,
                  "precond_alpha": 1.0, "trust_space": "precond",
                  "update_buffer_dtype": "fp32", "agc_clip": 0.02,
                  "trust_clip": 5.0, "rms_clip_threshold": 1.0,
                  "rms_clip_granularity": "param", "auto_lr": False,
                  "auto_blend": False}
    else:
        raise ValueError(optimizer_name)

    trace = []
    started = time.perf_counter()
    for step in range(STEPS + 1):
        if step in EVAL_STEPS:
            model.eval()
            with torch.no_grad():
                train_mse = torch.nn.functional.mse_loss(model(train_x), train_y).item()
                heldout_mse = torch.nn.functional.mse_loss(model(heldout_x), heldout_y).item()
            trace.append({"step": step, "train_mse": train_mse,
                          "heldout_mse": heldout_mse})
        if step == STEPS:
            break
        model.train()
        batch = batches[step]
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(train_x[batch]), train_y[batch])
        loss.backward()
        optimizer.step()
        assert all(torch.isfinite(p).all().item() for p in model.parameters())
    return {"optimizer": optimizer_name, "config": config, "trace": trace,
            "elapsed_seconds": time.perf_counter() - started}


def main():
    assert torch.get_default_device() == torch.device("cpu")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = REPO / "src/tiger_optim/tiger.py"
    result = {
        "label": "preliminary CPU-only synthetic learning-quality probe; no superiority claim",
        "source": {"repo": str(REPO),
                   "head": subprocess.check_output(
                       ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                   "tiger_py_sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "environment": {"python": sys.version.split()[0],
                        "torch": torch.__version__,
                        "torch_default_device": str(torch.get_default_device()),
                        "torch_threads": torch.get_num_threads(),
                        "site_customization": "disabled via python3 -S"},
        "task": {"kind": "2D nonlinear regression", "train_n": TRAIN_N,
                 "heldout_n": HELDOUT_N, "train_label_noise_std": 0.05,
                 "heldout_labels": "noise-free", "steps": STEPS,
                 "batch_size": BATCH_SIZE, "model": "2-64-64-1 SiLU MLP",
                 "evaluation_steps": EVAL_STEPS,
                 "minibatches": "same precomputed indices for both optimizers per seed"},
        "seeds": [],
    }
    for seed in SEEDS:
        data = data_for_seed(seed)
        torch.manual_seed(seed + 1000)
        initial = copy.deepcopy(MLP().to("cpu").state_dict())
        runs = [train(seed, name, initial, data) for name in ("AdamW", "Tiger")]
        result["seeds"].append({"seed": seed, "runs": runs})
        print(json.dumps({"seed": seed, "final_heldout_mse": {
            run["optimizer"]: run["trace"][-1]["heldout_mse"] for run in runs}},
            sort_keys=True), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("results=" + str(OUT / "results.json"), flush=True)


if __name__ == "__main__":
    main()
