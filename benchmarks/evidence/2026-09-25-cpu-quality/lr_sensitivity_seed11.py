"""Exploratory, bounded Tiger learning-rate sensitivity on seed 11 only."""

from __future__ import annotations

import copy
import hashlib
import json
import runpy
import time
from pathlib import Path

BASE = Path("/tmp/tiger-optim-20260925-quality")
probe = runpy.run_path(str(BASE / "quality_probe.py"))
torch = probe["torch"]
Tiger = probe["Tiger"]
MLP = probe["MLP"]


def run(lr: float, initial_state, data):
    train_x, train_y, heldout_x, heldout_y, batches = data
    model = MLP().to("cpu")
    model.load_state_dict(initial_state)
    probe["check_cpu"](model, train_x, train_y, heldout_x, heldout_y, batches)
    group = {"params": list(model.parameters()),
             "rms_clip_threshold": 1.0,
             "rms_clip_granularity": "param"}
    optimizer = Tiger(
        [group], lr=lr, weight_decay=0.01,
        factored=True, precond_alpha=1.0, trust_space="precond",
        update_buffer_dtype="fp32", agc_clip=0.0, trust_clip=5.0,
        auto_lr=False, auto_blend=False,
    )
    trace = []
    started = time.perf_counter()
    for step in range(probe["STEPS"] + 1):
        if step in probe["EVAL_STEPS"]:
            model.eval()
            with torch.no_grad():
                trace.append({
                    "step": step,
                    "train_mse": torch.nn.functional.mse_loss(model(train_x), train_y).item(),
                    "heldout_mse": torch.nn.functional.mse_loss(model(heldout_x), heldout_y).item(),
                })
        if step == probe["STEPS"]:
            break
        model.train()
        batch = batches[step]
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(train_x[batch]), train_y[batch])
        loss.backward()
        optimizer.step()
        assert all(torch.isfinite(p).all().item() for p in model.parameters())
    return {"lr": lr, "trace": trace, "elapsed_seconds": time.perf_counter() - started}


def main():
    assert torch.get_default_device() == torch.device("cpu")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed = 11
    data = probe["data_for_seed"](seed)
    torch.manual_seed(seed + 1000)
    initial_state = copy.deepcopy(MLP().to("cpu").state_dict())
    result = {
        "label": "exploratory seed-11 LR-scale sensitivity; no tuning or superiority claim",
        "seed": seed, "steps": probe["STEPS"], "agc_clip": 0.0,
        "sole_change_from_no_agc_ablation": "lr",
        "tested_lrs": [1e-3, 3e-3],
        "tiger_py_sha256": hashlib.sha256((probe["REPO"] / "src/tiger_optim/tiger.py").read_bytes()).hexdigest(),
        "quality_probe_sha256": hashlib.sha256((BASE / "quality_probe.py").read_bytes()).hexdigest(),
        "torch_default_device": str(torch.get_default_device()),
        "runs": [],
    }
    for lr in result["tested_lrs"]:
        entry = run(lr, initial_state, data)
        result["runs"].append(entry)
        print(json.dumps({"seed": seed, "lr": lr,
                          "final_train_mse": entry["trace"][-1]["train_mse"],
                          "final_heldout_mse": entry["trace"][-1]["heldout_mse"]},
                         sort_keys=True), flush=True)
    path = BASE / "lr_sensitivity_results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("results=" + str(path), flush=True)


if __name__ == "__main__":
    main()
