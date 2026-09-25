"""Bounded Tiger LR follow-up: seed 11 at .01/.03, conditional .01 on 23/37."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import runpy
import time
from pathlib import Path

BASE = Path("/tmp/tiger-optim-20260925-quality")
probe = runpy.run_path(str(BASE / "quality_probe.py"))
torch = probe["torch"]
Tiger = probe["Tiger"]
MLP = probe["MLP"]


def run(seed: int, lr: float):
    train_x, train_y, heldout_x, heldout_y, batches = probe["data_for_seed"](seed)
    torch.manual_seed(seed + 1000)
    initial_state = copy.deepcopy(MLP().to("cpu").state_dict())
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
    start = time.perf_counter()
    status = "complete"
    stopped_at_step = None
    for step in range(probe["STEPS"] + 1):
        if step in probe["EVAL_STEPS"]:
            model.eval()
            with torch.no_grad():
                train_mse = torch.nn.functional.mse_loss(model(train_x), train_y).item()
                heldout_mse = torch.nn.functional.mse_loss(model(heldout_x), heldout_y).item()
            trace.append({"step": step, "train_mse": train_mse,
                          "heldout_mse": heldout_mse})
            if not (math.isfinite(train_mse) and math.isfinite(heldout_mse)):
                status = "nonfinite_evaluation"
                stopped_at_step = step
                break
        if step == probe["STEPS"]:
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
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            status = "nonfinite_parameter"
            stopped_at_step = step + 1
            break
    return {"seed": seed, "lr": lr, "status": status,
            "stopped_at_step": stopped_at_step, "trace": trace,
            "elapsed_seconds": time.perf_counter() - start}


def main():
    assert torch.get_default_device() == torch.device("cpu")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    original = json.loads((BASE / "results.json").read_text())
    previous = json.loads((BASE / "lr_sensitivity_results.json").read_text())
    source_hash = hashlib.sha256((probe["REPO"] / "src/tiger_optim/tiger.py").read_bytes()).hexdigest()
    assert source_hash == original["source"]["tiger_py_sha256"] == previous["tiger_py_sha256"]
    prior_seed11 = next(item for item in previous["runs"] if item["lr"] == 3e-3)
    threshold = prior_seed11["trace"][-1]["heldout_mse"]
    result = {
        "label": "exploratory one-dimensional LR check; no superiority claim",
        "task": "same synthetic task, model, data, minibatches and 200-step budget as quality_probe.py",
        "config": {"factored": True, "precond_alpha": 1.0,
                   "trust_space": "precond", "agc_clip": 0.0,
                   "weight_decay": 0.01, "trust_clip": 5.0,
                   "rms_clip_threshold": 1.0,
                   "rms_clip_granularity": "param",
                   "update_buffer_dtype": "fp32",
                   "auto_lr": False, "auto_blend": False},
        "tiger_py_sha256": source_hash,
        "quality_probe_sha256": hashlib.sha256((BASE / "quality_probe.py").read_bytes()).hexdigest(),
        "previous_lr_results_sha256": hashlib.sha256((BASE / "lr_sensitivity_results.json").read_bytes()).hexdigest(),
        "previous_seed11_lr3e-3_heldout_mse": threshold,
        "torch_default_device": str(torch.get_default_device()),
        "runs": [], "condition_met_for_other_seeds": False,
    }
    for lr in (1e-2, 3e-2):
        entry = run(11, lr)
        result["runs"].append(entry)
        print(json.dumps({"seed": 11, "lr": lr, "status": entry["status"],
                          "stopped_at_step": entry["stopped_at_step"],
                          "last_heldout_mse": entry["trace"][-1]["heldout_mse"]},
                         sort_keys=True), flush=True)
    lr1e2 = result["runs"][0]
    if lr1e2["status"] == "complete" and lr1e2["trace"][-1]["heldout_mse"] < threshold:
        result["condition_met_for_other_seeds"] = True
        for seed in (23, 37):
            entry = run(seed, 1e-2)
            result["runs"].append(entry)
            print(json.dumps({"seed": seed, "lr": 1e-2,
                              "status": entry["status"],
                              "stopped_at_step": entry["stopped_at_step"],
                              "last_heldout_mse": entry["trace"][-1]["heldout_mse"]},
                             sort_keys=True), flush=True)
    path = BASE / "lr_followup_results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("results=" + str(path), flush=True)


if __name__ == "__main__":
    main()
