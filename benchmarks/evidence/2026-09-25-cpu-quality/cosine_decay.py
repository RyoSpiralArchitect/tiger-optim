"""Predeclared Tiger cosine-decay probe on the same three CPU synthetic seeds."""

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
SEEDS = (11, 23, 37)
LR_START = 1e-2
LR_END = 1e-4


def run(seed: int):
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
        [group], lr=LR_START, weight_decay=0.01,
        factored=True, precond_alpha=1.0, trust_space="precond",
        update_buffer_dtype="fp32", agc_clip=0.0, trust_clip=5.0,
        auto_lr=False, auto_blend=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=probe["STEPS"], eta_min=LR_END)
    trace = []
    status = "complete"
    stopped_at_step = None
    started = time.perf_counter()
    for step in range(probe["STEPS"] + 1):
        if step in probe["EVAL_STEPS"]:
            model.eval()
            with torch.no_grad():
                train_mse = torch.nn.functional.mse_loss(model(train_x), train_y).item()
                heldout_mse = torch.nn.functional.mse_loss(model(heldout_x), heldout_y).item()
            trace.append({"step": step,
                          "lr": float(optimizer.param_groups[0]["lr"]),
                          "train_mse": train_mse,
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
        scheduler.step()  # Explicitly after optimizer.step().
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            status = "nonfinite_parameter"
            stopped_at_step = step + 1
            break
    return {"seed": seed, "status": status,
            "stopped_at_step": stopped_at_step,
            "trace": trace, "elapsed_seconds": time.perf_counter() - started}


def main():
    assert torch.get_default_device() == torch.device("cpu")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    baseline = json.loads((BASE / "results.json").read_text())
    fixed = json.loads((BASE / "lr_followup_results.json").read_text())
    source_hash = hashlib.sha256((probe["REPO"] / "src/tiger_optim/tiger.py").read_bytes()).hexdigest()
    assert source_hash == baseline["source"]["tiger_py_sha256"] == fixed["tiger_py_sha256"]
    result = {
        "label": "predeclared cosine LR decay test of late regression; preliminary CPU-only evidence",
        "task": "same synthetic task, initial model, data, minibatches, and 200 steps as quality_probe.py",
        "optimizer_config": {"lr_start": LR_START, "lr_end": LR_END,
                             "weight_decay": 0.01, "factored": True,
                             "precond_alpha": 1.0, "trust_space": "precond",
                             "update_buffer_dtype": "fp32", "agc_clip": 0.0,
                             "trust_clip": 5.0, "rms_clip_threshold": 1.0,
                             "rms_clip_granularity": "param",
                             "auto_lr": False, "auto_blend": False},
        "scheduler": {"kind": "torch.optim.lr_scheduler.CosineAnnealingLR",
                      "T_max": probe["STEPS"], "eta_min": LR_END,
                      "order": "scheduler.step() after optimizer.step()"},
        "tiger_py_sha256": source_hash,
        "quality_probe_sha256": hashlib.sha256((BASE / "quality_probe.py").read_bytes()).hexdigest(),
        "fixed_lr_results_sha256": hashlib.sha256((BASE / "lr_followup_results.json").read_bytes()).hexdigest(),
        "torch_default_device": str(torch.get_default_device()),
        "runs": [],
    }
    for seed in SEEDS:
        entry = run(seed)
        fixed_entry = next(item for item in fixed["runs"] if item["seed"] == seed and item["lr"] == LR_START)
        base_entry = next(item for item in baseline["seeds"] if item["seed"] == seed)
        adamw = next(item for item in base_entry["runs"] if item["optimizer"] == "AdamW")
        entry["fixed_lr_heldout_trace"] = [
            {"step": item["step"], "heldout_mse": item["heldout_mse"]}
            for item in fixed_entry["trace"]]
        entry["adamw_heldout_trace"] = [
            {"step": item["step"], "heldout_mse": item["heldout_mse"]}
            for item in adamw["trace"]]
        result["runs"].append(entry)
        print(json.dumps({"seed": seed, "status": entry["status"],
                          "stopped_at_step": entry["stopped_at_step"],
                          "cosine_final_heldout_mse": entry["trace"][-1]["heldout_mse"],
                          "fixed_lr_final_heldout_mse": fixed_entry["trace"][-1]["heldout_mse"],
                          "adamw_final_heldout_mse": adamw["trace"][-1]["heldout_mse"]},
                         sort_keys=True), flush=True)
    path = BASE / "cosine_decay_results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("results=" + str(path), flush=True)


if __name__ == "__main__":
    main()
