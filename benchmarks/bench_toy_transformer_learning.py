#!/usr/bin/env python3
"""Finite causal Transformer learning probe with held-out periodic-copy data.

The first three tokens of each sequence are distinct random symbols. Later
tokens repeat that triple, so predicting positions three onward requires the
model to recover a token from its causal context. Train and held-out triples
are disjoint. Each mode uses the same initialization and minibatch schedule
for a given seed; AdamW is a contextual reference because Tiger's tagged
parameter groups have their own per-tag learning-rate scales.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tiger_optim import Tiger, build_tagged_param_groups  # noqa: E402

VOCAB_SIZE = 24
SEQ_LEN = 12
TRAIN_N = 768
HELDOUT_N = 256
BATCH_SIZE = 48
CPU = torch.device("cpu")


class ToyCausalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        width = 48
        self.token = nn.Embedding(VOCAB_SIZE, width)
        self.position = nn.Embedding(SEQ_LEN, width)
        self.attn_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn_up = nn.Linear(width, 96)
        self.ffn_down = nn.Linear(96, width)
        self.out_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, VOCAB_SIZE)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token(tokens) + self.position(positions)
        normalized = self.attn_norm(hidden)
        mask = torch.ones(
            (tokens.shape[1], tokens.shape[1]), dtype=torch.bool,
            device=tokens.device,
        ).triu(1)
        attended, _ = self.attn(
            normalized, normalized, normalized,
            attn_mask=mask, need_weights=False,
        )
        hidden = hidden + attended
        hidden = hidden + self.ffn_down(F.gelu(self.ffn_up(self.ffn_norm(hidden))))
        return self.output(self.out_norm(hidden))


def _corpus(seed, steps):
    all_triples = torch.tensor(
        list(itertools.permutations(range(1, VOCAB_SIZE), 3)),
        dtype=torch.long, device=CPU,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chosen = all_triples[
        torch.randperm(len(all_triples), generator=generator, device=CPU)
        [:TRAIN_N + HELDOUT_N]
    ]
    full = chosen[:, torch.arange(SEQ_LEN + 1, device=CPU) % 3]
    train = (full[:TRAIN_N, :-1], full[:TRAIN_N, 1:])
    heldout = (full[TRAIN_N:, :-1], full[TRAIN_N:, 1:])
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 9000)
    batches = torch.randint(
        0, TRAIN_N, (steps, BATCH_SIZE),
        generator=batch_generator, device=CPU,
    )
    return train, heldout, batches


def _loss_and_accuracy(logits, targets):
    # Targets at positions 0 and 1 depend on an unseen initial token.
    chosen_logits = logits[:, 2:, :].reshape(-1, VOCAB_SIZE)
    chosen_targets = targets[:, 2:].reshape(-1)
    loss = F.cross_entropy(chosen_logits, chosen_targets)
    accuracy = (chosen_logits.argmax(dim=-1) == chosen_targets).float().mean()
    return loss, accuracy


def _evaluation_steps(steps):
    points = {0, steps}
    for boundary in range(25, steps + 1, 25):
        points.update((boundary - 1, boundary, boundary + 1))
    return sorted(step for step in points if 0 <= step <= steps)


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _git_info():
    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            capture_output=True, text=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "head": git("rev-parse", "HEAD"),
        "status": git("status", "--porcelain", "--untracked-files=normal"),
    }


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_hash(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        cpu = tensor.detach().to(CPU).contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(str(cpu.dtype).encode("ascii"))
        if cpu.dtype == torch.float32:
            fmt = "<f"
        elif cpu.dtype == torch.int64:
            fmt = "<q"
        else:
            raise TypeError("unsupported hash dtype: " + str(cpu.dtype))
        for value in cpu.reshape(-1).tolist():
            digest.update(struct.pack(fmt, value))
    return digest.hexdigest()


def _optimizer(model, mode, lr):
    if mode == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0), None

    groups = build_tagged_param_groups(model, base_lr=lr, base_wd=0.0)
    for group in groups:
        group["rms_clip_threshold"] = 1.0
        group["rms_clip_granularity"] = "param"
    qkv_groups = [group for group in groups if group.get("block_tag") == "attn_qkv"]
    if len(qkv_groups) != 1:
        raise RuntimeError("expected one fused QKV parameter group")
    qkv_group = qkv_groups[0]
    names = {id(param): name for name, param in model.named_parameters()}
    qkv_names = {names[id(param)] for param in qkv_group["params"]}
    if qkv_names != {"attn.in_proj_weight", "attn.in_proj_bias"}:
        raise RuntimeError("fused QKV weight and bias were not both tagged")
    if any(qkv_group["qkv_rules"].get(id(param)) != (0, 3)
           for param in qkv_group["params"]):
        raise RuntimeError("fused QKV parameters must use dim-0 three-way slicing")
    optimizer = Tiger(
        groups, lr=lr, weight_decay=0.0,
        trust_space="precond", trust_clip=5.0,
        update_buffer_dtype="fp32", agc_clip=0.0,
        auto_lr=False, auto_blend=False,
        qkv_lr_autoadapt=True,
        qkv_spectral_adapt=(mode == "tiger-full"),
        qkv_lr_interval=25,
    )
    active_qkv_group = next(
        group for group in optimizer.param_groups
        if group.get("block_tag") == "attn_qkv"
    )
    return optimizer, active_qkv_group


def _evaluate(model, pair, device):
    model.eval()
    with torch.no_grad():
        inputs, targets = (tensor.to(device) for tensor in pair)
        loss, accuracy = _loss_and_accuracy(model(inputs), targets)
        _sync(device)
        return {"ce": float(loss.item()), "accuracy": float(accuracy.item())}


def _run_seed(seed, steps, mode, lr, schedule, min_lr, device):
    train, heldout, batches = _corpus(seed, steps)
    corpus_hash = _tensor_hash((
        ("train_inputs", train[0]), ("train_targets", train[1]),
        ("heldout_inputs", heldout[0]), ("heldout_targets", heldout[1]),
    ))
    batch_hash = _tensor_hash((("batch_indices", batches),))
    torch.manual_seed(seed + 1000)
    model = ToyCausalTransformer()
    initial_hash = _tensor_hash(model.state_dict().items())
    model = model.to(device)
    optimizer, qkv_group = _optimizer(model, mode, lr)
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=min_lr,
        )
    elif schedule == "tail-cosine":
        decay_start = int(0.9 * steps)
        decay_length = max(1, steps - decay_start)
        minimum_factor = min_lr / lr

        def tail_factor(update):
            if update <= decay_start:
                return 1.0
            progress = min(1.0, (update - decay_start) / decay_length)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return minimum_factor + (1.0 - minimum_factor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=tail_factor,
        )
    else:
        scheduler = None
    qkv_weight = model.attn.in_proj_weight
    checkpoints = set(_evaluation_steps(steps))
    trace = []
    evaluations = []
    status = "complete"
    stopped_at = None

    for step in range(steps + 1):
        if step in checkpoints:
            train_eval = _evaluate(model, train, device)
            heldout_eval = _evaluate(model, heldout, device)
            if not all(math.isfinite(value) for value in (
                train_eval["ce"], heldout_eval["ce"]
            )):
                status, stopped_at = "nonfinite_evaluation", step
                break
            evaluations.append({"step": step, "train": train_eval,
                                "heldout": heldout_eval})
        if step == steps:
            break

        model.train()
        indices = batches[step]
        inputs = train[0][indices].to(device)
        targets = train[1][indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _loss_and_accuracy(model(inputs), targets)
        if not bool(torch.isfinite(loss).item()):
            status, stopped_at = "nonfinite_batch_loss", step + 1
            break
        loss.backward()
        _sync(device)
        grad_norm = None
        if (step + 1) % 25 == 0:
            if qkv_weight.grad is None:
                raise RuntimeError("fused QKV gradient is missing")
            grad_norm = float(qkv_weight.grad.norm().item())
            if not math.isfinite(grad_norm) or grad_norm <= 0:
                status, stopped_at = "invalid_qkv_gradient", step + 1
                break
        t0 = time.perf_counter()
        step_lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        _sync(device)
        optimizer_ms = 1000.0 * (time.perf_counter() - t0)
        if scheduler is not None:
            scheduler.step()
        qkv_scales = (dict(qkv_group["qkv_lr_scales"])
                      if qkv_group is not None else None)
        metrics = {}
        if qkv_group is not None and (step + 1) % 25 == 0:
            raw = optimizer.get_last_metrics()
            for key in ("qkv_gamma_eff", "qkv_freq_factor",
                        "qkv_phase_boost", "qkv_disp", "qkv_step_clip_eff"):
                value = raw.get(key)
                if value is None or not math.isfinite(float(value)):
                    raise RuntimeError("QKV adaptation metric missing or nonfinite: " + key)
                metrics[key] = float(value)
            if mode == "tiger-no-spectral" and (
                metrics["qkv_freq_factor"] != 1.0 or
                metrics["qkv_phase_boost"] != 1.0
            ):
                raise RuntimeError("spectral-off ablation applied a spectral factor")
        trace.append({
            "step": step + 1,
            "batch_ce": float(loss.item()),
            "lr": step_lr,
            "optimizer_ms": optimizer_ms,
            "qkv_weight_grad_l2": grad_norm,
            "qkv_scales": qkv_scales,
            "qkv_metrics": metrics,
        })

    return {
        "seed": seed,
        "status": status,
        "stopped_at_step": stopped_at,
        "evaluations": evaluations,
        "steps": trace,
        "initialization_seed": seed + 1000,
        "data_seed": seed,
        "batch_seed": seed + 9000,
        "initial_state_sha256": initial_hash,
        "corpus_sha256": corpus_hash,
        "minibatches_sha256": batch_hash,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tiger-full", "tiger-no-spectral", "adamw"),
                        required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--lr", type=float,
                        help="learning rate (default: Tiger 0.01, AdamW 0.003)")
    parser.add_argument("--schedule", choices=("constant", "cosine", "tail-cosine"),
                        default="constant")
    parser.add_argument("--min-lr", type=float, default=1e-4,
                        help="cosine final LR (default: 0.0001)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.steps < 25:
        parser.error("--steps must be at least 25 to exercise QKV adaptation")
    lr = args.lr if args.lr is not None else (3e-3 if args.mode == "adamw" else 1e-2)
    if not math.isfinite(lr) or lr <= 0:
        parser.error("--lr must be positive and finite")
    if args.schedule != "constant" and (
        not math.isfinite(args.min_lr) or args.min_lr < 0 or args.min_lr >= lr
    ):
        parser.error("--min-lr must be nonnegative, finite, and below --lr")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    if args.output.exists():
        parser.error("output already exists: " + str(args.output))
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    # Some local Python installations inject a different default device.
    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")
    if args.device == "cpu":
        torch.set_num_threads(1)
    device = torch.device(args.device)
    result = {
        "label": "finite held-out causal Transformer learning probe",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "git": _git_info(),
        "sha256": {
            "script": _sha256(Path(__file__)),
            "tiger_source": _sha256(PROJECT_ROOT / "src/tiger_optim/tiger.py"),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": args.device,
            "cpu_threads": torch.get_num_threads(),
        },
        "task": {
            "name": "disjoint-triple period-3 causal next-token prediction",
            "train_sequences": TRAIN_N,
            "heldout_sequences": HELDOUT_N,
            "vocab_size": VOCAB_SIZE,
            "sequence_length": SEQ_LEN,
            "evaluated_positions": list(range(2, SEQ_LEN)),
            "model": "one-layer 48-wide 4-head causal MHA, 96-wide FFN",
            "batch_size": BATCH_SIZE,
            "steps": args.steps,
            "evaluation_steps": _evaluation_steps(args.steps),
            "minibatches": "same precomputed CPU indices for every mode per seed",
            "chance_ce": math.log(VOCAB_SIZE),
        },
        "optimizer": {
            "lr": lr,
            "weight_decay": 0.0,
            "schedule": args.schedule,
            "cosine_min_lr": args.min_lr if args.schedule != "constant" else None,
            "tail_decay_starts_after_fraction": 0.9 if args.schedule == "tail-cosine" else None,
            "scheduler_order": "scheduler.step() after optimizer.step()",
            "tiger_auto_lr": False,
            "tiger_auto_blend": False,
            "tiger_recipe": "tagged groups, param RMS clip 1, preconditioned trust clip 5, FP32 updates",
            "tiger_qkv_lr_interval": 25,
            "tiger_spectral_adapt": args.mode == "tiger-full",
            "adamw_role": "context only; Tiger has tagged per-parameter LR scales",
        },
        "limitations": [
            "Synthetic period-3 copy task; no language-model quality claim.",
            "AdamW is contextual because Tiger uses tagged LR scales.",
            "Single-process per mode timings are diagnostic, not throughput proof.",
        ],
        "runs": [],
    }
    for seed in args.seeds:
        result["runs"].append(_run_seed(
            seed, args.steps, args.mode, lr, args.schedule, args.min_lr, device
        ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "mode": args.mode,
        "summaries": [
            {"seed": run["seed"], "status": run["status"],
             "initial_heldout_ce": run["evaluations"][0]["heldout"]["ce"],
             "final_heldout_ce": run["evaluations"][-1]["heldout"]["ce"],
             "final_heldout_accuracy": run["evaluations"][-1]["heldout"]["accuracy"]}
            for run in result["runs"]
        ],
    }, indent=2))
    return 0 if all(run["status"] == "complete" for run in result["runs"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
