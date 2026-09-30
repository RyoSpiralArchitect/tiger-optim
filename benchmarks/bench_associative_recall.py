#!/usr/bin/env python3
"""Train a causal Transformer on fresh multi-query associative recall.

The context contains unique keys paired with random values, followed by a gap
and multiple queries. Only queried values are supervised; answers never appear
in the query suffix. This is a controlled retrieval task, not language modeling.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tiger_optim import Tiger, build_tagged_param_groups

MODES = ("adamw", "tiger-full", "tiger-no-spectral", "tiger-fixed-qkv")


def corpus(seed, count, *, symbols, pairs, gap, queries):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    keys = torch.rand(count, symbols, generator=rng, device="cpu").argsort(dim=1)[:, :pairs]
    values = torch.randint(symbols, (count, pairs), generator=rng, device="cpu")
    selected = torch.randint(pairs, (count, queries), generator=rng, device="cpu")
    length = 2 * pairs + gap + 2 * queries
    tokens = torch.full((count, length), 2 * symbols, dtype=torch.long, device="cpu")
    tokens[:, :2 * pairs:2] = keys
    tokens[:, 1:2 * pairs:2] = values + symbols
    tokens[:, 2 * pairs + gap::2] = 2 * symbols + 1
    tokens[:, 2 * pairs + gap + 1::2] = keys.gather(1, selected)
    return tokens, values.gather(1, selected)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.attn_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn_up = nn.Linear(width, 4 * width, bias=False)
        self.ffn_down = nn.Linear(4 * width, width, bias=False)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(
            batch, length, 3, self.heads, width // self.heads,
        ).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out_proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.ffn_down(F.gelu(self.ffn_up(self.ffn_norm(x))))


class RecallTransformer(nn.Module):
    def __init__(self, *, symbols, pairs, gap, queries, width, layers, heads):
        super().__init__()
        self.query_start = 2 * pairs + gap + 1
        self.token = nn.Embedding(2 * symbols + 2, width)
        self.position = nn.Embedding(2 * pairs + gap + 2 * queries, width)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, symbols, bias=False)

    def forward(self, tokens):
        x = self.token(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x[:, self.query_start::2]))


def tensor_hash(items):
    digest = hashlib.sha256()
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((name, tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def optimizer_for(model, mode, lr):
    if mode == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    groups = build_tagged_param_groups(model, base_lr=lr, base_wd=0.0)
    for group in groups:
        group["rms_clip_threshold"] = 1.0
        group["auto_ffn_asym"] = False
    qkv = next(group for group in groups if group["block_tag"] == "attn_qkv")
    expected = {id(block.qkv.weight) for block in model.blocks}
    if set(qkv["qkv_rules"]) != expected or set(map(id, qkv["params"])) != expected:
        raise RuntimeError("every block must have exactly one tagged fused QKV weight")
    return Tiger(
        groups, lr=lr, weight_decay=0.0, trust_space="precond", trust_clip=5.0,
        update_buffer_dtype="fp32", auto_lr=False, auto_blend=False,
        lora_cross_adapt=False, qkv_lr_autoadapt=mode != "tiger-fixed-qkv",
        qkv_spectral_adapt=mode == "tiger-full", qkv_lr_interval=25,
    )


def lr_factor(completed, total):
    warmup = max(1, total // 20)
    if completed < warmup:
        return (completed + 1) / warmup
    progress = min(1.0, (completed - warmup) / max(1, total - warmup))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate(model, data, batch_size, autocast):
    model.eval()
    loss_sum = correct = count = 0
    with torch.no_grad():
        for start in range(0, len(data[0]), batch_size):
            tokens, targets = (part[start:start + batch_size] for part in data)
            with autocast():
                logits = model(tokens)
            logits = logits.float()
            loss_sum += F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum").item()
            correct += (logits.argmax(-1) == targets).sum().item()
            count += targets.numel()
    return {"ce": loss_sum / count, "accuracy": correct / count, "queries": count}


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def run(args):
    torch.set_default_device("cpu")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if args.precision == "bf16" and device.type != "cuda":
        raise ValueError("BF16 autocast for this benchmark requires CUDA")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.precision == "bf16" else nullcontext
    data_config = {key: getattr(args, key) for key in ("symbols", "pairs", "gap", "queries")}
    model_config = dict(data_config, width=args.width, layers=args.layers, heads=args.heads)
    datasets = {
        "train": corpus(args.seed + 10000, args.steps * args.batch_size, **data_config),
        "validation": corpus(args.seed + 20000, args.eval_size, **data_config),
    }
    if not args.development:
        datasets["test"] = corpus(args.seed + 30000, args.test_size, **data_config)
    hashes = {name: tensor_hash(zip(("tokens", "targets"), data)) for name, data in datasets.items()}
    torch.manual_seed(args.seed)
    model = RecallTransformer(**model_config)
    initial_hash = tensor_hash(model.state_dict().items())
    model.to(device)
    datasets = {name: tuple(part.to(device) for part in data) for name, data in datasets.items()}
    optimizer = optimizer_for(model, args.mode, args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, args.steps))
    qkv = next((g for g in optimizer.param_groups if g.get("block_tag") == "attn_qkv"), None)
    source_paths = [Path(__file__), *sorted((ROOT / "src/tiger_optim").rglob("*.py"))]
    result = {
        "schema": 1, "status": "running", "config": {k: v.name if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": model_config, "parameter_count": sum(p.numel() for p in model.parameters()),
        "git": {"head": git("rev-parse", "HEAD"), "status": git("status", "--porcelain")},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                        "precision": args.precision, "cpu_threads": args.cpu_threads, "tf32": False},
        "initial_state_sha256": initial_hash, "data_sha256": hashes,
        "evaluations": [], "steps": [],
    }

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    train_seconds = 0.0
    try:
        for completed in range(args.steps + 1):
            if completed % args.eval_interval == 0 or completed == args.steps:
                score = evaluate(model, datasets["validation"], args.batch_size, autocast)
                if not math.isfinite(score["ce"]):
                    raise FloatingPointError("nonfinite validation loss")
                scales = dict(qkv["qkv_lr_scales"]) if qkv is not None else None
                result["evaluations"].append({"step": completed, "validation": score, "qkv_scales": scales})
                save()
                print(json.dumps({"mode": args.mode, "seed": args.seed, "step": completed, **score}), flush=True)
            if completed == args.steps:
                break
            model.train()
            start = completed * args.batch_size
            tokens, targets = (part[start:start + args.batch_size] for part in datasets["train"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(tokens)
                loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if not math.isfinite(loss.item()):
                raise FloatingPointError("nonfinite training loss")
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            train_seconds += elapsed
            result["steps"].append({"step": completed + 1, "loss": loss.item(), "lr": lr,
                                    "grad_norm": grad_norm.item(), "seconds": elapsed})
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise FloatingPointError("nonfinite model parameters")
        if "test" in datasets:
            result["test"] = evaluate(model, datasets["test"], args.batch_size, autocast)
            if not math.isfinite(result["test"]["ce"]):
                raise FloatingPointError("nonfinite test loss")
        result["final_state_sha256"] = tensor_hash(model.state_dict().items())
        result["train_seconds"] = train_seconds
        result["peak_cuda_bytes"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "config": result["config"]}, args.checkpoint)
        result["status"] = "complete"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = type(exc).__name__ + ": " + str(exc)
        save()
        raise
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    for key, default in (("steps", 1000), ("batch-size", 64), ("symbols", 64), ("pairs", 16),
                         ("gap", 64), ("queries", 4), ("width", 192), ("layers", 4), ("heads", 6),
                         ("eval-size", 512), ("test-size", 2048), ("eval-interval", 100), ("cpu-threads", 2)):
        parser.add_argument("--" + key, type=int, default=default)
    parser.add_argument("--development", action="store_true", help="evaluate validation only; never create test data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    for key in ("steps", "batch_size", "symbols", "pairs", "queries", "width", "layers", "heads", "eval_size", "test_size", "eval_interval", "cpu_threads"):
        if getattr(args, key) < 1:
            parser.error(key + " must be positive")
    if args.gap < 0 or args.pairs > args.symbols or args.width % args.heads:
        parser.error("require gap >= 0, pairs <= symbols, and width divisible by heads")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be positive and finite")
    if args.output.exists() or (args.checkpoint and args.checkpoint.exists()):
        parser.error("refusing to overwrite an existing result/checkpoint")
    run(args)


if __name__ == "__main__":
    main()
