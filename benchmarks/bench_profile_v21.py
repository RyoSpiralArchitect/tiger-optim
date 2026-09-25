#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime

import torch

from tiger_optim import Tiger, build_tagged_param_groups


class TinyMix(torch.nn.Module):
    def __init__(self, d: int = 256, ff: int = 512, heads: int = 4):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(d, heads, batch_first=True)
        self.up = torch.nn.Linear(d, ff)
        self.down = torch.nn.Linear(ff, d)
        self.ln = torch.nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.mha(x, x, x)
        h = torch.nn.functional.gelu(self.up(h))
        h = self.down(h)
        return self.ln(h)


def sync(dev: torch.device) -> None:
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif dev.type == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> Tiger:
    device_type = next(model.parameters()).device.type
    momentum_dtype = "fp32" if device_type == "mps" else "bf16"
    update_buffer_dtype = "fp32" if device_type == "mps" else "bf16"
    groups = build_tagged_param_groups(
        model,
        base_lr=3e-4,
        base_wd=0.01,
        enable_qkv_slicing=True,
        tag_overrides={
            "mlp": {
                "block_trust_chunks": 4,
                "foreach_min_bucket": 4,
                "bucket_standardize": True,
                "bucket_standardize_source": "global",
                "bucket_scalarless": True,
                "use_triton_bucket_stats": args.triton_stats,
                "use_triton_fused_apply": args.triton_fused,
            }
        },
    )

    return Tiger(
        groups,
        factored=True,
        precond_alpha=1.0,
        trust_space="precond",
        trust_ema_beta=0.9,
        mom_dtype=momentum_dtype,
        agc_clip=0.02,
        lora_density_adapt=True,
        lora_density_beta=0.9,
        lora_pid_mode="auto",
        lora_ki_min=0.01,
        lora_kd_min=0.005,
        lora_recover_rate=0.15,
        use_foreach_update=True,
        update_buffer_dtype=update_buffer_dtype,
        bucket_standardize=True,
        bucket_standardize_source="global",
        bucket_scalarless=True,
        qkv_lr_autoadapt=True,
        qkv_w_rms=0.7,
        qkv_w_trust=0.3,
        qkv_lr_gain=0.02,
        qkv_gain_shrink_gamma=1.2,
        qkv_disp_ema_beta=0.8,
        qkv_gamma_rate=0.6,
        qkv_lr_step_clip=0.08,
        qkv_clip_shrink_k=0.6,
        qkv_clip_min=0.02,
        qkv_clip_max=0.2,
        profiler_enabled=True,
        profiler_path=args.profiler_jsonl,
        profiler_interval=5,
    )


def run_iteration(
    model: torch.nn.Module,
    opt: Tiger,
    loss_fn: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    dev: torch.device,
) -> float:
    opt.zero_grad(set_to_none=True)
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    sync(dev)
    opt.report_metrics(loss.detach().item())
    opt.step()
    sync(dev)
    return loss.detach().item()


def maybe_profile_opt_step(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    opt: Tiger,
    loss_fn: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    dev: torch.device,
) -> dict:
    if not args.torch_profiler:
        return {}

    activities = [torch.profiler.ProfilerActivity.CPU]
    profile_rows = max(1, int(args.profile_steps))
    opt_step_ms = []

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(profile_rows):
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            sync(dev)
            opt.report_metrics(loss.detach().item())
            t0 = time.perf_counter()
            with torch.autograd.profiler.record_function("bench/opt_step"):
                opt.step()
            sync(dev)
            opt_step_ms.append(1000.0 * (time.perf_counter() - t0))

    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=max(1, int(args.topk)))
    print(table)

    trace_out = args.trace_out
    if trace_out:
        os.makedirs(os.path.dirname(trace_out), exist_ok=True)
        prof.export_chrome_trace(trace_out)

    table_out = args.table_out
    if table_out:
        os.makedirs(os.path.dirname(table_out), exist_ok=True)
        with open(table_out, "w", encoding="utf-8") as fh:
            fh.write(table)
            fh.write("\n")

    return {
        "profile_steps": profile_rows,
        "opt_step_ms_median": statistics.median(opt_step_ms) if opt_step_ms else None,
        "opt_step_ms_mean": (sum(opt_step_ms) / len(opt_step_ms)) if opt_step_ms else None,
        "trace_out": trace_out,
        "table_out": table_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser("Tiger v2.1 micro bench + opt.step profiler")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--triton-stats", action="store_true")
    parser.add_argument("--triton-fused", action="store_true")
    parser.add_argument("--torch-profiler", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=4)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--trace-out")
    parser.add_argument("--table-out")
    parser.add_argument("--profiler-jsonl", default="benchmarks/profiles/v21.jsonl")
    args = parser.parse_args()

    dev = torch.device(args.device if torch.device(args.device).type in {"cpu", "cuda", "mps"} else "cpu")
    torch.manual_seed(0)
    model = TinyMix().to(dev)
    x = torch.randn(16, 32, 256, device=dev)
    y = torch.randn(16, 32, 256, device=dev)
    loss_fn = torch.nn.MSELoss()
    opt = build_optimizer(args, model)

    for _ in range(args.warmup):
        run_iteration(model, opt, loss_fn, x, y, dev)

    ms = []
    metrics = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        run_iteration(model, opt, loss_fn, x, y, dev)
        ms.append(1000.0 * (time.perf_counter() - t0))
        metrics.append(opt.get_last_metrics())

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not args.trace_out:
        args.trace_out = os.path.join("benchmarks", "profiles", f"torch-profiler-{dev.type}-{ts}.json")
    if not args.table_out:
        args.table_out = os.path.join("benchmarks", "profiles", f"torch-profiler-{dev.type}-{ts}.txt")
    profiler_summary = maybe_profile_opt_step(
        args=args,
        model=model,
        opt=opt,
        loss_fn=loss_fn,
        x=x,
        y=y,
        dev=dev,
    )

    out_dir = os.path.join("benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"profile_v21-{dev.type}-{ts}.json")
    payload = {
        "ms_median": statistics.median(ms),
        "device": dev.type,
        "last_metrics": metrics[-1] if metrics else {},
        "profiler": profiler_summary,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("Saved", out_path)
    print("JSONL:", args.profiler_jsonl)
    if profiler_summary:
        print("Torch profiler trace:", profiler_summary.get("trace_out"))
        print("Torch profiler table:", profiler_summary.get("table_out"))


if __name__ == "__main__":
    main()
