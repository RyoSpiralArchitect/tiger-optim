#!/usr/bin/env python3
import argparse, os, json, time, statistics
from datetime import datetime
import torch

# Minimal local import (assumes 'pip install -e .' or PYTHONPATH set to src/)
from tiger_optim import Tiger, build_tagged_param_groups

class TinyMix(torch.nn.Module):
    def __init__(self, d=256, ff=512, heads=4):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(d, heads, batch_first=True)
        self.up = torch.nn.Linear(d, ff)
        self.down = torch.nn.Linear(ff, d)
        self.ln = torch.nn.LayerNorm(d)
    def forward(self, x):
        h,_ = self.mha(x,x,x)
        h = torch.nn.functional.gelu(self.up(h))
        h = self.down(h)
        return self.ln(h)

def sync(dev):
    if dev.type=="cuda" and torch.cuda.is_available(): torch.cuda.synchronize()
    elif dev.type=="mps" and hasattr(torch,"mps") and torch.backends.mps.is_available():
        try: torch.mps.synchronize()
        except: pass

def build_optimizer(model, mode: str, device: torch.device):
    mode = mode.lower()
    if mode == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.999), weight_decay=0.01)
    # Tiger modes
    groups = build_tagged_param_groups(model, base_lr=3e-4, base_wd=0.01, enable_qkv_slicing=True)
    common = dict(factored=True, precond_alpha=1.0, trust_space="precond", trust_ema_beta=0.9,
                  use_foreach_update=True, bucket_standardize=True, bucket_scalarless=True,
                  profiler_enabled=False)
    if mode == "tiger_v21_full":
        return Tiger(groups, **common,
            lora_density_adapt=True, lora_pid_mode="auto",
            qkv_lr_autoadapt=True, qkv_w_rms=0.7, qkv_w_trust=0.3,
            qkv_lr_gain=0.02, qkv_gain_shrink_gamma=1.2, qkv_disp_ema_beta=0.8, qkv_gamma_rate=0.6)
    if mode == "tiger_v21_no_qkv":
        for g in groups:
            if g.get("block_tag")=="attn_qkv":
                g["qkv_lr_scales"] = None; g["qkv_trust_split"] = False
        return Tiger(groups, **common, qkv_lr_autoadapt=False)
    if mode == "tiger_v21_no_ffn":
        for g in groups:
            if g.get("block_tag") in ("ffn_up","ffn_down"):
                g["auto_ffn_asym"] = False
        return Tiger(groups, **common, qkv_lr_autoadapt=True)
    if mode == "tiger_v13_compat":
        # Approximate older Tiger by disabling new features and inertia
        for g in groups:
            g["auto_ffn_asym"] = False
            if g.get("block_tag")=="attn_qkv":
                g["qkv_lr_scales"] = None; g["qkv_trust_split"] = False
        return Tiger(groups, **common, lora_density_adapt=False, qkv_lr_autoadapt=False, trust_ema_beta=0.0)
    raise ValueError(f"Unknown mode: {mode}")

def run(mode: str, device_str: str, steps=100, warmup=20, seed=0, batch=16, T=32, D=256):
    dev = torch.device(device_str if torch.device(device_str).type in {"cpu","cuda","mps"} else "cpu")
    torch.manual_seed(seed)
    model = TinyMix(d=D, ff=2*D, heads=4).to(dev)
    x = torch.randn(batch, T, D, device=dev)
    y = torch.randn(batch, T, D, device=dev)
    loss_fn = torch.nn.MSELoss()

    opt = build_optimizer(model, mode, dev)

    # warmup
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        out = model(x); loss = loss_fn(out, y)
        loss.backward(); sync(dev); 
        if hasattr(opt, "report_metrics"): opt.report_metrics(float(loss))
        opt.step(); sync(dev)

    ms, losses = [], []
    for _ in range(steps):
        t0=time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model(x); loss = loss_fn(out, y)
        loss.backward(); sync(dev); 
        if hasattr(opt, "report_metrics"): opt.report_metrics(float(loss))
        opt.step(); sync(dev)
        ms.append(1000.0*(time.perf_counter()-t0))
        losses.append(float(loss))

    summary = dict(mode=mode, device=dev.type, steps=steps, warmup=warmup,
                   ms_median=statistics.median(ms), ms_mean=sum(ms)/len(ms),
                   loss_last=losses[-1], loss_min=min(losses), loss_mean=sum(losses)/len(losses))

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("benchmarks", "results"); os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"compare-{mode}-{dev.type}-{ts}.json")
    with open(out_path, "w") as f: json.dump(dict(summary=summary, series=dict(ms=ms, loss=losses)), f, indent=2)
    print(json.dumps(summary, indent=2))
    print("Saved:", out_path)
    return out_path

def main():
    ap = argparse.ArgumentParser("AdamW vs Tiger compare")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--modes", nargs="+", default=["adamw","tiger_v13_compat","tiger_v21_no_qkv","tiger_v21_no_ffn","tiger_v21_full"])
    args = ap.parse_args()

    for m in args.modes:
        run(m, args.device, steps=args.steps, warmup=args.warmup)

if __name__ == "__main__":
    main()
