
import argparse, os, json, time, statistics
from datetime import datetime
import torch
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

def main():
    ap = argparse.ArgumentParser("Tiger v2.1 micro bench")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--triton-stats", action="store_true")
    ap.add_argument("--triton-fused", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.device(args.device).type in {"cpu","cuda","mps"} else "cpu")
    torch.manual_seed(0)
    m = TinyMix().to(dev)
    x = torch.randn(16, 32, 256, device=dev)
    y = torch.randn(16, 32, 256, device=dev)

    groups = build_tagged_param_groups(m, base_lr=3e-4, base_wd=0.01, enable_qkv_slicing=True,
        tag_overrides={"mlp": {"block_trust_chunks": 4, "foreach_min_bucket": 4,
                               "bucket_standardize": True, "bucket_standardize_source": "global",
                               "bucket_scalarless": True, "use_triton_bucket_stats": args.triton_stats, "use_triton_fused_apply": args.triton_fused}}
    )

    opt = Tiger(groups, factored=True, precond_alpha=1.0, trust_space="precond", trust_ema_beta=0.9,
                mom_dtype="bf16", agc_clip=0.02,
                lora_density_adapt=True, lora_density_beta=0.9, lora_pid_mode="auto",
                lora_ki_min=0.01, lora_kd_min=0.005, lora_recover_rate=0.15,
                use_foreach_update=True, update_buffer_dtype="bf16", bucket_standardize=True, bucket_standardize_source="global", bucket_scalarless=True,
                qkv_lr_autoadapt=True, qkv_w_rms=0.7, qkv_w_trust=0.3, qkv_lr_gain=0.02, qkv_gain_shrink_gamma=1.2,
                qkv_disp_ema_beta=0.8, qkv_gamma_rate=0.6, qkv_lr_step_clip=0.08, qkv_clip_shrink_k=0.6, qkv_clip_min=0.02, qkv_clip_max=0.2,
                profiler_enabled=True, profiler_path="benchmarks/profiles/v21.jsonl", profiler_interval=5)

    loss_fn = torch.nn.MSELoss()

    for _ in range(args.warmup):
        opt.zero_grad(set_to_none=True); o=m(x); l=loss_fn(o,y); l.backward(); sync(dev); opt.report_metrics(float(l)); opt.step()

    ms = []; metrics = []
    for _ in range(args.steps):
        t0=time.perf_counter()
        opt.zero_grad(set_to_none=True); o=m(x); l=loss_fn(o,y); l.backward(); sync(dev); opt.report_metrics(float(l)); opt.step(); sync(dev)
        ms.append(1000.0*(time.perf_counter()-t0))
        metrics.append(opt.get_last_metrics())

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join("benchmarks", "results"); os.makedirs(out, exist_ok=True)
    p = os.path.join(out, f"profile_v21-{dev.type}-{ts}.json")
    with open(p,"w") as f: json.dump({"ms_median": statistics.median(ms), "device": dev.type, "last_metrics": metrics[-1] if metrics else {}}, f, indent=2)
    print("Saved", p)
    print("JSONL:", "benchmarks/profiles/v21.jsonl")

if __name__ == "__main__":
    main()
