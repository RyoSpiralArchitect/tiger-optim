import os

import pytest
import torch

from tiger_optim import Tiger, build_tagged_param_groups


class _TinyMix(torch.nn.Module):
    def __init__(self, d=64, ff=128, heads=4):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(d, heads, batch_first=True)
        self.up = torch.nn.Linear(d, ff)
        self.down = torch.nn.Linear(ff, d)
        self.ln = torch.nn.LayerNorm(d)

    def forward(self, x):
        h, _ = self.mha(x, x, x)
        h = torch.nn.functional.gelu(self.up(h))
        return self.ln(self.down(h))


def _run(cap, device, steps=12):
    torch.manual_seed(0)
    model = _TinyMix().to(device)
    groups = build_tagged_param_groups(
        model, base_lr=3e-4, base_wd=0.01, enable_qkv_slicing=True)
    if cap is not None:
        os.environ["TIGER_FLAT_CACHE_MAX"] = str(cap)
    else:
        os.environ.pop("TIGER_FLAT_CACHE_MAX", None)
    opt = Tiger(groups, factored=True, precond_alpha=1.0, trust_space="precond",
                trust_ema_beta=0.9, use_foreach_update=True,
                bucket_standardize=True, bucket_scalarless=True,
                qkv_lr_autoadapt=True, lora_density_adapt=True,
                profiler_enabled=True, profiler_interval=2,
                profiler_path="/tmp/tiger_churn_test.jsonl")
    x = torch.randn(4, 8, 64, device=device)
    y = torch.randn(4, 8, 64, device=device)
    lf = torch.nn.MSELoss()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        lf(model(x), y).backward()
        opt.step()
    params = [p.detach().clone() for p in model.parameters()]
    moments = [{k: v.detach().clone() for k, v in opt.state[p].items()
                if isinstance(v, torch.Tensor)}
               for p in model.parameters()]
    sizes = {gi: len(c) for gi, c in opt._flat_cache.items()}
    return params, moments, sizes, opt._flat_cache_max


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flat_cache_lru_churn_bitwise(monkeypatch):
    pytest.importorskip("triton")
    monkeypatch.setenv("TIGER_FLAT_STEP", "1")
    try:
        ref_p, ref_m, ref_sizes, _ = _run(None, torch.device("cuda"))
        cap_p, cap_m, cap_sizes, cap_max = _run(1, torch.device("cuda"))
    finally:
        os.environ.pop("TIGER_FLAT_CACHE_MAX", None)
    assert cap_max == 1
    assert all(n <= 1 for n in cap_sizes.values())
    for a, b in zip(ref_p, cap_p):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(ref_m) == len(cap_m)
    for ma, mb in zip(ref_m, cap_m):
        assert ma.keys() == mb.keys()
        for k in ma:
            torch.testing.assert_close(ma[k], mb[k], rtol=0, atol=0)
