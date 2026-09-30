"""Group-level LR controls compose with the fused Q/K/V multipliers."""
import pytest
import torch

from tiger_optim import Tiger


def _optimizer(param, *, lr=0.1, group_scale=1.0, slices=None, trust=False, wd=0.0):
    return Tiger(
        [{"params": [param], "lr_scale": group_scale, "block_tag": "attn_qkv",
          "qkv_rules": {id(param): (0, 3)}, "qkv_trust_split": True,
          "qkv_lr_scales": slices if slices is not None else {"q": 0.5, "k": 1.0, "v": 1.5}}],
        lr=lr, weight_decay=wd, betas=(0.0, 0.0), factored=False,
        precond_alpha=0.0, sign_mode="sign", sign_blend=0.0,
        use_trust_ratio=trust, trust_space="precond", trust_clip=5.0,
        use_foreach=False, use_foreach_update=False, auto_lr=False,
        auto_blend=False, qkv_lr_autoadapt=False, qkv_spectral_adapt=False,
        lora_cross_adapt=False,
    )


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("trust", [False, True])
def test_group_scale_matches_scaled_base_lr(device, dtype, trust):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    scaled = torch.nn.Parameter(torch.ones((3, 8), device=device, dtype=dtype))
    reference = torch.nn.Parameter(scaled.detach().clone())
    opt = _optimizer(scaled, group_scale=0.25, trust=trust, wd=0.1)
    ref = _optimizer(reference, lr=0.025, trust=trust, wd=0.1)
    for value in (1.0, 2.0, 0.5):
        scaled.grad = torch.full_like(scaled, value)
        reference.grad = torch.full_like(reference, value)
        opt.step()
        ref.step()
        torch.testing.assert_close(scaled, reference, rtol=0, atol=0)


def test_staged_zero_group_scale_stops_all_qkv_updates():
    param = torch.nn.Parameter(torch.ones((3, 4), device="cpu"))
    opt = _optimizer(param, wd=0.1)
    opt.stage_group_update(0, {"lr_scale": 0.0})
    opt.reflect_pending()
    param.grad = torch.ones_like(param)
    opt.step()
    torch.testing.assert_close(param, torch.ones_like(param), rtol=0, atol=0)


def test_missing_slice_scale_uses_neutral_multiplier():
    param = torch.nn.Parameter(torch.ones((3, 4), device="cpu"))
    opt = _optimizer(param, group_scale=0.5, slices={"q": 0.5})
    param.grad = torch.ones_like(param)
    opt.step()
    torch.testing.assert_close(
        param[:, 0], torch.tensor([0.975, 0.95, 0.95], device="cpu")
    )


@pytest.mark.parametrize(
    "dtype,lr,group_scale,slice_scale",
    [(torch.float16, 1e3, 1e2, 1.0),
     (torch.bfloat16, 1e20, 1e10, 1e10),
     (torch.float32, 1e20, 1e10, 1e10)],
)
def test_composed_scale_overflow_skips_before_moment_commit(dtype, lr, group_scale, slice_scale):
    param = torch.nn.Parameter(torch.ones((3, 4), device="cpu", dtype=dtype))
    opt = _optimizer(param, lr=lr, group_scale=group_scale,
                     slices={key: slice_scale for key in ("q", "k", "v")})
    param.grad = torch.ones_like(param)
    opt.step()
    torch.testing.assert_close(param, torch.ones_like(param), rtol=0, atol=0)
    assert not opt.state.get(param)


def test_small_group_scale_allows_finite_fp16_update():
    param = torch.nn.Parameter(torch.ones((3, 4), device="cpu", dtype=torch.float16))
    opt = _optimizer(param, lr=1e5, group_scale=0.01,
                     slices={key: 1.0 for key in ("q", "k", "v")})
    param.grad = torch.ones_like(param)
    opt.step()
    torch.testing.assert_close(param, torch.full_like(param, -999), rtol=0, atol=0)
