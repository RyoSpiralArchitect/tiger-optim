"""Check spectral attenuation, its zero endpoint, and checkpoint continuation."""
from copy import deepcopy
import importlib

import pytest
import torch

from tiger_optim import Tiger

tiger_module = importlib.import_module("tiger_optim.tiger")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def build(weight, **options):
    param = torch.nn.Parameter(weight.clone())
    group = {"params": [param], "block_tag": "attn_qkv", "qkv_rules": {id(param): (0, 3)},
             "qkv_trust_split": True, "qkv_lr_scales": {"q": 0.9, "k": 0.8, "v": 1.1}}
    optimizer = Tiger([group], lr=0.003, qkv_lr_interval=1,
                      auto_lr=False, auto_blend=False, lora_cross_adapt=False, **options)
    return param, optimizer


@pytest.mark.parametrize("device", DEVICES)
def test_zero_strength_matches_disabled_feedback_and_skips_fft(device, monkeypatch):
    torch.manual_seed(17)
    weight = torch.randn(12, 4, device=device)
    zero_param, zero = build(weight, qkv_spectral_strength=0.0)
    off_param, off = build(weight, qkv_spectral_adapt=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("spectral collection must be skipped at zero strength")

    monkeypatch.setattr(tiger_module, "_spectral_dispersion_chunks", forbidden)
    for _ in range(4):
        grad = torch.randn_like(weight)
        zero_param.grad = grad.clone()
        off_param.grad = grad.clone()
        zero.step()
        off.step()
        torch.testing.assert_close(zero_param, off_param, rtol=0, atol=0)
        assert zero.param_groups[0]["qkv_lr_scales"] == off.param_groups[0]["qkv_lr_scales"]
    assert zero._qkv_spec_ema == {} and zero._qkv_lr_ema
    assert zero._last_metrics["qkv_freq_factor"] == zero._last_metrics["qkv_phase_boost"] == 1.0


def test_half_strength_blends_clipped_corrections_toward_one():
    torch.manual_seed(23)
    weight = torch.randn(12, 4, device="cpu")
    grad = torch.randn_like(weight)
    settings = {"qkv_lr_gain": 10.0, "qkv_gamma_spectral_clip": (0.9, 0.9),
                "qkv_phase_boost_clip": (1.4, 1.4)}
    full_param, full = build(weight, **settings)
    half_param, half = build(weight, qkv_spectral_strength=0.5, **settings)
    full_param.grad = grad.clone()
    half_param.grad = grad.clone()
    full.step()
    half.step()
    for key in ("qkv_freq_factor", "qkv_phase_boost"):
        assert half._last_metrics[key] == pytest.approx(1 + 0.5 * (full._last_metrics[key] - 1))
    assert full._qkv_spec_ema.keys() == half._qkv_spec_ema.keys()
    assert half._last_metrics["qkv_freq_factor"] == pytest.approx(0.95)
    assert half._last_metrics["qkv_phase_boost"] == pytest.approx(1.2)
    assert full.param_groups[0]["qkv_lr_scales"] != half.param_groups[0]["qkv_lr_scales"]
    # Adaptation is applied after the step; its scales affect the next update.
    full_param.grad = grad.clone()
    half_param.grad = grad.clone()
    full.step()
    half.step()
    assert not torch.equal(full_param, half_param)


@pytest.mark.parametrize("device", DEVICES)
def test_strength_checkpoint_resumes_after_pause_and_reactivation(device):
    torch.manual_seed(29)
    weight = torch.randn(12, 4, device=device)
    param, optimizer = build(weight)
    param.grad = torch.randn_like(weight)
    optimizer.step()
    history = deepcopy(optimizer._qkv_spec_ema)
    optimizer.set_qkv_spectral_strength(0.0)
    param.grad = torch.randn_like(weight)
    optimizer.step()
    for label in history[0]:
        for key in history[0][label]:
            torch.testing.assert_close(optimizer._qkv_spec_ema[0][label][key], history[0][label][key])
    restored_param, restored = build(param.detach())
    restored.load_state_dict(deepcopy(optimizer.state_dict()))
    assert restored.qkv_spectral_strength == 0.0
    for strength in (0.25, 0.5, 1.0):
        optimizer.set_qkv_spectral_strength(strength)
        restored.set_qkv_spectral_strength(strength)
        grad = torch.randn_like(weight)
        param.grad = grad.clone()
        restored_param.grad = grad.clone()
        optimizer.step()
        restored.step()
        torch.testing.assert_close(restored_param, param, rtol=0, atol=0)
        assert restored.param_groups[0]["qkv_lr_scales"] == optimizer.param_groups[0]["qkv_lr_scales"]


def test_checkpoint_without_strength_retains_full_feedback():
    torch.manual_seed(31)
    param, optimizer = build(torch.randn(12, 4, device="cpu"))
    saved = deepcopy(optimizer.state_dict())
    saved["tiger_state"]["defaults"].pop("qkv_spectral_strength")
    restored_param, restored = build(param.detach(), qkv_spectral_strength=0.0)
    restored.load_state_dict(saved)
    assert restored.qkv_spectral_strength == 1.0
    grad = torch.randn_like(param)
    param.grad = grad.clone()
    restored_param.grad = grad.clone()
    optimizer.step()
    restored.step()
    torch.testing.assert_close(restored_param, param, rtol=0, atol=0)
    assert restored.param_groups[0]["qkv_lr_scales"] == optimizer.param_groups[0]["qkv_lr_scales"]


def test_full_strength_keeps_extreme_custom_corrections_without_cancellation():
    param, optimizer = build(torch.ones(12, 4, device="cpu"),
                             qkv_gamma_spectral_clip=(1e-12, 1e-12),
                             qkv_phase_boost_clip=(1e-12, 1e-12))
    param.grad = torch.arange(48, device="cpu", dtype=torch.float32).reshape(12, 4) + 1
    optimizer.step()
    assert optimizer._last_metrics["qkv_freq_factor"] == 1e-12
    assert optimizer._last_metrics["qkv_phase_boost"] == 1e-12


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_strength_is_rejected_without_changing_control(invalid):
    weight = torch.ones(12, 4, device="cpu")
    with pytest.raises(ValueError, match="qkv_spectral_strength"):
        build(weight, qkv_spectral_strength=invalid)
    _, optimizer = build(weight, qkv_spectral_strength=0.5)
    with pytest.raises(ValueError, match="qkv_spectral_strength"):
        optimizer.set_qkv_spectral_strength(invalid)
    assert optimizer.qkv_spectral_strength == 0.5
