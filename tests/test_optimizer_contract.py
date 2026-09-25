"""Behavior that must hold for training loops and checkpoint continuation."""

from copy import deepcopy

import pytest
import torch

from tiger_optim import Tiger


def _simple_optimizer(params, **kwargs):
    options = dict(
        lr=0.1, betas=(0.0, 0.0), factored=False,
        precond_alpha=0.0, sign_mode="sign", sign_blend=0.0,
        use_foreach=False, use_foreach_update=False, auto_blend=False,
        lora_cross_adapt=False, qkv_lr_autoadapt=False,
        qkv_spectral_adapt=False,
    )
    options.update(kwargs)
    return Tiger(params, **options)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_gradient_skips_weight_decay_and_state(nonfinite):
    good = torch.nn.Parameter(torch.ones(2, device="cpu"))
    bad = torch.nn.Parameter(torch.ones(2, device="cpu"))
    opt = _simple_optimizer([good, bad], weight_decay=0.1, skip_if_nonfinite=True)
    good.grad = torch.ones_like(good)
    bad.grad = torch.tensor([1.0, nonfinite], device="cpu")

    opt.step()

    assert torch.all(good < 1.0)
    assert torch.equal(bad, torch.ones_like(bad))
    assert bad not in opt.state


def test_empty_parameter_passes_finite_gate_without_blocking_other_parameters():
    empty = torch.nn.Parameter(torch.empty(0, device="cpu"))
    valid = torch.nn.Parameter(torch.ones(2, device="cpu"))
    opt = _simple_optimizer([empty, valid], skip_if_nonfinite=True)
    empty.grad = torch.empty_like(empty)
    valid.grad = torch.ones_like(valid)

    opt.step()

    assert empty.numel() == 0
    assert torch.all(valid < 1.0)


def test_cpu_training_loop_descends_on_convex_quadratic():
    param = torch.nn.Parameter(torch.tensor([3.0, -4.0], device="cpu"))
    target = torch.tensor([0.5, 1.0], device="cpu")
    opt = Tiger(
        [param], lr=0.1, factored=False, precond_alpha=0.0,
        use_trust_ratio=False, weight_decay=0.0, auto_lr=False,
        auto_blend=False, lora_cross_adapt=False,
    )
    initial = (param.detach() - target).square().mean().item()
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        loss = (param - target).square().mean()
        loss.backward()
        opt.step()

    final = (param.detach() - target).square().mean().item()
    assert final < 0.1 * initial


def test_finite_gradient_overflow_does_not_poison_moments():
    param = torch.nn.Parameter(torch.ones(2, device="cpu"))
    opt = _simple_optimizer([param], skip_if_nonfinite=True)
    param.grad = torch.full_like(param, 1e30)

    opt.step()
    assert torch.equal(param, torch.ones_like(param))
    assert param not in opt.state

    param.grad = torch.ones_like(param)
    opt.step()
    assert torch.all(param < 1.0)
    assert torch.isfinite(opt.state[param]["v"]).all()


@pytest.mark.parametrize("use_foreach", [False, True])
def test_fp32_ordinary_update_overflow_skips_before_state_commit(use_foreach):
    param = torch.nn.Parameter(torch.ones(100, device="cpu"))
    opt = _simple_optimizer(
        [param], lr=3e38, sign_blend=1.0,
        use_trust_ratio=False, skip_if_nonfinite=True,
        use_foreach=use_foreach, use_foreach_update=use_foreach,
        foreach_min_bucket=1,
    )
    param.grad = torch.zeros_like(param)
    param.grad[0] = 1.0

    opt.step()

    assert torch.equal(param, torch.ones_like(param))
    assert param not in opt.state


def test_factored_moments_avoid_reduction_and_outer_product_overflow():
    param = torch.nn.Parameter(torch.ones((2, 1000), device="cpu"))
    opt = Tiger(
        [param], lr=0.1, betas=(0.0, 0.0), factored=True,
        precond_alpha=1.0, sign_mode="sign", sign_blend=0.0,
        use_trust_ratio=False, use_foreach=False, use_foreach_update=False,
        skip_if_nonfinite=True, auto_blend=False, lora_cross_adapt=False,
    )
    param.grad = torch.full_like(param, 1e18)
    opt.step()
    assert torch.isfinite(param).all()
    assert torch.isfinite(opt.state[param]["vr"]).all()
    assert torch.isfinite(opt.state[param]["vc"]).all()

    before = param.detach().clone()
    param.grad = torch.ones_like(param)
    opt.step()
    assert torch.all(param < before)


def test_fp16_moment_storage_overflow_skips_without_poisoning_state():
    param = torch.nn.Parameter(torch.ones(2, device="cpu"))
    opt = _simple_optimizer([param], mom_dtype="fp16", betas=(0.0, 0.98), weight_decay=0.1)
    param.grad = torch.full_like(param, 1e4)

    opt.step()
    assert torch.equal(param, torch.ones_like(param))
    assert param not in opt.state

    param.grad = torch.ones_like(param)
    opt.step()
    assert torch.all(param < 1.0)
    assert torch.isfinite(opt.state[param]["v"]).all()


def test_nonfinite_direction_skips_before_weight_decay_and_state_commit():
    param = torch.nn.Parameter(torch.ones(2, device="cpu"))
    opt = _simple_optimizer(
        [param], eps=0.0, weight_decay=0.1,
        sign_blend=1.0, skip_if_nonfinite=True,
    )
    param.grad = torch.zeros_like(param)
    opt.step()
    assert torch.equal(param, torch.ones_like(param))
    assert param not in opt.state


@pytest.mark.parametrize("route", ["scalar", "foreach", "qkv", "block"])
@pytest.mark.parametrize("lr,should_update", [(2e-4, True), (1.0, False)])
def test_fp16_scaled_update_stays_finite_or_skips_transactionally(route, lr, should_update):
    shape = (3, 2) if route == "qkv" else ((2, 2) if route == "block" else (2,))
    param = torch.nn.Parameter(torch.ones(shape, dtype=torch.float16, device="cpu"))
    group = {"params": [param]}
    if route == "qkv":
        group.update(_qkv_group(param))
    elif route == "block":
        group["block_trust_chunks"] = 2
    use_foreach = route == "foreach"
    opt = Tiger(
        [group], lr=lr, factored=False, precond_alpha=1.0,
        use_trust_ratio=False, agc_clip=0.0, weight_decay=0.1,
        qkv_lr_autoadapt=False, auto_lr=False, auto_blend=False,
        lora_cross_adapt=False, use_foreach=use_foreach,
        use_foreach_update=use_foreach, foreach_min_bucket=1,
        skip_if_nonfinite=True,
    )
    param.grad = torch.full_like(param, 1e-6)
    opt.step()
    if should_update:
        assert torch.isfinite(param).all()
        assert torch.all(param < 1.0)
        assert torch.isfinite(opt.state[param]["m"]).all()
    else:
        assert torch.equal(param, torch.ones_like(param))
        assert param not in opt.state


def test_fp16_unrepresentable_update_preserves_existing_moments():
    param = torch.nn.Parameter(torch.ones(2, dtype=torch.float16, device="cpu"))
    opt = Tiger(
        [param], lr=2e-4, factored=False, precond_alpha=1.0,
        use_trust_ratio=False, weight_decay=0.1,
        auto_lr=False, auto_blend=False, lora_cross_adapt=False,
        use_foreach=False, use_foreach_update=False,
        skip_if_nonfinite=True,
    )
    param.grad = torch.full_like(param, 1e-6)
    opt.step()
    before_param = param.detach().clone()
    before_state = deepcopy(opt.state[param])
    opt.param_groups[0]["lr"] = 1.0
    param.grad = torch.full_like(param, 1e-6)
    opt.step()
    torch.testing.assert_close(param, before_param)
    assert opt.state[param]["_state_sane"] is before_state["_state_sane"]
    torch.testing.assert_close(opt.state[param]["m"], before_state["m"])
    torch.testing.assert_close(opt.state[param]["v"], before_state["v"])


def test_fp16_decay_preview_matches_inplace_kernel_at_overflow_edge():
    param = torch.nn.Parameter(torch.tensor([65408.0], dtype=torch.float16, device="cpu"))
    opt = _simple_optimizer(
        [param], lr=1.0, weight_decay=2.001,
        use_trust_ratio=False, skip_if_nonfinite=True,
    )
    param.grad = torch.zeros_like(param)

    opt.step()

    assert torch.equal(param, torch.tensor([65408.0], dtype=torch.float16, device="cpu"))
    assert param not in opt.state


@pytest.mark.parametrize("skip_if_nonfinite", [False, True])
def test_fp16_foreach_buffer_scales_before_narrowing(skip_if_nonfinite):
    param = torch.nn.Parameter(torch.ones(2, dtype=torch.float32, device="cpu"))
    opt = Tiger(
        [param], lr=2e-4, factored=False, precond_alpha=1.0,
        use_trust_ratio=False, auto_lr=False, auto_blend=False,
        lora_cross_adapt=False, use_foreach=True,
        use_foreach_update=True, foreach_min_bucket=1,
        update_buffer_dtype="fp16", skip_if_nonfinite=skip_if_nonfinite,
    )
    param.grad = torch.full_like(param, 1e-6)
    opt.step()
    assert torch.isfinite(param).all()
    assert torch.all(param < 1.0)


@pytest.mark.parametrize("source", ["global", "mean"])
def test_fp16_bucket_standardization_updates_on_finite_gradients(source):
    params = [torch.nn.Parameter(torch.ones(8, dtype=torch.float16, device="cpu")) for _ in range(4)]
    opt = Tiger(
        [{"params": params, "bucket_standardize": True,
          "bucket_standardize_source": source}],
        lr=2e-4, factored=False, precond_alpha=1.0,
        use_trust_ratio=False, auto_lr=False, auto_blend=False,
        lora_cross_adapt=False, foreach_min_bucket=4,
        skip_if_nonfinite=True,
    )
    for param in params:
        param.grad = torch.ones_like(param)
    opt.step()
    assert all(torch.isfinite(param).all() and torch.all(param < 1.0) for param in params)


@pytest.mark.parametrize("source", ["global", "mean"])
def test_fp16_bucket_standardization_rejects_unstandardized_overflow(source):
    params = [torch.nn.Parameter(torch.ones(8, dtype=torch.float16, device="cpu")) for _ in range(4)]
    opt = Tiger(
        [{"params": params, "bucket_standardize": True,
          "bucket_standardize_source": source}],
        lr=1e35, factored=False, precond_alpha=1.0,
        use_trust_ratio=False, auto_lr=False, auto_blend=False,
        lora_cross_adapt=False, foreach_min_bucket=4,
        skip_if_nonfinite=True,
    )
    for param in params:
        param.grad = torch.full_like(param, 1e-6)

    opt.step()

    assert all(torch.equal(param, torch.ones_like(param)) for param in params)
    assert not opt.state


def test_fp16_median_bucket_standardization_rejects_unsupported_precommit():
    params = [torch.nn.Parameter(torch.ones(8, dtype=torch.float16, device="cpu")) for _ in range(4)]
    opt = Tiger(
        [{"params": params, "bucket_standardize": True,
          "bucket_standardize_source": "median"}],
        lr=2e-4, foreach_min_bucket=4, skip_if_nonfinite=True,
    )
    for param in params:
        param.grad = torch.ones_like(param)
    with pytest.raises(ValueError, match="median bucket standardization"):
        opt.step()
    assert all(torch.equal(param, torch.ones_like(param)) for param in params)
    assert not opt.state
    assert opt._global_step == 0


def test_fp16_qkv_agc_and_trust_preview_matches_applied_step():
    param = torch.nn.Parameter(torch.ones((3, 2), dtype=torch.float16, device="cpu"))
    opt = Tiger(
        [_qkv_group(param)], lr=0.1, factored=False,
        precond_alpha=1.0, sign_blend=1.0,
        use_trust_ratio=True, trust_clip=100.0,
        agc_clip=0.5, qkv_lr_autoadapt=False,
        auto_lr=False, auto_blend=False, lora_cross_adapt=False,
        use_foreach=False, use_foreach_update=False,
        skip_if_nonfinite=True,
    )
    param.grad = torch.full_like(param, 1e-6)
    opt.step()
    torch.testing.assert_close(
        param.detach().float(), torch.full_like(param, 0.9).float(),
        rtol=0.0, atol=0.002,
    )
    assert param in opt.state


def test_loaded_corrupt_moment_is_sanitized_before_update():
    param = torch.nn.Parameter(torch.ones(2, device="cpu"))
    old = _simple_optimizer([param])
    old.state[param]["m"] = torch.zeros_like(param)
    old.state[param]["v"] = torch.full_like(param, float("inf"))
    old.state[param]["_state_sane"] = True
    legacy_checkpoint = old.state_dict()
    legacy_checkpoint.pop("tiger_state")

    restored_param = torch.nn.Parameter(param.detach().clone())
    restored = _simple_optimizer([restored_param])
    restored.load_state_dict(legacy_checkpoint)
    assert restored.state[restored_param]["_state_sane"] is False
    restored_param.grad = torch.ones_like(restored_param)
    restored.step()

    assert torch.all(restored_param < 1.0)
    assert torch.isfinite(restored.state[restored_param]["v"]).all()


def _qkv_group(param, scales=None):
    return {
        "params": [param],
        "block_tag": "attn_qkv",
        "qkv_rules": {id(param): (0, 3)},
        "qkv_trust_split": True,
        "qkv_lr_scales": scales or {"q": 1.0, "k": 1.0, "v": 1.0},
    }


def test_qkv_slice_learning_rates_work_without_trust():
    param = torch.nn.Parameter(torch.ones((3, 2), device="cpu"))
    opt = _simple_optimizer(
        [_qkv_group(param, {"q": 0.5, "k": 1.0, "v": 1.5})],
        use_trust_ratio=False,
    )
    param.grad = torch.ones_like(param)

    opt.step()

    torch.testing.assert_close(param.detach()[:, 0], torch.tensor([0.95, 0.9, 0.85], device="cpu"))


def test_qkv_preconditioned_trust_uses_preconditioned_momentum():
    param = torch.nn.Parameter(torch.ones((3, 2), device="cpu"))
    opt = _simple_optimizer([_qkv_group(param)], use_trust_ratio=True, trust_space="precond")
    param.grad = torch.full_like(param, 2.0)

    opt.step()

    # Norm(param) / Norm(momentum * preconditioner) = 1 / 2 per slice.
    torch.testing.assert_close(param.detach(), torch.full_like(param, 0.95))


@pytest.mark.parametrize("qkv", [False, True])
def test_trust_ema_is_independent_for_parameters_in_one_group(qkv):
    small = torch.nn.Parameter(torch.ones((3, 2), device="cpu"))
    large = torch.nn.Parameter(torch.full((3, 2), 100.0, device="cpu"))
    reference = torch.nn.Parameter(small.detach().clone())

    if qkv:
        grouped = {
            "params": [small, large], "block_tag": "attn_qkv",
            "qkv_rules": {id(small): (0, 3), id(large): (0, 3)},
            "qkv_trust_split": True,
            "qkv_lr_scales": {"q": 1.0, "k": 1.0, "v": 1.0},
        }
        reference_group = _qkv_group(reference)
    else:
        grouped = {"params": [small, large]}
        reference_group = {"params": [reference]}

    options = dict(use_trust_ratio=True, trust_clip=1000.0, trust_ema_beta=0.9)
    opt = _simple_optimizer([grouped], **options)
    ref_opt = _simple_optimizer([reference_group], **options)
    for _ in range(2):
        small.grad = torch.ones_like(small)
        large.grad = torch.ones_like(large)
        reference.grad = torch.ones_like(reference)
        opt.step()
        ref_opt.step()

    torch.testing.assert_close(small.detach(), reference.detach())


@pytest.mark.parametrize("spectral", [False, True])
def test_qkv_adaptation_aggregates_all_group_parameters_independent_of_order(spectral):
    def run(order):
        first = torch.nn.Parameter(torch.ones((3, 8), device="cpu"))
        second = torch.nn.Parameter(torch.ones((3, 8), device="cpu"))
        params = {"first": first, "second": second}
        group = {
            "params": [params[name] for name in order],
            "block_tag": "attn_qkv",
            "qkv_rules": {id(first): (0, 3), id(second): (0, 3)},
            "qkv_trust_split": True,
            "qkv_lr_scales": {"q": 1.0, "k": 1.0, "v": 1.0},
        }
        opt = _simple_optimizer(
            [group], use_trust_ratio=True, trust_clip=100.0,
            sign_blend=1.0,
            qkv_lr_interval=1, qkv_lr_gain=0.5, qkv_lr_ema_beta=0.0,
            qkv_lr_autoadapt=True, qkv_spectral_adapt=spectral,
        )
        pattern = torch.linspace(-0.2, 0.2, 8, device="cpu")
        first_grad = torch.tensor([1.0, 2.0, 4.0], device="cpu")[:, None] + pattern
        second_grad = torch.tensor([4.0, 1.5, 2.0], device="cpu")[:, None] - pattern
        scales = []
        for factor in (1.0, 1.2):
            first.grad = first_grad * factor
            second.grad = second_grad * factor
            opt.step()
            scales.append(dict(opt.param_groups[0]["qkv_lr_scales"]))
        return scales, opt._qkv_spec_ema, first.detach(), second.detach()

    forward = run(("first", "second"))
    reversed_order = run(("second", "first"))
    for actual, expected in zip(forward[0], reversed_order[0]):
        assert actual == pytest.approx(expected)
    assert any(abs(scale - 1.0) > 1e-3 for scale in forward[0][-1].values())
    torch.testing.assert_close(forward[2], reversed_order[2])
    torch.testing.assert_close(forward[3], reversed_order[3])
    if spectral:
        for label in ("q", "k", "v"):
            for band in ("low", "high", "phase"):
                torch.testing.assert_close(
                    forward[1][0][label][band], reversed_order[1][0][label][band]
                )


def test_auto_lr_decays_once_per_plateau_until_floor():
    param = torch.nn.Parameter(torch.ones(2, device="cpu"))
    opt = _simple_optimizer(
        [param], auto_lr=True, lr_decay=0.5, lr_min=0.025,
        plateau_patience=2,
    )
    rates = []
    for _ in range(7):
        opt.report_metrics(loss=1.0)
        param.grad = torch.ones_like(param)
        opt.step()
        rates.append(opt.param_groups[0]["lr"])

    assert rates == pytest.approx([0.1, 0.1, 0.05, 0.05, 0.025, 0.025, 0.025])


def test_nonfinite_reported_loss_does_not_poison_plateau_baseline():
    param = torch.nn.Parameter(torch.ones(2, device="cpu"))
    opt = _simple_optimizer([param], auto_lr=True, plateau_patience=1)
    opt.report_metrics(loss=float("nan"))
    assert opt._plateau["best"] is None

    for loss in (2.0, 1.0, 0.5):
        opt.report_metrics(loss=loss)
        param.grad = torch.ones_like(param)
        opt.step()

    assert opt._plateau["best"] == pytest.approx(0.5)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)


def _adaptive_qkv_optimizer(param, **kwargs):
    return Tiger(
        [_qkv_group(param)],
        lr=0.05,
        betas=(0.2, 0.5),
        factored=False,
        precond_alpha=0.0,
        sign_blend=0.2,
        blend_to=0.8,
        blend_steps=4,
        trust_ema_beta=0.5,
        qkv_lr_interval=1,
        auto_blend=True,
        plateau_patience=1,
        lora_cross_adapt=False,
        use_foreach=False,
        use_foreach_update=False,
        **kwargs,
    )


def _adaptive_step(opt, param, loss):
    opt.report_metrics(loss=loss)
    param.grad = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]], device="cpu")
    opt.step()


def test_checkpoint_resumes_adaptation_and_qkv_slices():
    continuous = torch.nn.Parameter(torch.ones((3, 2), device="cpu"))
    opt = _adaptive_qkv_optimizer(continuous)
    _adaptive_step(opt, continuous, 1.0)

    checkpoint = deepcopy(opt.state_dict())
    assert checkpoint["tiger_state"]["_global_step"] == 1
    assert set(checkpoint["param_groups"][0]["qkv_rules"]) == set(checkpoint["param_groups"][0]["params"])

    restored_param = torch.nn.Parameter(continuous.detach().clone())
    restored = _adaptive_qkv_optimizer(restored_param)
    restored.load_state_dict(checkpoint)
    assert restored._global_step == 1
    assert id(restored_param) in restored.param_groups[0]["qkv_rules"]

    _adaptive_step(opt, continuous, 1.1)
    _adaptive_step(restored, restored_param, 1.1)

    torch.testing.assert_close(restored_param.detach(), continuous.detach(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(restored.state[restored_param]["m"], opt.state[continuous]["m"])
    assert restored.param_groups[0]["qkv_lr_scales"] == pytest.approx(opt.param_groups[0]["qkv_lr_scales"])
    assert restored.param_groups[0]["sign_blend"] == pytest.approx(opt.param_groups[0]["sign_blend"])


def test_profiler_does_not_change_spectral_adaptation(tmp_path):
    plain_param = torch.nn.Parameter(torch.ones((3, 2), device="cpu"))
    profiled_param = torch.nn.Parameter(plain_param.detach().clone())
    plain = _adaptive_qkv_optimizer(plain_param)
    profiled = _adaptive_qkv_optimizer(
        profiled_param,
        profiler_enabled=True,
        profiler_interval=1,
        profiler_path=str(tmp_path / "profile.jsonl"),
    )
    plain.defaults["qkv_lr_interval"] = 3
    profiled.defaults["qkv_lr_interval"] = 3

    for _ in range(3):
        _adaptive_step(plain, plain_param, 1.0)
        _adaptive_step(profiled, profiled_param, 1.0)
        torch.testing.assert_close(profiled_param.detach(), plain_param.detach())
    assert profiled.param_groups[0]["qkv_lr_scales"] == pytest.approx(plain.param_groups[0]["qkv_lr_scales"])
