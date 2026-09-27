"""Contracts for the opt-in tail cosine learning-rate schedule."""

from copy import deepcopy

import pytest
import torch

from tiger_optim import TailCosineLR, Tiger


def test_archived_100_update_learning_rates():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    scheduler = TailCosineLR(
        optimizer, total_steps=100, decay_start=90, min_lr_ratio=0.1,
    )

    used_lrs = []
    for _ in range(100):
        used_lrs.append(optimizer.param_groups[0]["lr"])
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()

    assert [used_lrs[step - 1] for step in (1, 91, 92, 100)] == pytest.approx(
        [0.01, 0.01, 0.009779754323328192, 0.0012202456766718093]
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)


def test_group_ratios_and_lr_scales_remain_unchanged():
    first = torch.nn.Parameter(torch.zeros(1))
    second = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD(
        [
            {"params": [first], "lr": 0.01, "lr_scale": 0.5},
            {"params": [second], "lr": 0.003, "lr_scale": 1.25},
        ]
    )
    scheduler = TailCosineLR(
        optimizer, total_steps=10, decay_start=8, min_lr_ratio=0.2,
    )

    for _ in range(12):
        left, right = optimizer.param_groups
        assert left["lr"] / right["lr"] == pytest.approx(0.01 / 0.003)
        assert (left["lr_scale"], right["lr_scale"]) == (0.5, 1.25)
        first.grad = torch.ones_like(first)
        second.grad = torch.ones_like(second)
        optimizer.step()
        scheduler.step()

    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [0.002, 0.0006]
    )


@pytest.mark.parametrize(
    "options",
    [
        {"total_steps": 0},
        {"total_steps": -1},
        {"total_steps": 100.0},
        {"total_steps": True},
        {"decay_start": -1},
        {"decay_start": 100},
        {"decay_start": 90.0},
        {"decay_start": True},
        {"min_lr_ratio": -0.01},
        {"min_lr_ratio": 1.01},
        {"min_lr_ratio": float("nan")},
        {"min_lr_ratio": float("inf")},
        {"min_lr_ratio": None},
        {"min_lr_ratio": True},
    ],
)
def test_invalid_schedule_inputs(options):
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    kwargs = {"total_steps": 100, "decay_start": 90, "min_lr_ratio": 0.1}
    kwargs.update(options)
    with pytest.raises(ValueError):
        TailCosineLR(optimizer, **kwargs)


def _training_pair(parameter):
    optimizer = Tiger(
        [parameter], lr=0.01, weight_decay=0.01, betas=(0.8, 0.9),
        factored=False, precond_alpha=0.0, use_trust_ratio=False,
        use_foreach=False, use_foreach_update=False,
        auto_lr=False, auto_blend=False, lora_cross_adapt=False,
        qkv_lr_autoadapt=False, qkv_spectral_adapt=False,
    )
    scheduler = TailCosineLR(
        optimizer, total_steps=100, decay_start=90, min_lr_ratio=0.1,
    )
    return optimizer, scheduler


def _train(parameter, optimizer, scheduler, steps):
    target = torch.tensor([-0.5, 0.25])
    trace = []
    for _ in range(steps):
        trace.append((optimizer.param_groups[0]["lr"], parameter.detach().clone()))
        optimizer.zero_grad(set_to_none=True)
        (parameter - target).square().mean().backward()
        optimizer.step()
        scheduler.step()
    return trace


def test_tiger_and_scheduler_resume_at_update_95():
    parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    optimizer, scheduler = _training_pair(parameter)
    _train(parameter, optimizer, scheduler, 95)

    saved_parameter = parameter.detach().clone()
    saved_optimizer = deepcopy(optimizer.state_dict())
    saved_scheduler = deepcopy(scheduler.state_dict())
    uninterrupted = _train(parameter, optimizer, scheduler, 5)

    resumed_parameter = torch.nn.Parameter(saved_parameter)
    resumed_optimizer, resumed_scheduler = _training_pair(resumed_parameter)
    resumed_optimizer.load_state_dict(saved_optimizer)
    resumed_scheduler.load_state_dict(saved_scheduler)
    resumed = _train(resumed_parameter, resumed_optimizer, resumed_scheduler, 5)

    for (expected_lr, expected_value), (actual_lr, actual_value) in zip(
        uninterrupted, resumed
    ):
        assert actual_lr == pytest.approx(expected_lr, rel=0, abs=0)
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
    torch.testing.assert_close(resumed_parameter, parameter, rtol=0, atol=0)
    assert resumed_scheduler.last_epoch == scheduler.last_epoch == 100
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
