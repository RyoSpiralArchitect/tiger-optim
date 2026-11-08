import pytest
import torch

from tiger_optim import Tiger, build_tagged_param_groups


class _TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(64, 4, batch_first=True)
        self.proj = torch.nn.Linear(64, 64)


def _run_two_phase(monkeypatch, enabled):
    monkeypatch.setenv("TIGER_FLAT_STEP", "1")
    monkeypatch.setenv("TIGER_FLAT_STEP_TWO_PHASE", "1" if enabled else "0")
    torch.manual_seed(23)
    torch.cuda.manual_seed_all(23)
    model = _TinyAttention().cuda()
    groups = build_tagged_param_groups(model, enable_qkv_slicing=True)
    qkv_group = next(group for group in groups if group["block_tag"] == "attn_qkv")
    qkv_group_index = next(index for index, group in enumerate(groups) if group is qkv_group)
    assert qkv_group["qkv_trust_split"]
    assert qkv_group["qkv_lr_scales"]
    opt = Tiger(
        groups,
        factored=True,
        precond_alpha=1.0,
        trust_space="precond",
        trust_ema_beta=0.9,
        use_trust_ratio=True,
        skip_if_nonfinite=True,
    )

    qkv_param = model.attn.in_proj_weight
    before_nonfinite_step = None
    for step in range(5):
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        if step == 4:
            before_nonfinite_step = qkv_param.detach().clone()
            qkv_param.grad.reshape(-1)[0] = float("nan")
        opt.step()

    assert opt._flat_cache.get(qkv_group_index)
    assert sum(bool(cache) for cache in opt._flat_cache.values()) >= 2
    assert torch.equal(qkv_param, before_nonfinite_step)
    params = {name: param.detach().clone() for name, param in model.named_parameters()}
    state = {
        name: {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in opt.state[param].items()
        }
        for name, param in model.named_parameters()
    }
    trust = {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in opt._qkv_trust_ema.items()
    }
    return params, state, trust


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flat_step_two_phase_matches_legacy_with_qkv_and_nonfinite(monkeypatch):
    pytest.importorskip("triton")
    legacy = _run_two_phase(monkeypatch, enabled=False)
    pipelined = _run_two_phase(monkeypatch, enabled=True)
    for legacy_values, pipelined_values in zip(legacy, pipelined):
        assert legacy_values.keys() == pipelined_values.keys()
        for key in legacy_values:
            old = legacy_values[key]
            new = pipelined_values[key]
            if isinstance(old, dict):
                assert old.keys() == new.keys()
                for subkey in old:
                    old_value = old[subkey]
                    new_value = new[subkey]
                    if isinstance(old_value, torch.Tensor):
                        torch.testing.assert_close(old_value, new_value, rtol=0, atol=0)
                    else:
                        assert old_value == new_value
            elif isinstance(old, torch.Tensor):
                torch.testing.assert_close(old, new, rtol=0, atol=0)
            else:
                assert old == new
