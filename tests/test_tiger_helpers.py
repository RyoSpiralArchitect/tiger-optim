import contextlib

import tiger_optim.triton_kernels as triton_kernels

import pytest
import torch
import torch.nn.functional as F

from tiger_optim import tiger


@contextlib.contextmanager
def _restore_default_dtype(new_dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(new_dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def test_scalar_like_without_reference_uses_default_dtype():
    with _restore_default_dtype(torch.float64):
        value = tiger._scalar_like(None, 1.25)
        assert value.dtype == torch.float64
        assert value.device.type == "cpu"


def test_scalar_like_explicit_device_without_reference():
    value = tiger._scalar_like(None, 0.5, dtype=torch.float32, device=torch.device("cpu"))
    assert value.dtype == torch.float32
    assert value.device.type == "cpu"


def test_scalar_like_overrides_reference_dtype():
    reference = torch.ones((), dtype=torch.float16)
    value = tiger._scalar_like(reference, 2.0, dtype=torch.float32)
    assert value.dtype == torch.float32
    assert value.device == reference.device


def test_scalar_like_allows_device_override_with_reference():
    reference = torch.ones((), dtype=torch.float32)
    value = tiger._scalar_like(reference, 3.0, device=torch.device("meta"))
    assert value.device.type == "meta"
    assert value.dtype == reference.dtype


def test_median_tensor_empty_uses_reference_metadata():
    reference = torch.ones((), dtype=torch.float16)
    result = tiger._median_tensor([], reference=reference)
    assert result.device == reference.device
    assert result.dtype == reference.dtype
    assert result.item() == pytest.approx(0.0)


def test_median_tensor_nonempty_aligns_reference_dtype():
    reference = torch.ones((), dtype=torch.float16)
    values = [torch.tensor(0.25, dtype=torch.float32), torch.tensor(0.5, dtype=torch.float32)]
    result = tiger._median_tensor(values, reference=reference)
    assert result.dtype == reference.dtype
    assert result.device == reference.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_median_tensor_empty_on_cuda_matches_reference_device():
    reference = torch.zeros((), dtype=torch.float32, device=torch.device("cuda"))
    result = tiger._median_tensor([], reference=reference)
    assert result.device == reference.device
    assert result.dtype == reference.dtype
    assert result.item() == pytest.approx(0.0)


def test_reference_tensor_scans_nested_iterables():
    tensor = torch.ones((), dtype=torch.float32)
    nested = {"payload": [None, {"value": tensor}]}
    assert tiger._reference_tensor(None, nested) is tensor
    assert tiger._reference_tensor(None) is None


def test_spectral_dispersion_tensors_return_device_scalars():
    x = torch.linspace(-1.0, 1.0, steps=16)

    low, high, phase = tiger._spectral_dispersion_tensors(x, 0.25, 0.25)

    assert all(isinstance(value, torch.Tensor) for value in (low, high, phase))
    assert all(value.ndim == 0 for value in (low, high, phase))
    assert all(value.device == x.device for value in (low, high, phase))
    assert all(torch.isfinite(value) for value in (low, high, phase))


def test_foreach_all_finite_reports_expected_result():
    good = [torch.ones(4), torch.full((4,), 2.0)]
    bad = [torch.ones(4), torch.tensor([1.0, float("nan"), 2.0, 3.0])]

    assert tiger._foreach_all_finite(good) is True
    assert tiger._foreach_all_finite(bad) is False


def test_chunk_scalar_norms_match_manual_equal_split():
    x = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    norms_dim0 = tiger._chunk_scalar_norms(x, 0, 3)
    manual_dim0 = torch.stack([chunk.square().sum().sqrt() for chunk in torch.chunk(x, 3, dim=0)])
    assert torch.allclose(norms_dim0, manual_dim0)

    y = torch.arange(36, dtype=torch.float32).reshape(3, 12)
    norms_dim1 = tiger._chunk_scalar_norms(y, 1, 3)
    manual_dim1 = torch.stack([chunk.square().sum().sqrt() for chunk in torch.chunk(y, 3, dim=1)])
    assert torch.allclose(norms_dim1, manual_dim1)


def test_scalars_to_host_batches_scalar_conversion():
    values = tiger._scalars_to_host([torch.tensor(1.25), 2, 3.5])
    assert values == pytest.approx([1.25, 2.0, 3.5])


def test_step_skips_only_nonfinite_param_when_group_probe_fails():
    good = torch.nn.Parameter(torch.ones(4))
    bad = torch.nn.Parameter(torch.ones(4))
    optimizer = tiger.Tiger(
        [{"params": [good, bad]}],
        lr=1e-1,
        betas=(0.0, 0.0),
        weight_decay=0.0,
        factored=False,
        precond_alpha=0.0,
        sign_mode="sign",
        sign_blend=1.0,
        use_foreach=False,
        use_foreach_update=False,
        skip_if_nonfinite=True,
    )

    good_before = good.detach().clone()
    bad_before = bad.detach().clone()
    good.grad = torch.ones_like(good)
    bad.grad = torch.tensor([1.0, float("nan"), 1.0, 1.0])

    optimizer.step()

    assert not torch.equal(good.detach(), good_before)
    assert torch.equal(bad.detach(), bad_before)


def test_fused_apply_triton_records_kernel_result(monkeypatch):
    param = torch.nn.Parameter(torch.zeros(4))
    opt = tiger.Tiger([param], lr=1e-3)

    tensors = [param.detach()]
    updates = [torch.ones_like(param.detach())]

    opt._fused_apply_failures = []
    assert opt._fused_apply_triton(tensors, updates) is False
    assert opt._fused_apply_failures
    reason = opt._fused_apply_failures[-1]
    assert reason.startswith("cuda_unavailable") or reason.startswith("non_cuda_param")

    if torch.cuda.is_available():
        param_cuda = torch.nn.Parameter(torch.zeros(4, device=torch.device("cuda")))
        opt_cuda = tiger.Tiger([param_cuda], lr=1e-3)
        tensors_cuda = [param_cuda.detach()]
        updates_cuda = [torch.ones_like(param_cuda.detach())]

        def _success(params, updates):
            return True

        monkeypatch.setattr(triton_kernels, "fused_apply_updates", _success)
        opt_cuda._fused_apply_failures = []
        assert opt_cuda._fused_apply_triton(tensors_cuda, updates_cuda) is True
        assert opt_cuda._fused_apply_failures == []

        def _decline(params, updates):
            return False

        monkeypatch.setattr(triton_kernels, "fused_apply_updates", _decline)
        opt_cuda._fused_apply_failures = []
        assert opt_cuda._fused_apply_triton(tensors_cuda, updates_cuda) is False
        assert opt_cuda._fused_apply_failures[-1].startswith("kernel_declined")

        def _error(params, updates):
            raise RuntimeError("boom")

        monkeypatch.setattr(triton_kernels, "fused_apply_updates", _error)
        opt_cuda._fused_apply_failures = []
        assert opt_cuda._fused_apply_triton(tensors_cuda, updates_cuda) is False
        assert opt_cuda._fused_apply_failures[-1].startswith("kernel_exception")


def test_step_profiler_includes_triton_failures(tmp_path):
    param = torch.nn.Parameter(torch.ones(4))
    opt = tiger.Tiger(
        [param],
        lr=1e-3,
        use_triton_fused_apply=True,
        foreach_min_bucket=1,
        profiler_enabled=True,
        profiler_path=str(tmp_path / "tiger_profile.jsonl"),
        profiler_interval=1,
    )
    param.grad = torch.ones_like(param)
    opt.step()
    payload = opt._prof.last_payload()
    failures = payload.get("foreach_triton_failures")
    assert failures is None or isinstance(failures, list)
    if failures:
        assert all(isinstance(item, str) for item in failures)


def test_factored_step_keeps_preconditioner_finite():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(6, 4))
    opt = tiger.Tiger(
        [param],
        lr=1e-3,
        factored=True,
        precond_alpha=1.0,
        use_foreach=False,
        use_foreach_update=False,
    )

    for _ in range(2):
        param.grad = torch.randn_like(param)
        opt.step()
        state = opt.state[param]
        assert torch.isfinite(param).all()
        assert torch.isfinite(state["m"]).all()
        assert torch.isfinite(state["vr"]).all()
        assert torch.isfinite(state["vc"]).all()


def test_qkv_spectral_collection_skips_when_not_due(monkeypatch):
    param = torch.nn.Parameter(torch.randn(6, 4))
    calls = []

    def _spy(*args, **kwargs):
        calls.append(1)
        return (
            torch.tensor(0.1, dtype=torch.float32),
            torch.tensor(0.2, dtype=torch.float32),
            torch.tensor(0.3, dtype=torch.float32),
        )

    monkeypatch.setattr(tiger, "_spectral_dispersion_tensors", _spy)
    opt = tiger.Tiger(
        [{
            "params": [param],
            "block_tag": "attn_qkv",
            "qkv_rules": {id(param): (0, 3)},
            "qkv_trust_split": True,
            "qkv_lr_scales": {"q": 1.0, "k": 1.0, "v": 1.0},
        }],
        lr=1e-3,
        factored=False,
        precond_alpha=0.0,
        qkv_spectral_adapt=True,
        qkv_lr_autoadapt=True,
        qkv_lr_interval=10,
        profiler_enabled=False,
    )

    param.grad = torch.randn_like(param)
    opt.step()

    assert calls == []


def test_qkv_spectral_collection_runs_when_adapt_due(monkeypatch):
    param = torch.nn.Parameter(torch.randn(6, 4))
    calls = []

    def _spy(*args, **kwargs):
        calls.append(1)
        return (
            torch.tensor(0.1, dtype=torch.float32),
            torch.tensor(0.2, dtype=torch.float32),
            torch.tensor(0.3, dtype=torch.float32),
        )

    monkeypatch.setattr(tiger, "_spectral_dispersion_tensors", _spy)
    opt = tiger.Tiger(
        [{
            "params": [param],
            "block_tag": "attn_qkv",
            "qkv_rules": {id(param): (0, 3)},
            "qkv_trust_split": True,
            "qkv_lr_scales": {"q": 1.0, "k": 1.0, "v": 1.0},
        }],
        lr=1e-3,
        factored=False,
        precond_alpha=0.0,
        qkv_spectral_adapt=True,
        qkv_lr_autoadapt=True,
        qkv_lr_interval=1,
        profiler_enabled=False,
    )

    param.grad = torch.randn_like(param)
    opt.step()

    assert len(calls) == 3


def test_loaded_dense_state_is_sanitized_once_before_update():
    param = torch.nn.Parameter(torch.ones(4))
    opt = tiger.Tiger(
        [param],
        lr=1e-3,
        factored=False,
        precond_alpha=0.0,
        skip_if_nonfinite=True,
        use_foreach=False,
        use_foreach_update=False,
        mom_dtype="fp32",
    )

    opt.state[param]["m"] = torch.tensor([float("nan"), 1.0, float("-inf"), 2.0], dtype=torch.float32)
    opt.state[param]["v"] = torch.tensor([float("nan"), -1.0, float("inf"), 2.0], dtype=torch.float32)
    opt.state[param]["_state_sane"] = False
    param.grad = torch.ones_like(param)

    opt.step()

    state = opt.state[param]
    assert state["_state_sane"] is True
    assert torch.isfinite(state["m"]).all()
    assert torch.isfinite(state["v"]).all()
    assert torch.all(state["v"] >= 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tiger_step_amp_fp16_remains_finite():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    ).cuda()
    optimizer = tiger.Tiger(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    data = torch.randn(4, 8, device=torch.device("cuda"), dtype=torch.float32)
    target = torch.randn(4, 4, device=torch.device("cuda"), dtype=torch.float32)

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = model(data)
            loss = F.mse_loss(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        assert torch.isfinite(loss)
        for param in model.parameters():
            assert torch.isfinite(param).all()
            state = optimizer.state[param]
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    assert torch.isfinite(value).all()


def test_lora_cross_density_bridge_updates_trust_clip():
    p_lora = torch.nn.Parameter(torch.zeros(4))
    p_attn = torch.nn.Parameter(torch.zeros(4))
    p_ffn = torch.nn.Parameter(torch.zeros(4))

    optimizer = tiger.Tiger(
        [
            {"params": [p_lora], "block_tag": "lora_a"},
            {"params": [p_attn], "block_tag": "attn_qkv"},
            {"params": [p_ffn], "block_tag": "ffn_up"},
        ],
        lr=1e-3,
        betas=(0.0, 0.0),
        weight_decay=0.0,
        factored=False,
        precond_alpha=0.0,
        sign_mode="sign",
        sign_blend=1.0,
        use_foreach=False,
        use_foreach_update=False,
        lora_interval=1,
        lora_density_adapt=True,
        lora_cross_adapt=True,
        lora_cross_beta=0.2,
        lora_cross_sync=0.5,
        lora_cross_gain=0.1,
        lora_bridge=True,
        lora_bridge_gain=0.5,
        lora_bridge_bounds=(0.8, 1.2),
        lora_bridge_beta=0.0,
    )

    p_lora.grad = torch.tensor([0.4, 0.02, 0.0, 0.0])
    p_attn.grad = torch.tensor([0.3, 0.3, 0.3, 0.3])
    p_ffn.grad = torch.tensor([0.01, 0.0, 0.0, 0.0])

    optimizer.step()

    cross_state = optimizer._lora_cross_state
    assert cross_state["global_density"] is not None
    tag_map = cross_state.get("tag_density", {})
    assert "attn_qkv" in tag_map and "ffn_up" in tag_map

    lora_state = optimizer._lora_pid.get(0)
    assert lora_state is not None
    assert lora_state["ema"] == pytest.approx(cross_state["global_density"], rel=0.5)

    default_trust = optimizer.defaults["trust_clip"]
    attn_trust = optimizer.param_groups[1]["trust_clip"]
    ffn_trust = optimizer.param_groups[2]["trust_clip"]
    assert attn_trust != pytest.approx(default_trust)
    assert attn_trust != pytest.approx(ffn_trust)

    metrics = optimizer.get_last_metrics()
    assert "lora_bridge_global" in metrics
    assert metrics["lora_bridge_global"] == pytest.approx(cross_state["global_density"])


def test_lora_cross_sync_aligns_lora_ema():
    p_a = torch.nn.Parameter(torch.zeros(4))
    p_b = torch.nn.Parameter(torch.zeros(4))

    optimizer = tiger.Tiger(
        [
            {"params": [p_a], "block_tag": "lora_a", "sign_blend": 1.0},
            {"params": [p_b], "block_tag": "lora_b", "sign_blend": 1.0},
        ],
        lr=1e-3,
        betas=(0.0, 0.0),
        weight_decay=0.0,
        factored=False,
        precond_alpha=0.0,
        sign_mode="sign",
        sign_blend=1.0,
        use_foreach=False,
        use_foreach_update=False,
        lora_interval=1,
        lora_density_adapt=True,
        lora_density_k=0.25,
        lora_density_beta=0.0,
        lora_cross_adapt=True,
        lora_cross_beta=0.0,
        lora_cross_sync=0.5,
        lora_cross_gain=0.0,
        lora_bridge=False,
    )

    p_a.grad = torch.tensor([1.0, 1.0, 1.0, 1.0])
    p_b.grad = torch.tensor([1.0, 0.0, 0.0, 0.0])

    optimizer.step()

    state_a = optimizer._lora_pid[0]
    state_b = optimizer._lora_pid[1]
    cross_state = optimizer._lora_cross_state

    assert cross_state["global_density"] == pytest.approx(0.625, rel=1e-4, abs=1e-4)
    assert state_a["ema"] == pytest.approx(0.8125, rel=1e-4, abs=1e-4)
    assert state_b["ema"] == pytest.approx(0.4375, rel=1e-4, abs=1e-4)
