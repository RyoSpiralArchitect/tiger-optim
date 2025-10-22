import contextlib

import tiger_optim.triton_kernels as triton_kernels

import pytest
import torch

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_median_tensor_empty_on_cuda_matches_reference_device():
    reference = torch.zeros((), dtype=torch.float32, device=torch.device("cuda"))
    result = tiger._median_tensor([], reference=reference)
    assert result.device == reference.device
    assert result.dtype == reference.dtype
    assert result.item() == pytest.approx(0.0)


def test_fused_apply_triton_respects_kernel_result(monkeypatch):
    param = torch.nn.Parameter(torch.zeros(4))
    opt = tiger.Tiger([param], lr=1e-3)

    tensors = [param.detach()]
    updates = [torch.ones_like(param.detach())]

    def _success(params, updates):
        return True

    monkeypatch.setattr(triton_kernels, "fused_apply_updates", _success)
    assert opt._fused_apply_triton(tensors, updates) is True

    def _failure(params, updates):
        return False

    monkeypatch.setattr(triton_kernels, "fused_apply_updates", _failure)
    assert opt._fused_apply_triton(tensors, updates) is False

    def _error(params, updates):
        raise RuntimeError("boom")

    monkeypatch.setattr(triton_kernels, "fused_apply_updates", _error)
    assert opt._fused_apply_triton(tensors, updates) is False
