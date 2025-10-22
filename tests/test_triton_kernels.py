import pytest
import torch

from tiger_optim.triton_kernels import fused_apply_updates


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_apply_updates_applies_changes_in_place():
    pytest.importorskip("triton")

    device = torch.device("cuda")
    params = [
        torch.randn(4, 4, device=device, dtype=torch.float16),
        torch.randn(3, device=device, dtype=torch.float16),
    ]
    updates = [
        torch.randn_like(params[0], dtype=torch.float32),
        torch.randn_like(params[1], dtype=torch.float32),
    ]

    originals = [p.clone() for p in params]

    assert fused_apply_updates(params, updates) is True

    for original, update, param in zip(originals, updates, params):
        expected = original + update.to(device=param.device, dtype=param.dtype)
        assert torch.allclose(param, expected, atol=1e-4, rtol=1e-4)
