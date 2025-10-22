import pytest
import torch

from tiger_optim import Tiger


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tiger_amp_step_keeps_states_finite():
    torch.manual_seed(0)

    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16, bias=False),
    ).cuda()
    model.train()

    optimizer = Tiger(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    data = torch.randn(8, 16, device="cuda")
    target = torch.randn(8, 16, device="cuda")

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = model(data)
            loss = torch.nn.functional.mse_loss(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for param in model.parameters():
            assert torch.all(torch.isfinite(param))
            state = optimizer.state[param]
            for key in ("m", "v", "vr", "vc"):
                tensor = state.get(key)
                if isinstance(tensor, torch.Tensor):
                    assert torch.all(torch.isfinite(tensor)), key
