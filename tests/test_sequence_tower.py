from __future__ import annotations

import torch

from rna3d_local.se3.sequence_tower import _MambaLikeBlock


def _reference_scan(u: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    state = torch.zeros((u.shape[1],), dtype=u.dtype, device=u.device)
    out = torch.zeros_like(u)
    for index in range(int(u.shape[0])):
        state = (decay * state) + u[index]
        out[index] = state
    return out


def test_parallel_associative_scan_matches_reference() -> None:
    torch.manual_seed(7)
    for length in (1, 2, 7, 31):
        channels = 9
        u = torch.randn((length, channels), dtype=torch.float32)
        decay = torch.sigmoid(torch.randn((channels,), dtype=torch.float32))
        expected_f = _reference_scan(u, decay)
        got_f = _MambaLikeBlock._parallel_associative_scan(u, decay)
        assert torch.allclose(got_f, expected_f, atol=1e-6, rtol=1e-5)

        expected_b = torch.flip(_reference_scan(torch.flip(u, dims=(0,)), decay), dims=(0,))
        got_b = torch.flip(_MambaLikeBlock._parallel_associative_scan(torch.flip(u, dims=(0,)), decay), dims=(0,))
        assert torch.allclose(got_b, expected_b, atol=1e-6, rtol=1e-5)


def test_mamba_like_block_uses_future_context() -> None:
    torch.manual_seed(0)
    hidden_dim = 4
    block = _MambaLikeBlock(hidden_dim=hidden_dim).eval()
    with torch.no_grad():
        block.in_proj.weight.zero_()
        block.in_proj.bias.zero_()
        block.in_proj.weight.copy_(torch.eye(hidden_dim))

        block.gate_proj.weight.zero_()
        block.gate_proj.bias.fill_(10.0)

        block.out_proj.weight.zero_()
        block.out_proj.bias.zero_()
        block.out_proj.weight.copy_(torch.eye(hidden_dim))

        block.state_decay.data.fill_(10.0)

        for layer in block.ff:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()

    length = 8
    x = torch.zeros((length, hidden_dim), dtype=torch.float32)
    y0 = block(x)

    x2 = x.clone()
    x2[-1, 0] = 3.0
    y1 = block(x2)

    assert torch.allclose(y0[-1], y1[-1]) is False
    assert torch.allclose(y0[0], y1[0]) is False
