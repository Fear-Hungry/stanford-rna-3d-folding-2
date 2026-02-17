from __future__ import annotations

import torch

from rna3d_local.se3.sequence_tower import _MambaLikeBlock


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

