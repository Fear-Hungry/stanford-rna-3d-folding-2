from __future__ import annotations

import math

import torch

from rna3d_local.se3.fusion import Se3Fusion


def _rotation_z(angle_rad: float) -> torch.Tensor:
    c = float(math.cos(float(angle_rad)))
    s = float(math.sin(float(angle_rad)))
    return torch.tensor(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def test_se3_fusion_is_rotation_equivariant_for_coords() -> None:
    torch.manual_seed(0)
    n = 6
    hidden = 8
    model = Se3Fusion(hidden_dim=hidden).eval()

    # Non-degenerate chain-like coordinates to avoid frame fallbacks.
    base = torch.stack(
        [
            torch.linspace(0.0, 5.0, n),
            torch.linspace(0.0, 1.0, n),
            torch.linspace(0.0, 0.5, n),
        ],
        dim=-1,
    )
    x_egnn = base + 0.05 * torch.randn(n, 3)
    x_ipa = base + 0.05 * torch.randn(n, 3)
    h_egnn = torch.randn(n, hidden)
    h_ipa = torch.randn(n, hidden)

    h_out, x_out = model(h_egnn=h_egnn, x_egnn=x_egnn, h_ipa=h_ipa, x_ipa=x_ipa)
    assert h_out.shape == (n, hidden)
    assert x_out.shape == (n, 3)

    r = _rotation_z(0.7)
    x_egnn_r = x_egnn @ r.transpose(0, 1)
    x_ipa_r = x_ipa @ r.transpose(0, 1)
    _h_out_r, x_out_r = model(h_egnn=h_egnn, x_egnn=x_egnn_r, h_ipa=h_ipa, x_ipa=x_ipa_r)

    expected = x_out @ r.transpose(0, 1)
    assert torch.allclose(x_out_r, expected, atol=2e-4, rtol=2e-4)


def test_se3_fusion_is_translation_equivariant_for_coords() -> None:
    torch.manual_seed(0)
    n = 6
    hidden = 8
    model = Se3Fusion(hidden_dim=hidden).eval()

    base = torch.stack(
        [
            torch.linspace(0.0, 5.0, n),
            torch.linspace(0.0, 1.0, n),
            torch.linspace(0.0, 0.5, n),
        ],
        dim=-1,
    )
    x_egnn = base + 0.05 * torch.randn(n, 3)
    x_ipa = base + 0.05 * torch.randn(n, 3)
    h_egnn = torch.randn(n, hidden)
    h_ipa = torch.randn(n, hidden)

    _h_out, x_out = model(h_egnn=h_egnn, x_egnn=x_egnn, h_ipa=h_ipa, x_ipa=x_ipa)

    t = torch.tensor([[3.0, -2.0, 1.0]], dtype=torch.float32)
    _h_out_t, x_out_t = model(h_egnn=h_egnn, x_egnn=x_egnn + t, h_ipa=h_ipa, x_ipa=x_ipa + t)
    assert torch.allclose(x_out_t, x_out + t, atol=2e-4, rtol=2e-4)

