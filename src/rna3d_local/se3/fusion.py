from __future__ import annotations

import torch
from torch import nn


class Se3Fusion(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.h_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.x_proj = nn.Linear(hidden_dim, 3)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_egnn: torch.Tensor,
        x_egnn: torch.Tensor,
        h_ipa: torch.Tensor,
        x_ipa: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mix = torch.cat([h_egnn, h_ipa], dim=-1)
        gate = self.gate(mix)
        h = self.norm(gate * h_egnn + (1.0 - gate) * h_ipa + self.h_proj(mix))
        # Keep SE(3)-equivariance: use an isotropic (scalar) gate for coordinate mixing.
        # Avoid per-axis gating (non-equivariant) and avoid adding a learned global vector to coordinates.
        gate_x = gate[:, :1]
        x_mix = (gate_x * x_egnn) + ((1.0 - gate_x) * x_ipa)
        # Optional equivariant residual: scalar step along the (x_egnn - x_ipa) direction.
        diff = x_egnn - x_ipa
        diff_norm = torch.clamp(torch.linalg.norm(diff, dim=-1, keepdim=True), min=1e-6)
        diff_unit = diff / diff_norm
        step = 0.25 * torch.tanh(torch.linalg.norm(self.x_proj(h), dim=-1, keepdim=True))
        x = x_mix + (step * diff_unit)
        return h, x
