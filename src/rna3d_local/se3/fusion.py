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
        x = gate[:, :3] * x_egnn + (1.0 - gate[:, :3]) * x_ipa + self.x_proj(h)
        return h, x
