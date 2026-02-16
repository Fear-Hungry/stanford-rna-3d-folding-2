from __future__ import annotations

import torch
from torch import nn


def _time_embedding(t: torch.Tensor) -> torch.Tensor:
    value = t.float().unsqueeze(-1)
    return torch.cat([value, torch.sin(value), torch.cos(value)], dim=-1)


class Se3FlowMatching(nn.Module):
    def __init__(self, hidden_dim: int, num_steps: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_steps = int(num_steps)
        if self.num_steps <= 1:
            raise ValueError("num_steps must be > 1")
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 3 + 3 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def _predict_velocity(self, h: torch.Tensor, x_t: torch.Tensor, x_cond: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        t_emb = _time_embedding(tau).expand(x_t.shape[0], -1)
        feat = torch.cat([h, x_t, x_cond, t_emb], dim=-1)
        return self.net(feat)

    def forward_loss(self, h: torch.Tensor, x_cond: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        device = x_true.device
        tau = torch.rand((1,), device=device)
        noise = torch.randn_like(x_true) * 0.01
        x_t = ((1.0 - tau.view(1, 1)) * x_cond) + (tau.view(1, 1) * x_true) + noise
        vel_true = x_true - x_cond
        vel_pred = self._predict_velocity(h, x_t, x_cond, tau)
        return torch.mean((vel_pred - vel_true) ** 2)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, x_cond: torch.Tensor, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=x_cond.device)
        generator.manual_seed(int(seed))
        x = x_cond + (0.03 * torch.randn(x_cond.shape, generator=generator, device=x_cond.device, dtype=x_cond.dtype))
        dt = 1.0 / float(self.num_steps)
        for step in range(self.num_steps):
            tau = torch.tensor([step / float(self.num_steps - 1)], device=x_cond.device, dtype=x_cond.dtype)
            vel = self._predict_velocity(h, x, x_cond, tau)
            x = x + (dt * vel)
        return x
