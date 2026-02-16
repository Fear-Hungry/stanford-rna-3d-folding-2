from __future__ import annotations

import torch
from torch import nn


def _time_embedding(t: torch.Tensor) -> torch.Tensor:
    value = t.float().unsqueeze(-1)
    return torch.cat([value, torch.sin(value), torch.cos(value)], dim=-1)


class Se3Diffusion(nn.Module):
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
        betas = torch.linspace(1e-4, 0.02, self.num_steps)
        alphas = 1.0 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hat", alpha_hat)

    def _predict_noise(self, h: torch.Tensor, x_noisy: torch.Tensor, x_cond: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        t_emb = _time_embedding(t_scalar).expand(x_noisy.shape[0], -1)
        feat = torch.cat([h, x_noisy, x_cond, t_emb], dim=-1)
        return self.net(feat)

    def forward_loss(self, h: torch.Tensor, x_cond: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        device = x_true.device
        t = torch.randint(0, self.num_steps, (1,), device=device)
        noise = torch.randn_like(x_true)
        ah = self.alpha_hat[t].view(1, 1)
        x_noisy = torch.sqrt(ah) * x_true + torch.sqrt(1.0 - ah) * noise
        t_norm = t.float() / float(self.num_steps - 1)
        noise_pred = self._predict_noise(h, x_noisy, x_cond, t_norm)
        return torch.mean((noise_pred - noise) ** 2)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, x_cond: torch.Tensor, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=x_cond.device)
        generator.manual_seed(int(seed))
        x = torch.randn(x_cond.shape, generator=generator, device=x_cond.device, dtype=x_cond.dtype)
        for idx in reversed(range(self.num_steps)):
            t_tensor = torch.tensor([idx / float(self.num_steps - 1)], device=x_cond.device, dtype=x_cond.dtype)
            noise_pred = self._predict_noise(h, x, x_cond, t_tensor)
            alpha = self.alphas[idx]
            alpha_hat = self.alpha_hat[idx]
            beta = self.betas[idx]
            x = (1.0 / torch.sqrt(alpha)) * (x - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_hat)) * noise_pred)
            if idx > 0:
                noise = torch.randn(x.shape, generator=generator, device=x_cond.device, dtype=x_cond.dtype)
                x = x + torch.sqrt(beta) * noise
        return x
