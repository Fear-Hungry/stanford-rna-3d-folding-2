from __future__ import annotations

import torch
from torch import nn


def _time_embedding(t: torch.Tensor) -> torch.Tensor:
    value = t.float().unsqueeze(-1)
    return torch.cat([value, torch.sin(value), torch.cos(value)], dim=-1)


def _safe_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp((x * x).sum(dim=-1, keepdim=True), min=float(eps)))


def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / _safe_norm(x, eps=eps)


def _safe_direction(primary: torch.Tensor, fallback: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    out = primary.clone()
    bad = (_safe_norm(out, eps=eps) < float(eps)).squeeze(-1)
    if bool(bad.any()):
        out[bad] = fallback[bad]
    return _normalize(out, eps=eps)


def _build_equivariant_frames(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or int(x.shape[-1]) != 3:
        raise ValueError(f"x com shape invalido para frames: {tuple(x.shape)}")
    # Use float64 for geometric ops to reduce numerical drift under large translations.
    x = x.to(dtype=torch.float64)
    count = int(x.shape[0])
    if count <= 0:
        raise ValueError("x vazio para frames")
    if count == 1:
        return torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0)

    prev = torch.zeros_like(x)
    nxt = torch.zeros_like(x)
    prev[1:] = x[1:] - x[:-1]
    prev[0] = x[1] - x[0]
    nxt[:-1] = x[1:] - x[:-1]
    nxt[-1] = x[-1] - x[-2]

    tangent = _safe_direction(prev + nxt, fallback=nxt)
    curvature = _safe_direction(nxt - prev, fallback=prev)

    # Use only translation-invariant vectors (differences) to keep exact translation equivariance.
    axis_y_raw = curvature - (tangent * torch.sum(curvature * tangent, dim=-1, keepdim=True))
    axis_y = _safe_direction(axis_y_raw, fallback=torch.cross(tangent, prev, dim=-1))
    axis_z = _safe_direction(torch.cross(tangent, axis_y, dim=-1), fallback=torch.cross(tangent, curvature, dim=-1))
    axis_y = _safe_direction(torch.cross(axis_z, tangent, dim=-1), fallback=axis_y)

    return torch.stack([tangent, axis_y, axis_z], dim=-1)


class Se3Diffusion(nn.Module):
    def __init__(self, hidden_dim: int, num_steps: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_steps = int(num_steps)
        if self.num_steps <= 1:
            raise ValueError("num_steps must be > 1")
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 3 + 3, hidden_dim),
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

    def _predict_noise(self, h: torch.Tensor, delta_noisy: torch.Tensor, x_cond: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        t_emb = _time_embedding(t_scalar).expand(delta_noisy.shape[0], -1)
        frames = _build_equivariant_frames(x_cond.to(dtype=torch.float32)).to(dtype=x_cond.dtype)
        delta_local = torch.einsum("nij,nj->ni", frames.transpose(1, 2), delta_noisy)
        feat = torch.cat([h, delta_local, t_emb], dim=-1)
        noise_local = self.net(feat)
        return torch.einsum("nij,nj->ni", frames, noise_local)

    def forward_loss(self, h: torch.Tensor, x_cond: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        device = x_true.device
        t = torch.randint(0, self.num_steps, (1,), device=device)
        delta_true = x_true - x_cond
        noise = torch.randn_like(delta_true)
        ah = self.alpha_hat[t].view(1, 1)
        delta_noisy = torch.sqrt(ah) * delta_true + torch.sqrt(1.0 - ah) * noise
        t_norm = t.float() / float(self.num_steps - 1)
        noise_pred = self._predict_noise(h, delta_noisy, x_cond, t_norm)
        return torch.mean((noise_pred - noise) ** 2)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, x_cond: torch.Tensor, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=x_cond.device)
        generator.manual_seed(int(seed))
        frames = _build_equivariant_frames(x_cond.to(dtype=torch.float32)).to(dtype=x_cond.dtype)
        noise_local = torch.randn(x_cond.shape, generator=generator, device=x_cond.device, dtype=x_cond.dtype)
        delta = torch.einsum("nij,nj->ni", frames, noise_local)
        for idx in reversed(range(self.num_steps)):
            t_tensor = torch.tensor([idx / float(self.num_steps - 1)], device=x_cond.device, dtype=x_cond.dtype)
            noise_pred = self._predict_noise(h, delta, x_cond, t_tensor)
            alpha = self.alphas[idx]
            alpha_hat = self.alpha_hat[idx]
            beta = self.betas[idx]
            delta = (1.0 / torch.sqrt(alpha)) * (delta - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_hat)) * noise_pred)
            if idx > 0:
                noise_local = torch.randn(delta.shape, generator=generator, device=x_cond.device, dtype=x_cond.dtype)
                noise = torch.einsum("nij,nj->ni", frames, noise_local)
                delta = delta + torch.sqrt(beta) * noise
        return x_cond + delta
