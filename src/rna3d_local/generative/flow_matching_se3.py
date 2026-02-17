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

    axis_y_raw = curvature - (tangent * torch.sum(curvature * tangent, dim=-1, keepdim=True))
    axis_y = _safe_direction(axis_y_raw, fallback=torch.cross(tangent, prev, dim=-1))
    axis_z = _safe_direction(torch.cross(tangent, axis_y, dim=-1), fallback=torch.cross(tangent, curvature, dim=-1))
    axis_y = _safe_direction(torch.cross(axis_z, tangent, dim=-1), fallback=axis_y)

    return torch.stack([tangent, axis_y, axis_z], dim=-1)


class Se3FlowMatching(nn.Module):
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

    def _predict_velocity(self, h: torch.Tensor, x_t: torch.Tensor, x_cond: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        t_emb = _time_embedding(tau).expand(x_t.shape[0], -1)
        frames = _build_equivariant_frames(x_cond.to(dtype=torch.float32)).to(dtype=x_cond.dtype)
        delta = x_t - x_cond
        delta_local = torch.einsum("nij,nj->ni", frames.transpose(1, 2), delta)
        feat = torch.cat([h, delta_local, t_emb], dim=-1)
        vel_local = self.net(feat)
        return torch.einsum("nij,nj->ni", frames, vel_local)

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
        frames = _build_equivariant_frames(x_cond.to(dtype=torch.float32)).to(dtype=x_cond.dtype)
        noise_local = torch.randn(x_cond.shape, generator=generator, device=x_cond.device, dtype=x_cond.dtype)
        x = x_cond + (0.03 * torch.einsum("nij,nj->ni", frames, noise_local))
        dt = 1.0 / float(self.num_steps)
        for step in range(self.num_steps):
            tau = torch.tensor([step / float(self.num_steps - 1)], device=x_cond.device, dtype=x_cond.dtype)
            vel = self._predict_velocity(h, x, x_cond, tau)
            x = x + (dt * vel)
        return x
