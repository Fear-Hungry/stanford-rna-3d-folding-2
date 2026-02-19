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


class _EquivariantLocalMessageDenoiser(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        edge_hidden = max(16, int(hidden_dim) // 2)
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.src_gate = nn.Linear(hidden_dim, 1)
        self.dst_gate = nn.Linear(hidden_dim, 1)
        self.neighbor_scale = nn.Linear(hidden_dim, 1)
        self.edge_gate = nn.Sequential(
            nn.Linear(4, edge_hidden),
            nn.SiLU(),
            nn.Linear(edge_hidden, 1),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim + 3 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def _aggregate_local_messages(self, node_hidden: torch.Tensor, x_current: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        n_res = int(x_current.shape[0])
        if n_res <= 1:
            return torch.zeros((n_res, 3), device=x_current.device, dtype=x_current.dtype)

        rel = x_current.unsqueeze(0) - x_current.unsqueeze(1)
        rel_local = torch.einsum("nij,nmj->nmi", frames.transpose(1, 2), rel)
        dist = torch.linalg.norm(rel_local, dim=-1, keepdim=True)
        edge_feat = torch.cat([rel_local, dist], dim=-1)
        logits = self.src_gate(node_hidden).unsqueeze(1)
        logits = logits + self.dst_gate(node_hidden).unsqueeze(0)
        logits = logits + self.edge_gate(edge_feat)
        logits_2d = logits.squeeze(-1)

        mask_diag = torch.eye(n_res, device=x_current.device, dtype=torch.bool)
        logits_2d = logits_2d.masked_fill(mask_diag, -1e4)
        attn = torch.softmax(logits_2d, dim=1)
        attn = attn.masked_fill(mask_diag, 0.0)
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        neighbor_scale = torch.sigmoid(self.neighbor_scale(node_hidden)).squeeze(-1)
        return torch.einsum("nm,m,nmi->ni", attn, neighbor_scale, rel_local)

    def forward(
        self,
        h: torch.Tensor,
        delta_local: torch.Tensor,
        t_emb: torch.Tensor,
        x_current: torch.Tensor,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        node_feat = torch.cat([h, delta_local, t_emb], dim=-1)
        node_hidden = self.node_proj(node_feat)
        local_messages = self._aggregate_local_messages(node_hidden=node_hidden, x_current=x_current, frames=frames)
        update_feat = torch.cat([node_hidden, delta_local, local_messages], dim=-1)
        local_update = self.out(update_feat)
        # Keep an explicit geometric residual so the denoiser cannot collapse to purely pointwise behavior.
        return local_update + (0.25 * local_messages)


class Se3Diffusion(nn.Module):
    def __init__(self, hidden_dim: int, num_steps: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_steps = int(num_steps)
        if self.num_steps <= 1:
            raise ValueError("num_steps must be > 1")
        self.denoiser = _EquivariantLocalMessageDenoiser(hidden_dim=hidden_dim)
        betas = torch.linspace(1e-4, 0.02, self.num_steps)
        alphas = 1.0 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hat", alpha_hat)

    def _predict_noise(self, h: torch.Tensor, delta_noisy: torch.Tensor, x_cond: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        geom_dtype = torch.float32
        frames = _build_equivariant_frames(x_cond.to(dtype=geom_dtype)).to(dtype=geom_dtype, device=x_cond.device)
        delta_noisy_f = delta_noisy.to(dtype=geom_dtype)
        x_current = x_cond.to(dtype=geom_dtype) + delta_noisy_f
        delta_local = torch.einsum("nij,nj->ni", frames.transpose(1, 2), delta_noisy_f)
        t_emb = _time_embedding(t_scalar).expand(delta_noisy.shape[0], -1).to(device=h.device, dtype=geom_dtype)
        h_feat = h.to(dtype=geom_dtype)
        noise_local = self.denoiser(
            h=h_feat,
            delta_local=delta_local,
            t_emb=t_emb,
            x_current=x_current,
            frames=frames,
        )
        noise_global = torch.einsum("nij,nj->ni", frames, noise_local)
        return noise_global.to(dtype=delta_noisy.dtype)

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
