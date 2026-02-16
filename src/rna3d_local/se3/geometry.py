from __future__ import annotations

import torch


def _safe_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp((x * x).sum(dim=-1, keepdim=True), min=eps))


def normalize_vector(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / _safe_norm(x, eps=eps)


def center_coordinates(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def pairwise_distance_matrix(x: torch.Tensor) -> torch.Tensor:
    diff = x[:, None, :] - x[None, :, :]
    return torch.sqrt(torch.clamp((diff * diff).sum(dim=-1), min=0.0))


def build_ribose_like_frames(x: torch.Tensor) -> torch.Tensor:
    count = int(x.shape[0])
    if count < 2:
        eye = torch.eye(3, dtype=x.dtype, device=x.device)
        return eye[None, :, :].repeat(count, 1, 1)
    forward = torch.zeros_like(x)
    forward[:-1] = x[1:] - x[:-1]
    forward[-1] = x[-1] - x[-2]
    tangent = normalize_vector(forward)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=x.dtype, device=x.device).expand_as(tangent)
    normal = torch.cross(tangent, up, dim=-1)
    bad = (_safe_norm(normal) < 1e-6).squeeze(-1)
    if bool(bad.any()):
        alt = torch.tensor([0.0, 1.0, 0.0], dtype=x.dtype, device=x.device).expand_as(tangent[bad])
        normal[bad] = torch.cross(tangent[bad], alt, dim=-1)
    normal = normalize_vector(normal)
    binormal = normalize_vector(torch.cross(tangent, normal, dim=-1))
    return torch.stack([tangent, normal, binormal], dim=1)
