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


def _safe_cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cross(a, b, dim=-1)


def _safe_direction(primary: torch.Tensor, fallback: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    out = primary.clone()
    bad = (_safe_norm(out, eps=eps) < eps).squeeze(-1)
    if bool(bad.any()):
        out[bad] = fallback[bad]
    return normalize_vector(out, eps=eps)


def infer_purine_mask(base_features: torch.Tensor) -> torch.Tensor:
    if base_features.ndim != 2 or int(base_features.shape[-1]) < 4:
        raise ValueError(f"base_features shape invalido para infer_purine_mask: {tuple(base_features.shape)}")
    base_code = torch.argmax(base_features[:, :4], dim=-1)
    return (base_code == 0) | (base_code == 2)


def build_rna_local_frames(
    *,
    c1_coords: torch.Tensor,
    base_features: torch.Tensor,
    p_distance: float = 2.2,
    c4_distance: float = 1.5,
    n_distance: float = 1.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if c1_coords.ndim != 2 or int(c1_coords.shape[-1]) != 3:
        raise ValueError(f"c1_coords com shape invalido: {tuple(c1_coords.shape)}")
    if base_features.ndim != 2 or int(base_features.shape[0]) != int(c1_coords.shape[0]) or int(base_features.shape[-1]) < 4:
        raise ValueError(f"base_features com shape invalido: {tuple(base_features.shape)}")
    count = int(c1_coords.shape[0])
    if count <= 0:
        raise ValueError("c1_coords vazio para build_rna_local_frames")
    if count == 1:
        eye = torch.eye(3, dtype=c1_coords.dtype, device=c1_coords.device).unsqueeze(0)
        p_proxy = c1_coords + torch.tensor([[-float(p_distance), 0.0, 0.0]], dtype=c1_coords.dtype, device=c1_coords.device)
        c4_proxy = c1_coords + torch.tensor([[0.0, float(c4_distance), 0.0]], dtype=c1_coords.dtype, device=c1_coords.device)
        purine_mask = infer_purine_mask(base_features=base_features)
        n_sign = torch.where(purine_mask, torch.ones((1,), dtype=c1_coords.dtype, device=c1_coords.device), -torch.ones((1,), dtype=c1_coords.dtype, device=c1_coords.device))
        n_proxy = c1_coords + torch.stack([torch.zeros_like(n_sign), torch.zeros_like(n_sign), n_sign * float(n_distance)], dim=-1)
        return eye, p_proxy, c4_proxy, n_proxy

    prev_vec = torch.zeros_like(c1_coords)
    next_vec = torch.zeros_like(c1_coords)
    prev_vec[1:] = c1_coords[1:] - c1_coords[:-1]
    prev_vec[0] = c1_coords[1] - c1_coords[0]
    next_vec[:-1] = c1_coords[1:] - c1_coords[:-1]
    next_vec[-1] = c1_coords[-1] - c1_coords[-2]

    backbone_dir = _safe_direction(prev_vec + next_vec, fallback=next_vec)
    plane_normal = _safe_cross(prev_vec, next_vec)
    z_ref = torch.tensor([0.0, 0.0, 1.0], dtype=c1_coords.dtype, device=c1_coords.device).expand_as(backbone_dir)
    y_ref = torch.tensor([0.0, 1.0, 0.0], dtype=c1_coords.dtype, device=c1_coords.device).expand_as(backbone_dir)
    normal_dir = _safe_direction(plane_normal, fallback=_safe_cross(backbone_dir, z_ref))
    normal_dir = _safe_direction(normal_dir, fallback=_safe_cross(backbone_dir, y_ref))
    sugar_dir = _safe_direction(_safe_cross(backbone_dir, normal_dir), fallback=normal_dir)

    purine_mask = infer_purine_mask(base_features=base_features)
    n_sign = torch.where(purine_mask, torch.ones_like(backbone_dir[:, 0]), -torch.ones_like(backbone_dir[:, 0]))
    n_axis = normal_dir * n_sign.unsqueeze(-1)

    p_proxy = c1_coords - (backbone_dir * float(p_distance))
    c4_proxy = c1_coords + (sugar_dir * float(c4_distance))
    n_proxy = c1_coords + (n_axis * float(n_distance))

    axis_x = _safe_direction(c4_proxy - p_proxy, fallback=backbone_dir)
    axis_y_raw = n_proxy - p_proxy
    axis_y = axis_y_raw - (axis_x * torch.sum(axis_y_raw * axis_x, dim=-1, keepdim=True))
    axis_y = _safe_direction(axis_y, fallback=sugar_dir)
    axis_z = _safe_direction(_safe_cross(axis_x, axis_y), fallback=normal_dir)
    axis_y = _safe_direction(_safe_cross(axis_z, axis_x), fallback=axis_y)
    # Convention: columns are local basis vectors in global coordinates (local->global).
    # This matches rotation_matrix_from_6d() and downstream einsums in ipa_backbone/losses_se3.
    frames = torch.stack([axis_x, axis_y, axis_z], dim=-1)
    return frames, p_proxy, c4_proxy, n_proxy


def build_ribose_like_frames(x: torch.Tensor, base_features: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or int(x.shape[-1]) != 3:
        raise ValueError(f"x com shape invalido para build_ribose_like_frames: {tuple(x.shape)}")
    if int(x.shape[0]) == 0:
        raise ValueError("x vazio para build_ribose_like_frames")
    if base_features.ndim != 2 or int(base_features.shape[0]) != int(x.shape[0]) or int(base_features.shape[-1]) < 4:
        raise ValueError(f"base_features com shape invalido para build_ribose_like_frames: {tuple(base_features.shape)}")
    frames, _p_proxy, _c4_proxy, _n_proxy = build_rna_local_frames(
        c1_coords=x,
        base_features=base_features.to(device=x.device, dtype=x.dtype),
    )
    return frames


def rotation_matrix_from_6d(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.ndim != 2 or int(x.shape[-1]) != 6:
        raise ValueError(f"x com shape invalido para rotation_matrix_from_6d: {tuple(x.shape)}")
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]
    b1 = normalize_vector(a1, eps=eps)
    proj = torch.sum(b1 * a2, dim=-1, keepdim=True)
    b2 = normalize_vector(a2 - (proj * b1), eps=eps)
    b3 = normalize_vector(_safe_cross(b1, b2), eps=eps)
    # columns = [b1 b2 b3] to match frame convention used in ipa_backbone (local->global).
    return torch.stack([b1, b2, b3], dim=-1)
