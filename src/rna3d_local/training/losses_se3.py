from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from ..errors import raise_error
from ..se3.geometry import build_ribose_like_frames


@dataclass(frozen=True)
class StructuralLossTerms:
    mse: torch.Tensor
    fape: torch.Tensor
    tm: torch.Tensor
    clash: torch.Tensor


def _validate_coords(*, x_pred: torch.Tensor, x_true: torch.Tensor, stage: str, location: str) -> None:
    if x_pred.ndim != 2 or x_pred.shape[-1] != 3:
        raise_error(stage, location, "x_pred com shape invalido para loss estrutural", impact="1", examples=[str(tuple(x_pred.shape))])
    if x_true.ndim != 2 or x_true.shape[-1] != 3:
        raise_error(stage, location, "x_true com shape invalido para loss estrutural", impact="1", examples=[str(tuple(x_true.shape))])
    if int(x_pred.shape[0]) != int(x_true.shape[0]):
        raise_error(
            stage,
            location,
            "x_pred e x_true com numero de residuos divergente",
            impact="1",
            examples=[f"pred={int(x_pred.shape[0])}", f"true={int(x_true.shape[0])}"],
        )
    if int(x_pred.shape[0]) <= 1:
        raise_error(stage, location, "loss estrutural requer ao menos 2 residuos", impact="1", examples=[str(int(x_pred.shape[0]))])


def _validate_node_features(*, node_features: torch.Tensor, n_res: int, stage: str, location: str) -> None:
    if node_features.ndim != 2:
        raise_error(stage, location, "node_features com shape invalido para loss estrutural", impact="1", examples=[str(tuple(node_features.shape))])
    if int(node_features.shape[0]) != int(n_res):
        raise_error(
            stage,
            location,
            "node_features com tamanho divergente das coordenadas",
            impact="1",
            examples=[f"coords={int(n_res)}", f"node_features={int(node_features.shape[0])}"],
        )
    if int(node_features.shape[1]) < 4:
        raise_error(
            stage,
            location,
            "node_features requer ao menos 4 canais de base para FAPE",
            impact="1",
            examples=[f"node_features_dim={int(node_features.shape[1])}"],
        )


def _kabsch_align(*, mobile: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    autocast_context = (
        torch.autocast(device_type=mobile.device.type, enabled=False)
        if mobile.device.type in {"cuda", "cpu"}
        else nullcontext()
    )
    with autocast_context:
        mobile_fp32 = mobile.to(dtype=torch.float32)
        target_fp32 = target.to(dtype=torch.float32)
        mobile_center = mobile_fp32 - mobile_fp32.mean(dim=0, keepdim=True)
        target_center = target_fp32 - target_fp32.mean(dim=0, keepdim=True)
        with torch.no_grad():
            cov = mobile_center.detach().transpose(0, 1) @ target_center.detach()
            u, _, vh = torch.linalg.svd(cov, full_matrices=False)
            v = vh.transpose(0, 1)
            d = torch.det(v @ u.transpose(0, 1))
            sign = torch.where(
                d < 0.0,
                torch.tensor(-1.0, dtype=torch.float32, device=mobile.device),
                torch.tensor(1.0, dtype=torch.float32, device=mobile.device),
            )
            correction = torch.eye(3, dtype=torch.float32, device=mobile.device)
            correction[2, 2] = sign
            rotation = v @ correction @ u.transpose(0, 1)
        return (mobile_center @ rotation.transpose(0, 1)) + target_fp32.mean(dim=0, keepdim=True)


def _fape_chunked(
    *,
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    node_features: torch.Tensor,
    chunk_size: int,
    clamp_distance: float,
    length_scale: float,
) -> torch.Tensor:
    base_features = node_features[:, :4].to(device=x_pred.device, dtype=x_pred.dtype)
    frame_pred = build_ribose_like_frames(x_pred, base_features=base_features).to(dtype=x_pred.dtype)
    frame_true = build_ribose_like_frames(x_true, base_features=base_features).to(dtype=x_true.dtype)
    count = int(x_pred.shape[0])
    total = x_pred.new_tensor(0.0)
    denom = x_pred.new_tensor(0.0)
    for start in range(0, count, int(chunk_size)):
        end = min(count, start + int(chunk_size))
        idx = torch.arange(start, end, dtype=torch.long, device=x_pred.device)
        pred_delta = x_pred.unsqueeze(0) - x_pred[idx].unsqueeze(1)
        true_delta = x_true.unsqueeze(0) - x_true[idx].unsqueeze(1)
        pred_local = torch.einsum("bij,bnj->bni", frame_pred[idx].transpose(1, 2), pred_delta)
        true_local = torch.einsum("bij,bnj->bni", frame_true[idx].transpose(1, 2), true_delta)
        dist = torch.linalg.norm(pred_local - true_local, dim=-1)
        clipped = torch.clamp(dist, max=float(clamp_distance))
        total = total + (clipped / float(length_scale)).sum()
        denom = denom + torch.tensor(float(clipped.numel()), dtype=x_pred.dtype, device=x_pred.device)
    return total / torch.clamp(denom, min=1.0)


def _tm_core_loss(*, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    aligned = _kabsch_align(mobile=x_pred, target=x_true)
    dist = torch.linalg.norm(aligned - x_true.to(dtype=torch.float32), dim=-1)
    length = int(dist.shape[0])
    if length <= 15:
        d0 = 0.5
    else:
        d0 = max(0.5, (1.24 * ((float(length - 15)) ** (1.0 / 3.0))) - 1.8)
    tm_score = torch.mean(1.0 / (1.0 + torch.square(dist / float(d0))))
    return 1.0 - tm_score


def _clash_loss_chunked(
    *,
    x_pred: torch.Tensor,
    chain_index: torch.Tensor,
    residue_index: torch.Tensor,
    chunk_size: int,
    min_distance: float,
    repulsion_power: int,
) -> torch.Tensor:
    count = int(x_pred.shape[0])
    index_all = torch.arange(count, dtype=torch.long, device=x_pred.device)
    total = x_pred.new_tensor(0.0)
    denom = x_pred.new_tensor(0.0)
    alpha = float(repulsion_power)
    critical_alpha = float(repulsion_power) * 2.5
    critical_distance = 2.0
    for start in range(0, count, int(chunk_size)):
        end = min(count, start + int(chunk_size))
        idx = torch.arange(start, end, dtype=torch.long, device=x_pred.device)
        delta = x_pred.unsqueeze(0) - x_pred[idx].unsqueeze(1)
        dist = torch.sqrt(torch.clamp(torch.sum(delta * delta, dim=-1), min=1e-8))
        same_chain = chain_index[idx].unsqueeze(1) == chain_index.unsqueeze(0)
        seq_gap = torch.abs(residue_index[idx].unsqueeze(1) - residue_index.unsqueeze(0))
        is_covalent = same_chain & (seq_gap <= 1)
        upper = idx.unsqueeze(1) < index_all.unsqueeze(0)
        mask = upper & (~is_covalent)
        if bool(mask.any()):
            pair_dist = dist[mask]
            penetration = torch.clamp(float(min_distance) - pair_dist, min=0.0)
            critical_penetration = torch.clamp(float(critical_distance) - pair_dist, min=0.0)
            active = (penetration > 0.0) | (critical_penetration > 0.0)
            if bool(active.any()):
                penalty_main = torch.expm1(torch.clamp(alpha * penetration[active], max=50.0))
                penalty_critical = torch.expm1(torch.clamp(critical_alpha * critical_penetration[active], max=50.0))
                total = total + torch.sum(penalty_main + penalty_critical)
                denom = denom + torch.tensor(float(active.sum().item()), dtype=x_pred.dtype, device=x_pred.device)
    if float(denom.item()) <= 0.0:
        return x_pred.new_tensor(0.0)
    return total / denom


def compute_structural_loss_terms(
    *,
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    node_features: torch.Tensor,
    chain_index: torch.Tensor,
    residue_index: torch.Tensor,
    fape_clamp_distance: float,
    fape_length_scale: float,
    vdw_min_distance: float,
    vdw_repulsion_power: int,
    loss_chunk_size: int,
    stage: str,
    location: str,
) -> StructuralLossTerms:
    _validate_coords(x_pred=x_pred, x_true=x_true, stage=stage, location=location)
    _validate_node_features(node_features=node_features, n_res=int(x_pred.shape[0]), stage=stage, location=location)
    if chain_index.ndim != 1 or residue_index.ndim != 1:
        raise_error(stage, location, "chain_index/residue_index invalidos para loss estrutural", impact="1", examples=[str(tuple(chain_index.shape)), str(tuple(residue_index.shape))])
    if int(chain_index.shape[0]) != int(x_pred.shape[0]) or int(residue_index.shape[0]) != int(x_pred.shape[0]):
        raise_error(
            stage,
            location,
            "chain_index/residue_index com tamanho divergente das coordenadas",
            impact="1",
            examples=[f"coords={int(x_pred.shape[0])}", f"chain={int(chain_index.shape[0])}", f"res={int(residue_index.shape[0])}"],
        )
    if float(fape_clamp_distance) <= 0.0:
        raise_error(stage, location, "fape_clamp_distance invalido", impact="1", examples=[str(fape_clamp_distance)])
    if float(fape_length_scale) <= 0.0:
        raise_error(stage, location, "fape_length_scale invalido", impact="1", examples=[str(fape_length_scale)])
    if float(vdw_min_distance) <= 0.0:
        raise_error(stage, location, "vdw_min_distance invalido", impact="1", examples=[str(vdw_min_distance)])
    if int(vdw_repulsion_power) <= 1:
        raise_error(stage, location, "vdw_repulsion_power deve ser > 1", impact="1", examples=[str(vdw_repulsion_power)])
    if int(loss_chunk_size) <= 0:
        raise_error(stage, location, "loss_chunk_size deve ser > 0", impact="1", examples=[str(loss_chunk_size)])

    mse = torch.mean(torch.square(x_pred - x_true))
    fape = _fape_chunked(
        x_pred=x_pred,
        x_true=x_true,
        node_features=node_features,
        chunk_size=int(loss_chunk_size),
        clamp_distance=float(fape_clamp_distance),
        length_scale=float(fape_length_scale),
    )
    tm = _tm_core_loss(x_pred=x_pred, x_true=x_true)
    clash = _clash_loss_chunked(
        x_pred=x_pred,
        chain_index=chain_index.to(dtype=torch.long),
        residue_index=residue_index.to(dtype=torch.long),
        chunk_size=int(loss_chunk_size),
        min_distance=float(vdw_min_distance),
        repulsion_power=int(vdw_repulsion_power),
    )
    return StructuralLossTerms(mse=mse, fape=fape, tm=tm, clash=clash)
