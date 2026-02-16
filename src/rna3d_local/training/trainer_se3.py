from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from ..errors import raise_error
from ..generative.diffusion_se3 import Se3Diffusion
from ..generative.flow_matching_se3 import Se3FlowMatching
from ..se3.egnn_backbone import EgnnBackbone
from ..se3.fusion import Se3Fusion
from ..se3.ipa_backbone import IpaBackbone
from ..se3.sequence_tower import SequenceTower
from ..utils import rel_or_abs, sha256_file, utc_now_iso, write_json
from .config_se3 import Se3TrainConfig, load_se3_train_config
from .dataset_se3 import load_training_graphs
from .losses_se3 import compute_structural_loss_terms


@dataclass(frozen=True)
class TrainSe3Result:
    model_dir: Path
    manifest_path: Path
    metrics_path: Path
    config_effective_path: Path


@dataclass(frozen=True)
class Se3RuntimeModels:
    config: Se3TrainConfig
    input_dim: int
    sequence_tower: SequenceTower
    egnn: EgnnBackbone
    ipa: IpaBackbone
    fusion: Se3Fusion
    coarse_head: nn.Linear
    diffusion: Se3Diffusion | None
    flow: Se3FlowMatching | None


def _model_paths(model_dir: Path) -> dict[str, Path]:
    return {
        "sequence_tower": model_dir / "sequence_tower.pt",
        "egnn": model_dir / "backbone_egnn.pt",
        "ipa": model_dir / "backbone_ipa.pt",
        "fusion": model_dir / "backbone_fusion.pt",
        "coarse_head": model_dir / "coarse_head.pt",
        "diffusion": model_dir / "generator_diffusion.pt",
        "flow": model_dir / "generator_flow.pt",
        "config_effective": model_dir / "config_effective.json",
        "metrics": model_dir / "metrics.json",
        "manifest": model_dir / "train_manifest.json",
    }


def _forward_backbone(
    *,
    sequence_tower: SequenceTower,
    egnn: EgnnBackbone,
    ipa: IpaBackbone,
    fusion: Se3Fusion,
    coarse_head: nn.Linear,
    node_features: torch.Tensor,
    coords_init: torch.Tensor,
    bpp_pair_src: torch.Tensor,
    bpp_pair_dst: torch.Tensor,
    bpp_pair_prob: torch.Tensor,
    msa_pair_src: torch.Tensor,
    msa_pair_dst: torch.Tensor,
    msa_pair_prob: torch.Tensor,
    residue_index: torch.Tensor,
    chain_index: torch.Tensor,
    chem_exposure: torch.Tensor,
    chain_break_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_seq = sequence_tower(node_features)
    h_egnn, x_egnn = egnn(
        node_features=h_seq,
        coords=coords_init,
        bpp_pair_src=bpp_pair_src,
        bpp_pair_dst=bpp_pair_dst,
        bpp_pair_prob=bpp_pair_prob,
        msa_pair_src=msa_pair_src,
        msa_pair_dst=msa_pair_dst,
        msa_pair_prob=msa_pair_prob,
        residue_index=residue_index,
        chain_index=chain_index,
        chem_exposure=chem_exposure,
        chain_break_offset=chain_break_offset,
    )
    h_ipa, x_ipa = ipa(
        node_features=h_seq,
        coords=coords_init,
        bpp_pair_src=bpp_pair_src,
        bpp_pair_dst=bpp_pair_dst,
        bpp_pair_prob=bpp_pair_prob,
        msa_pair_src=msa_pair_src,
        msa_pair_dst=msa_pair_dst,
        msa_pair_prob=msa_pair_prob,
        residue_index=residue_index,
        chain_index=chain_index,
        chem_exposure=chem_exposure,
        chain_break_offset=chain_break_offset,
    )
    h_fused, x_fused = fusion(h_egnn=h_egnn, x_egnn=x_egnn, h_ipa=h_ipa, x_ipa=x_ipa)
    x_coarse = x_fused + coarse_head(h_fused)
    return h_fused, x_coarse


def train_se3_generator(
    *,
    repo_root: Path,
    targets_path: Path,
    pairings_path: Path,
    chemical_features_path: Path,
    labels_path: Path,
    config_path: Path,
    out_dir: Path,
    seed: int,
) -> TrainSe3Result:
    stage = "TRAIN_SE3"
    location = "src/rna3d_local/training/trainer_se3.py:train_se3_generator"
    cfg = load_se3_train_config(config_path, stage=stage, location=location)
    graphs = load_training_graphs(
        targets_path=targets_path,
        pairings_path=pairings_path,
        chemical_features_path=chemical_features_path,
        labels_path=labels_path,
        thermo_backend=cfg.thermo_backend,
        rnafold_bin=cfg.rnafold_bin,
        linearfold_bin=cfg.linearfold_bin,
        thermo_cache_dir=(None if cfg.thermo_cache_dir is None else (repo_root / cfg.thermo_cache_dir).resolve()),
        msa_backend=cfg.msa_backend,
        mmseqs_bin=cfg.mmseqs_bin,
        mmseqs_db=cfg.mmseqs_db,
        msa_cache_dir=(None if cfg.msa_cache_dir is None else (repo_root / cfg.msa_cache_dir).resolve()),
        chain_separator=cfg.chain_separator,
        max_msa_sequences=cfg.max_msa_sequences,
        max_cov_positions=cfg.max_cov_positions,
        max_cov_pairs=cfg.max_cov_pairs,
        stage=stage,
        location=location,
    )
    input_dim = int(graphs[0].node_features.shape[1])
    if input_dim <= 0:
        raise_error(stage, location, "input_dim invalido para treino", impact="1", examples=[str(input_dim)])

    torch.manual_seed(int(seed))
    sequence_tower = SequenceTower(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        tower_type=cfg.sequence_tower,
        heads=cfg.sequence_heads,
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
    )
    egnn = EgnnBackbone(
        input_dim=cfg.hidden_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        graph_backend=cfg.graph_backend,
        radius_angstrom=cfg.radius_angstrom,
        max_neighbors=cfg.max_neighbors,
        graph_chunk_size=cfg.graph_chunk_size,
        stage=stage,
        location=location,
    )
    ipa = IpaBackbone(
        input_dim=cfg.hidden_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        heads=cfg.ipa_heads,
        graph_backend=cfg.graph_backend,
        radius_angstrom=cfg.radius_angstrom,
        max_neighbors=cfg.max_neighbors,
        graph_chunk_size=cfg.graph_chunk_size,
        stage=stage,
        location=location,
    )
    fusion = Se3Fusion(hidden_dim=cfg.hidden_dim)
    coarse_head = nn.Linear(cfg.hidden_dim, 3)
    diffusion = Se3Diffusion(hidden_dim=cfg.hidden_dim, num_steps=cfg.diffusion_steps) if cfg.method in {"diffusion", "both"} else None
    flow = Se3FlowMatching(hidden_dim=cfg.hidden_dim, num_steps=cfg.flow_steps) if cfg.method in {"flow", "both"} else None

    params = list(sequence_tower.parameters()) + list(egnn.parameters()) + list(ipa.parameters()) + list(fusion.parameters()) + list(coarse_head.parameters())
    if diffusion is not None:
        params.extend(diffusion.parameters())
    if flow is not None:
        params.extend(flow.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)

    loss_trace: list[float] = []
    loss_mse_trace: list[float] = []
    loss_fape_trace: list[float] = []
    loss_tm_trace: list[float] = []
    loss_clash_trace: list[float] = []
    loss_generative_trace: list[float] = []
    for _ in range(cfg.epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_fape = 0.0
        epoch_tm = 0.0
        epoch_clash = 0.0
        epoch_generative = 0.0
        for graph in graphs:
            if graph.coords_true is None:
                raise_error(stage, location, "coords_true ausente no treino", impact="1", examples=[graph.target_id])
            optimizer.zero_grad(set_to_none=True)
            h_fused, x_coarse = _forward_backbone(
                sequence_tower=sequence_tower,
                egnn=egnn,
                ipa=ipa,
                fusion=fusion,
                coarse_head=coarse_head,
                node_features=graph.node_features,
                coords_init=graph.coords_init,
                bpp_pair_src=graph.bpp_pair_src,
                bpp_pair_dst=graph.bpp_pair_dst,
                bpp_pair_prob=graph.bpp_pair_prob,
                msa_pair_src=graph.msa_pair_src,
                msa_pair_dst=graph.msa_pair_dst,
                msa_pair_prob=graph.msa_pair_prob,
                residue_index=graph.residue_index,
                chain_index=graph.chain_index,
                chem_exposure=graph.chem_exposure,
                chain_break_offset=cfg.chain_break_offset,
            )
            structural_terms = compute_structural_loss_terms(
                x_pred=x_coarse,
                x_true=graph.coords_true,
                chain_index=graph.chain_index,
                residue_index=graph.residue_index,
                fape_clamp_distance=cfg.fape_clamp_distance,
                fape_length_scale=cfg.fape_length_scale,
                vdw_min_distance=cfg.vdw_min_distance,
                vdw_repulsion_power=cfg.vdw_repulsion_power,
                loss_chunk_size=cfg.loss_chunk_size,
                stage=stage,
                location=location,
            )
            loss_structural = (
                (float(cfg.loss_weight_mse) * structural_terms.mse)
                + (float(cfg.loss_weight_fape) * structural_terms.fape)
                + (float(cfg.loss_weight_tm) * structural_terms.tm)
                + (float(cfg.loss_weight_clash) * structural_terms.clash)
            )
            loss_generative = x_coarse.new_tensor(0.0)
            if diffusion is not None:
                loss_generative = loss_generative + diffusion.forward_loss(h=h_fused, x_cond=x_coarse, x_true=graph.coords_true)
            if flow is not None:
                loss_generative = loss_generative + flow.forward_loss(h=h_fused, x_cond=x_coarse, x_true=graph.coords_true)
            loss = loss_structural + loss_generative
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            epoch_mse += float(structural_terms.mse.detach().cpu().item())
            epoch_fape += float(structural_terms.fape.detach().cpu().item())
            epoch_tm += float(structural_terms.tm.detach().cpu().item())
            epoch_clash += float(structural_terms.clash.detach().cpu().item())
            epoch_generative += float(loss_generative.detach().cpu().item())
        loss_trace.append(epoch_loss / float(len(graphs)))
        loss_mse_trace.append(epoch_mse / float(len(graphs)))
        loss_fape_trace.append(epoch_fape / float(len(graphs)))
        loss_tm_trace.append(epoch_tm / float(len(graphs)))
        loss_clash_trace.append(epoch_clash / float(len(graphs)))
        loss_generative_trace.append(epoch_generative / float(len(graphs)))

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _model_paths(out_dir)
    torch.save(sequence_tower.state_dict(), paths["sequence_tower"])
    torch.save(egnn.state_dict(), paths["egnn"])
    torch.save(ipa.state_dict(), paths["ipa"])
    torch.save(fusion.state_dict(), paths["fusion"])
    torch.save(coarse_head.state_dict(), paths["coarse_head"])
    if diffusion is not None:
        torch.save(diffusion.state_dict(), paths["diffusion"])
    if flow is not None:
        torch.save(flow.state_dict(), paths["flow"])

    effective_config_payload = {
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "ipa_heads": cfg.ipa_heads,
        "diffusion_steps": cfg.diffusion_steps,
        "flow_steps": cfg.flow_steps,
        "epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "method": cfg.method,
        "input_dim": input_dim,
        "sequence_tower": cfg.sequence_tower,
        "sequence_heads": cfg.sequence_heads,
        "use_gradient_checkpointing": bool(cfg.use_gradient_checkpointing),
        "graph_backend": cfg.graph_backend,
        "radius_angstrom": cfg.radius_angstrom,
        "max_neighbors": cfg.max_neighbors,
        "graph_chunk_size": cfg.graph_chunk_size,
        "thermo_backend": cfg.thermo_backend,
        "rnafold_bin": cfg.rnafold_bin,
        "linearfold_bin": cfg.linearfold_bin,
        "thermo_cache_dir": cfg.thermo_cache_dir,
        "msa_backend": cfg.msa_backend,
        "mmseqs_bin": cfg.mmseqs_bin,
        "mmseqs_db": cfg.mmseqs_db,
        "msa_cache_dir": cfg.msa_cache_dir,
        "chain_separator": cfg.chain_separator,
        "chain_break_offset": cfg.chain_break_offset,
        "max_msa_sequences": cfg.max_msa_sequences,
        "max_cov_positions": cfg.max_cov_positions,
        "max_cov_pairs": cfg.max_cov_pairs,
        "loss_weight_mse": cfg.loss_weight_mse,
        "loss_weight_fape": cfg.loss_weight_fape,
        "loss_weight_tm": cfg.loss_weight_tm,
        "loss_weight_clash": cfg.loss_weight_clash,
        "fape_clamp_distance": cfg.fape_clamp_distance,
        "fape_length_scale": cfg.fape_length_scale,
        "vdw_min_distance": cfg.vdw_min_distance,
        "vdw_repulsion_power": cfg.vdw_repulsion_power,
        "loss_chunk_size": cfg.loss_chunk_size,
        "seed": int(seed),
    }
    write_json(paths["config_effective"], effective_config_payload)
    write_json(
        paths["metrics"],
        {
            "created_utc": utc_now_iso(),
            "n_targets": len(graphs),
            "epochs": cfg.epochs,
            "loss_final": loss_trace[-1],
            "loss_mse_final": loss_mse_trace[-1],
            "loss_fape_final": loss_fape_trace[-1],
            "loss_tm_final": loss_tm_trace[-1],
            "loss_clash_final": loss_clash_trace[-1],
            "loss_generative_final": loss_generative_trace[-1],
            "loss_trace": [float(item) for item in loss_trace],
            "loss_mse_trace": [float(item) for item in loss_mse_trace],
            "loss_fape_trace": [float(item) for item in loss_fape_trace],
            "loss_tm_trace": [float(item) for item in loss_tm_trace],
            "loss_clash_trace": [float(item) for item in loss_clash_trace],
            "loss_generative_trace": [float(item) for item in loss_generative_trace],
        },
    )
    chemical_source_counts: dict[str, int] = {}
    for graph in graphs:
        source = str(graph.chem_source)
        chemical_source_counts[source] = int(chemical_source_counts.get(source, 0) + 1)
    manifest_payload = {
        "created_utc": utc_now_iso(),
        "paths": {
            "targets": rel_or_abs(targets_path, repo_root),
            "pairings": rel_or_abs(pairings_path, repo_root),
            "chemical_features": rel_or_abs(chemical_features_path, repo_root),
            "labels": rel_or_abs(labels_path, repo_root),
            "config": rel_or_abs(config_path, repo_root),
            "model_dir": rel_or_abs(out_dir, repo_root),
        },
        "params": effective_config_payload,
        "stats": {
            "n_targets": len(graphs),
            "n_residues_total": int(sum(len(item.resids) for item in graphs)),
            "chemical_mapping_source_counts": chemical_source_counts,
        },
        "sha256": {
            "sequence_tower.pt": sha256_file(paths["sequence_tower"]),
            "backbone_egnn.pt": sha256_file(paths["egnn"]),
            "backbone_ipa.pt": sha256_file(paths["ipa"]),
            "backbone_fusion.pt": sha256_file(paths["fusion"]),
            "coarse_head.pt": sha256_file(paths["coarse_head"]),
            "generator_diffusion.pt": (sha256_file(paths["diffusion"]) if paths["diffusion"].exists() else None),
            "generator_flow.pt": (sha256_file(paths["flow"]) if paths["flow"].exists() else None),
            "config_effective.json": sha256_file(paths["config_effective"]),
            "metrics.json": sha256_file(paths["metrics"]),
        },
    }
    write_json(paths["manifest"], manifest_payload)
    return TrainSe3Result(
        model_dir=out_dir,
        manifest_path=paths["manifest"],
        metrics_path=paths["metrics"],
        config_effective_path=paths["config_effective"],
    )


def load_se3_runtime_models(*, model_dir: Path, stage: str, location: str) -> Se3RuntimeModels:
    paths = _model_paths(model_dir)
    for required in ["sequence_tower", "egnn", "ipa", "fusion", "coarse_head", "config_effective"]:
        path = paths[required]
        if not path.exists():
            raise_error(stage, location, "arquivo de runtime se3 ausente", impact="1", examples=[str(path)])
    cfg = load_se3_train_config(paths["config_effective"], stage=stage, location=location)
    payload = json.loads(paths["config_effective"].read_text(encoding="utf-8"))
    input_dim = int(payload.get("input_dim", 0))
    if input_dim <= 0:
        raise_error(stage, location, "config_effective sem input_dim valido", impact="1", examples=[str(input_dim)])

    sequence_tower = SequenceTower(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        tower_type=cfg.sequence_tower,
        heads=cfg.sequence_heads,
        use_gradient_checkpointing=False,
    )
    egnn = EgnnBackbone(
        input_dim=cfg.hidden_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        graph_backend=cfg.graph_backend,
        radius_angstrom=cfg.radius_angstrom,
        max_neighbors=cfg.max_neighbors,
        graph_chunk_size=cfg.graph_chunk_size,
        stage=stage,
        location=location,
    )
    ipa = IpaBackbone(
        input_dim=cfg.hidden_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        heads=cfg.ipa_heads,
        graph_backend=cfg.graph_backend,
        radius_angstrom=cfg.radius_angstrom,
        max_neighbors=cfg.max_neighbors,
        graph_chunk_size=cfg.graph_chunk_size,
        stage=stage,
        location=location,
    )
    fusion = Se3Fusion(hidden_dim=cfg.hidden_dim)
    coarse_head = nn.Linear(cfg.hidden_dim, 3)
    sequence_tower.load_state_dict(torch.load(paths["sequence_tower"], map_location="cpu"))
    egnn.load_state_dict(torch.load(paths["egnn"], map_location="cpu"))
    ipa.load_state_dict(torch.load(paths["ipa"], map_location="cpu"))
    fusion.load_state_dict(torch.load(paths["fusion"], map_location="cpu"))
    coarse_head.load_state_dict(torch.load(paths["coarse_head"], map_location="cpu"))
    diffusion = None
    if cfg.method in {"diffusion", "both"}:
        if not paths["diffusion"].exists():
            raise_error(stage, location, "generator_diffusion ausente no runtime", impact="1", examples=[str(paths["diffusion"])])
        diffusion = Se3Diffusion(hidden_dim=cfg.hidden_dim, num_steps=cfg.diffusion_steps)
        diffusion.load_state_dict(torch.load(paths["diffusion"], map_location="cpu"))
        diffusion.eval()
    flow = None
    if cfg.method in {"flow", "both"}:
        if not paths["flow"].exists():
            raise_error(stage, location, "generator_flow ausente no runtime", impact="1", examples=[str(paths["flow"])])
        flow = Se3FlowMatching(hidden_dim=cfg.hidden_dim, num_steps=cfg.flow_steps)
        flow.load_state_dict(torch.load(paths["flow"], map_location="cpu"))
        flow.eval()
    egnn.eval()
    ipa.eval()
    fusion.eval()
    coarse_head.eval()
    sequence_tower.eval()
    return Se3RuntimeModels(
        config=cfg,
        input_dim=input_dim,
        sequence_tower=sequence_tower,
        egnn=egnn,
        ipa=ipa,
        fusion=fusion,
        coarse_head=coarse_head,
        diffusion=diffusion,
        flow=flow,
    )


def run_backbone_for_graph(
    *,
    runtime: Se3RuntimeModels,
    node_features: torch.Tensor,
    coords_init: torch.Tensor,
    bpp_pair_src: torch.Tensor,
    bpp_pair_dst: torch.Tensor,
    bpp_pair_prob: torch.Tensor,
    msa_pair_src: torch.Tensor,
    msa_pair_dst: torch.Tensor,
    msa_pair_prob: torch.Tensor,
    residue_index: torch.Tensor,
    chain_index: torch.Tensor,
    chem_exposure: torch.Tensor,
    chain_break_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _forward_backbone(
        sequence_tower=runtime.sequence_tower,
        egnn=runtime.egnn,
        ipa=runtime.ipa,
        fusion=runtime.fusion,
        coarse_head=runtime.coarse_head,
        node_features=node_features,
        coords_init=coords_init,
        bpp_pair_src=bpp_pair_src,
        bpp_pair_dst=bpp_pair_dst,
        bpp_pair_prob=bpp_pair_prob,
        msa_pair_src=msa_pair_src,
        msa_pair_dst=msa_pair_dst,
        msa_pair_prob=msa_pair_prob,
        residue_index=residue_index,
        chain_index=chain_index,
        chem_exposure=chem_exposure,
        chain_break_offset=chain_break_offset,
    )
