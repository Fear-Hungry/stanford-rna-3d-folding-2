from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..errors import raise_error


@dataclass(frozen=True)
class Se3TrainConfig:
    hidden_dim: int
    num_layers: int
    ipa_heads: int
    diffusion_steps: int
    flow_steps: int
    epochs: int
    learning_rate: float
    method: str
    sequence_tower: str
    sequence_heads: int
    use_gradient_checkpointing: bool
    graph_backend: str
    radius_angstrom: float
    max_neighbors: int
    graph_chunk_size: int
    ipa_edge_chunk_size: int
    graph_pair_edges: str
    graph_pair_min_prob: float
    graph_pair_max_per_node: int
    thermo_pair_min_prob: float
    thermo_pair_max_per_node: int
    thermo_soft_constraint_strength: float
    thermo_backend: str
    rnafold_bin: str
    linearfold_bin: str
    thermo_cache_dir: str | None
    msa_backend: str
    mmseqs_bin: str
    mmseqs_db: str
    msa_cache_dir: str | None
    chain_separator: str
    chain_break_offset: int
    max_msa_sequences: int
    max_cov_positions: int
    max_cov_pairs: int
    loss_weight_mse: float
    loss_weight_fape: float
    loss_weight_tm: float
    loss_weight_clash: float
    fape_clamp_distance: float
    fape_length_scale: float
    vdw_min_distance: float
    vdw_repulsion_power: int
    loss_chunk_size: int
    dynamic_cropping: bool
    crop_min_length: int
    crop_max_length: int
    crop_sequence_fraction: float
    gradient_accumulation_steps: int
    autocast_bfloat16: bool
    training_protocol: str


def load_se3_train_config(path: Path, *, stage: str, location: str) -> Se3TrainConfig:
    if not path.exists():
        raise_error(stage, location, "config de treino ausente", impact="1", examples=[str(path)])
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ["hidden_dim", "num_layers", "ipa_heads", "diffusion_steps", "flow_steps", "epochs", "learning_rate", "method"]
    missing = [item for item in required if item not in payload]
    if missing:
        raise_error(stage, location, "config de treino sem campo obrigatorio", impact=str(len(missing)), examples=missing[:8])
    method = str(payload["method"]).strip().lower()
    if method not in {"diffusion", "flow", "both"}:
        raise_error(stage, location, "method invalido na config", impact="1", examples=[method])
    sequence_tower = str(payload.get("sequence_tower", "mamba_like")).strip().lower()
    if sequence_tower not in {"flash", "mamba_like"}:
        raise_error(stage, location, "sequence_tower invalido na config", impact="1", examples=[sequence_tower])
    graph_backend = str(payload.get("graph_backend", "torch_sparse")).strip().lower()
    if graph_backend not in {"torch_sparse", "torch_geometric"}:
        raise_error(stage, location, "graph_backend invalido na config", impact="1", examples=[graph_backend])
    graph_pair_edges = str(payload.get("graph_pair_edges", "none")).strip().lower()
    if graph_pair_edges not in {"none", "bpp"}:
        raise_error(stage, location, "graph_pair_edges invalido na config", impact="1", examples=[graph_pair_edges])
    graph_pair_min_prob = float(payload.get("graph_pair_min_prob", 0.10))
    graph_pair_max_per_node = int(payload.get("graph_pair_max_per_node", 8))
    if graph_pair_edges != "none":
        if not (0.0 <= float(graph_pair_min_prob) <= 1.0):
            raise_error(stage, location, "graph_pair_min_prob fora de [0,1]", impact="1", examples=[str(graph_pair_min_prob)])
        if int(graph_pair_max_per_node) <= 0:
            raise_error(stage, location, "graph_pair_max_per_node invalido (<=0)", impact="1", examples=[str(graph_pair_max_per_node)])
    thermo_pair_min_prob = float(payload.get("thermo_pair_min_prob", graph_pair_min_prob if graph_pair_edges != "none" else 0.0))
    thermo_pair_max_per_node = int(payload.get("thermo_pair_max_per_node", graph_pair_max_per_node if graph_pair_edges != "none" else 0))
    thermo_soft_constraint_strength = float(payload.get("thermo_soft_constraint_strength", 0.0))
    if not (0.0 <= float(thermo_pair_min_prob) <= 1.0):
        raise_error(stage, location, "thermo_pair_min_prob fora de [0,1]", impact="1", examples=[str(thermo_pair_min_prob)])
    if int(thermo_pair_max_per_node) < 0:
        raise_error(stage, location, "thermo_pair_max_per_node invalido (<0)", impact="1", examples=[str(thermo_pair_max_per_node)])
    if float(thermo_soft_constraint_strength) < 0.0:
        raise_error(stage, location, "thermo_soft_constraint_strength invalido (<0)", impact="1", examples=[str(thermo_soft_constraint_strength)])
    thermo_backend = str(payload.get("thermo_backend", "rnafold")).strip().lower()
    if thermo_backend not in {"rnafold", "linearfold", "viennarna"}:
        raise_error(stage, location, "thermo_backend invalido na config", impact="1", examples=[thermo_backend])
    msa_backend = str(payload.get("msa_backend", "mmseqs2")).strip().lower()
    if msa_backend not in {"mmseqs2"}:
        raise_error(stage, location, "msa_backend invalido na config", impact="1", examples=[msa_backend])
    chain_separator = str(payload.get("chain_separator", "|"))
    if len(chain_separator) != 1:
        raise_error(stage, location, "chain_separator deve ter 1 caractere", impact="1", examples=[chain_separator])
    training_protocol = str(payload.get("training_protocol", "custom")).strip().lower()
    cfg = Se3TrainConfig(
        hidden_dim=int(payload["hidden_dim"]),
        num_layers=int(payload["num_layers"]),
        ipa_heads=int(payload["ipa_heads"]),
        diffusion_steps=int(payload["diffusion_steps"]),
        flow_steps=int(payload["flow_steps"]),
        epochs=int(payload["epochs"]),
        learning_rate=float(payload["learning_rate"]),
        method=method,
        sequence_tower=sequence_tower,
        sequence_heads=int(payload.get("sequence_heads", payload["ipa_heads"])),
        use_gradient_checkpointing=bool(payload.get("use_gradient_checkpointing", False)),
        graph_backend=graph_backend,
        radius_angstrom=float(payload.get("radius_angstrom", 14.0)),
        max_neighbors=int(payload.get("max_neighbors", 64)),
        graph_chunk_size=int(payload.get("graph_chunk_size", 512)),
        ipa_edge_chunk_size=int(payload.get("ipa_edge_chunk_size", 128 if training_protocol == "local_16gb" else 4096)),
        graph_pair_edges=graph_pair_edges,
        graph_pair_min_prob=float(graph_pair_min_prob),
        graph_pair_max_per_node=int(graph_pair_max_per_node),
        thermo_pair_min_prob=float(thermo_pair_min_prob),
        thermo_pair_max_per_node=int(thermo_pair_max_per_node),
        thermo_soft_constraint_strength=float(thermo_soft_constraint_strength),
        thermo_backend=thermo_backend,
        rnafold_bin=str(payload.get("rnafold_bin", "RNAfold")).strip(),
        linearfold_bin=str(payload.get("linearfold_bin", "linearfold")).strip(),
        thermo_cache_dir=(None if payload.get("thermo_cache_dir", None) in {None, ""} else str(payload.get("thermo_cache_dir"))),
        msa_backend=msa_backend,
        mmseqs_bin=str(payload.get("mmseqs_bin", "mmseqs")).strip(),
        mmseqs_db=str(payload.get("mmseqs_db", "")).strip(),
        msa_cache_dir=(None if payload.get("msa_cache_dir", None) in {None, ""} else str(payload.get("msa_cache_dir"))),
        chain_separator=chain_separator,
        chain_break_offset=int(payload.get("chain_break_offset", 1000)),
        max_msa_sequences=int(payload.get("max_msa_sequences", 96)),
        max_cov_positions=int(payload.get("max_cov_positions", 256)),
        max_cov_pairs=int(payload.get("max_cov_pairs", 8192)),
        loss_weight_mse=float(payload.get("loss_weight_mse", 0.0)),
        loss_weight_fape=float(payload.get("loss_weight_fape", 1.0)),
        loss_weight_tm=float(payload.get("loss_weight_tm", 1.0)),
        loss_weight_clash=float(payload.get("loss_weight_clash", 5.0)),
        fape_clamp_distance=float(payload.get("fape_clamp_distance", 10.0)),
        fape_length_scale=float(payload.get("fape_length_scale", 10.0)),
        vdw_min_distance=float(payload.get("vdw_min_distance", 2.1)),
        vdw_repulsion_power=int(payload.get("vdw_repulsion_power", 4)),
        loss_chunk_size=int(payload.get("loss_chunk_size", 256)),
        dynamic_cropping=bool(payload.get("dynamic_cropping", True)),
        crop_min_length=int(payload.get("crop_min_length", 256)),
        crop_max_length=int(payload.get("crop_max_length", 384)),
        crop_sequence_fraction=float(payload.get("crop_sequence_fraction", 0.60)),
        gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps", 16)),
        autocast_bfloat16=bool(payload.get("autocast_bfloat16", True)),
        training_protocol=training_protocol,
    )
    numeric_checks = [
        ("hidden_dim", cfg.hidden_dim),
        ("num_layers", cfg.num_layers),
        ("ipa_heads", cfg.ipa_heads),
        ("diffusion_steps", cfg.diffusion_steps),
        ("flow_steps", cfg.flow_steps),
        ("epochs", cfg.epochs),
        ("sequence_heads", cfg.sequence_heads),
        ("max_neighbors", cfg.max_neighbors),
        ("graph_chunk_size", cfg.graph_chunk_size),
        ("ipa_edge_chunk_size", cfg.ipa_edge_chunk_size),
        ("chain_break_offset", cfg.chain_break_offset),
        ("max_msa_sequences", cfg.max_msa_sequences),
        ("max_cov_positions", cfg.max_cov_positions),
        ("max_cov_pairs", cfg.max_cov_pairs),
        ("vdw_repulsion_power", cfg.vdw_repulsion_power),
        ("loss_chunk_size", cfg.loss_chunk_size),
        ("crop_min_length", cfg.crop_min_length),
        ("crop_max_length", cfg.crop_max_length),
        ("gradient_accumulation_steps", cfg.gradient_accumulation_steps),
    ]
    bad = [name for name, value in numeric_checks if int(value) <= 0]
    if bad:
        raise_error(stage, location, "config com valor invalido (<=0)", impact=str(len(bad)), examples=bad[:8])
    if cfg.learning_rate <= 0:
        raise_error(stage, location, "learning_rate invalido (<=0)", impact="1", examples=[str(cfg.learning_rate)])
    if cfg.training_protocol not in {"custom", "local_16gb"}:
        raise_error(stage, location, "training_protocol invalido", impact="1", examples=[str(cfg.training_protocol)])
    if cfg.radius_angstrom <= 0:
        raise_error(stage, location, "radius_angstrom invalido (<=0)", impact="1", examples=[str(cfg.radius_angstrom)])
    if cfg.graph_pair_edges != "none" and int(cfg.graph_pair_max_per_node) > int(cfg.max_neighbors):
        raise_error(
            stage,
            location,
            "graph_pair_max_per_node nao pode exceder max_neighbors",
            impact="1",
            examples=[f"pair_max={int(cfg.graph_pair_max_per_node)}", f"max_neighbors={int(cfg.max_neighbors)}"],
        )
    if int(cfg.thermo_pair_max_per_node) > 0 and int(cfg.thermo_pair_max_per_node) > int(cfg.max_neighbors):
        raise_error(
            stage,
            location,
            "thermo_pair_max_per_node nao pode exceder max_neighbors",
            impact="1",
            examples=[f"thermo_pair_max={int(cfg.thermo_pair_max_per_node)}", f"max_neighbors={int(cfg.max_neighbors)}"],
        )
    if cfg.fape_clamp_distance <= 0:
        raise_error(stage, location, "fape_clamp_distance invalido (<=0)", impact="1", examples=[str(cfg.fape_clamp_distance)])
    if cfg.fape_length_scale <= 0:
        raise_error(stage, location, "fape_length_scale invalido (<=0)", impact="1", examples=[str(cfg.fape_length_scale)])
    if cfg.vdw_min_distance <= 0:
        raise_error(stage, location, "vdw_min_distance invalido (<=0)", impact="1", examples=[str(cfg.vdw_min_distance)])
    if cfg.vdw_repulsion_power <= 1:
        raise_error(stage, location, "vdw_repulsion_power deve ser > 1", impact="1", examples=[str(cfg.vdw_repulsion_power)])
    if cfg.crop_min_length < 2:
        raise_error(stage, location, "crop_min_length deve ser >= 2", impact="1", examples=[str(cfg.crop_min_length)])
    if cfg.crop_max_length < cfg.crop_min_length:
        raise_error(
            stage,
            location,
            "crop_max_length deve ser >= crop_min_length",
            impact="1",
            examples=[f"min={cfg.crop_min_length}", f"max={cfg.crop_max_length}"],
        )
    if cfg.crop_sequence_fraction <= 0.0 or cfg.crop_sequence_fraction > 1.0:
        raise_error(stage, location, "crop_sequence_fraction deve estar em (0,1]", impact="1", examples=[str(cfg.crop_sequence_fraction)])
    loss_weights = {
        "loss_weight_mse": cfg.loss_weight_mse,
        "loss_weight_fape": cfg.loss_weight_fape,
        "loss_weight_tm": cfg.loss_weight_tm,
        "loss_weight_clash": cfg.loss_weight_clash,
    }
    bad_weights = [name for name, value in loss_weights.items() if float(value) < 0.0]
    if bad_weights:
        raise_error(stage, location, "loss_weight invalido (<0)", impact=str(len(bad_weights)), examples=bad_weights[:8])
    if sum(float(value) for value in loss_weights.values()) <= 0.0:
        raise_error(stage, location, "soma dos loss_weight deve ser > 0", impact="1", examples=[str(loss_weights)])
    if cfg.hidden_dim % cfg.ipa_heads != 0:
        raise_error(
            stage,
            location,
            "hidden_dim deve ser multiplo de ipa_heads",
            impact="1",
            examples=[f"hidden_dim={cfg.hidden_dim}", f"ipa_heads={cfg.ipa_heads}"],
        )
    if cfg.hidden_dim % cfg.sequence_heads != 0:
        raise_error(
            stage,
            location,
            "hidden_dim deve ser multiplo de sequence_heads",
            impact="1",
            examples=[f"hidden_dim={cfg.hidden_dim}", f"sequence_heads={cfg.sequence_heads}"],
        )
    if not cfg.rnafold_bin:
        raise_error(stage, location, "rnafold_bin vazio na config", impact="1", examples=[cfg.rnafold_bin])
    if not cfg.linearfold_bin:
        raise_error(stage, location, "linearfold_bin vazio na config", impact="1", examples=[cfg.linearfold_bin])
    if not cfg.mmseqs_bin:
        raise_error(stage, location, "mmseqs_bin vazio na config", impact="1", examples=[cfg.mmseqs_bin])
    if cfg.msa_backend == "mmseqs2" and not cfg.mmseqs_db:
        raise_error(stage, location, "msa_backend=mmseqs2 exige mmseqs_db", impact="1", examples=[cfg.mmseqs_db])
    if cfg.training_protocol == "local_16gb":
        if not bool(cfg.dynamic_cropping):
            raise_error(stage, location, "training_protocol=local_16gb exige dynamic_cropping=true", impact="1", examples=[str(cfg.dynamic_cropping)])
        if str(cfg.graph_backend) != "torch_geometric":
            raise_error(
                stage,
                location,
                "training_protocol=local_16gb exige graph_backend=torch_geometric",
                impact="1",
                examples=[str(cfg.graph_backend)],
            )
        if int(cfg.crop_min_length) < 256 or int(cfg.crop_min_length) > 384:
            raise_error(stage, location, "training_protocol=local_16gb exige crop_min_length em [256,384]", impact="1", examples=[str(cfg.crop_min_length)])
        if int(cfg.crop_max_length) < 256 or int(cfg.crop_max_length) > 384:
            raise_error(stage, location, "training_protocol=local_16gb exige crop_max_length em [256,384]", impact="1", examples=[str(cfg.crop_max_length)])
        if int(cfg.crop_min_length) > int(cfg.crop_max_length):
            raise_error(
                stage,
                location,
                "training_protocol=local_16gb exige crop_min_length <= crop_max_length",
                impact="1",
                examples=[f"min={cfg.crop_min_length}", f"max={cfg.crop_max_length}"],
            )
        if not bool(cfg.autocast_bfloat16):
            raise_error(stage, location, "training_protocol=local_16gb exige autocast_bfloat16=true", impact="1", examples=[str(cfg.autocast_bfloat16)])
        if not bool(cfg.use_gradient_checkpointing):
            raise_error(
                stage,
                location,
                "training_protocol=local_16gb exige use_gradient_checkpointing=true",
                impact="1",
                examples=[str(cfg.use_gradient_checkpointing)],
            )
        if int(cfg.ipa_edge_chunk_size) > 256:
            raise_error(
                stage,
                location,
                "training_protocol=local_16gb exige ipa_edge_chunk_size <= 256",
                impact="1",
                examples=[str(cfg.ipa_edge_chunk_size)],
            )
        if int(cfg.gradient_accumulation_steps) < 16 or int(cfg.gradient_accumulation_steps) > 32:
            raise_error(
                stage,
                location,
                "training_protocol=local_16gb exige gradient_accumulation_steps em [16,32]",
                impact="1",
                examples=[str(cfg.gradient_accumulation_steps)],
            )
    return cfg
