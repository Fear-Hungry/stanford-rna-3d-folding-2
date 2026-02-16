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
    ]
    bad = [name for name, value in numeric_checks if int(value) <= 0]
    if bad:
        raise_error(stage, location, "config com valor invalido (<=0)", impact=str(len(bad)), examples=bad[:8])
    if cfg.learning_rate <= 0:
        raise_error(stage, location, "learning_rate invalido (<=0)", impact="1", examples=[str(cfg.learning_rate)])
    if cfg.radius_angstrom <= 0:
        raise_error(stage, location, "radius_angstrom invalido (<=0)", impact="1", examples=[str(cfg.radius_angstrom)])
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
    return cfg
