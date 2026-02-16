from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from .contracts import require_columns
from .errors import raise_error
from .io_tables import read_table, write_table
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


FEATURE_COLUMNS = [
    "cosine_score",
    "family_prior_score",
    "alignment_refine_score",
    "chem_p_open_mean",
    "chem_p_paired_mean",
]


@dataclass(frozen=True)
class RerankerTrainResult:
    model_path: Path
    config_path: Path
    metrics_path: Path


@dataclass(frozen=True)
class RerankerScoreResult:
    scored_path: Path
    manifest_path: Path


class LinearReranker(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.linear(x).squeeze(-1)


def _prepare_features(
    candidates: pl.DataFrame,
    chemical_features: pl.DataFrame,
    *,
    stage: str,
    location: str,
) -> pl.DataFrame:
    chem_target = chemical_features.group_by("target_id").agg(
        pl.col("p_open").mean().alias("chem_p_open_mean"),
        pl.col("p_paired").mean().alias("chem_p_paired_mean"),
    )
    base = candidates.join(chem_target, on="target_id", how="left")
    base = base.with_columns(
        pl.col("chem_p_open_mean").fill_null(0.5),
        pl.col("chem_p_paired_mean").fill_null(0.5),
    )
    require_columns(base, FEATURE_COLUMNS, stage=stage, location=location, label="training_features")
    return base


def train_template_reranker(
    *,
    repo_root: Path,
    candidates_path: Path,
    chemical_features_path: Path,
    out_dir: Path,
    labels_path: Path | None,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> RerankerTrainResult:
    stage = "TRAIN_RERANKER"
    location = "src/rna3d_local/reranker.py:train_template_reranker"
    if epochs <= 0:
        raise_error(stage, location, "epochs deve ser > 0", impact="1", examples=[str(epochs)])
    if learning_rate <= 0:
        raise_error(stage, location, "learning_rate deve ser > 0", impact="1", examples=[str(learning_rate)])

    candidates = read_table(candidates_path, stage=stage, location=location)
    require_columns(candidates, ["target_id", "template_uid"], stage=stage, location=location, label="candidates")
    chemical = read_table(chemical_features_path, stage=stage, location=location)
    require_columns(chemical, ["target_id", "p_open", "p_paired"], stage=stage, location=location, label="chemical_features")

    if labels_path is None:
        if "label" not in candidates.columns:
            raise_error(stage, location, "labels ausentes: informe labels_path ou coluna label em candidates", impact="1", examples=["label"])
        labeled = candidates
    else:
        labels = read_table(labels_path, stage=stage, location=location)
        require_columns(labels, ["target_id", "template_uid", "label"], stage=stage, location=location, label="labels")
        labeled = candidates.join(
            labels.select("target_id", "template_uid", pl.col("label").cast(pl.Float64)),
            on=["target_id", "template_uid"],
            how="inner",
        )
        if labeled.height == 0:
            raise_error(stage, location, "join candidates x labels vazio", impact="0", examples=[])

    features_df = _prepare_features(labeled, chemical, stage=stage, location=location)
    require_columns(features_df, ["label"], stage=stage, location=location, label="features_df")
    if features_df.height < 8:
        raise_error(stage, location, "dados insuficientes para treino", impact=str(features_df.height), examples=["min=8"])

    x_np = features_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
    y_np = features_df.get_column("label").to_numpy().astype(np.float32)
    x_mean = x_np.mean(axis=0)
    x_std = x_np.std(axis=0)
    x_std = np.where(x_std <= 1e-8, 1.0, x_std)
    x_norm = (x_np - x_mean) / x_std

    torch.manual_seed(int(seed))
    model = LinearReranker(input_dim=len(FEATURE_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_fn = nn.BCEWithLogitsLoss()

    x_tensor = torch.tensor(x_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tensor)
        loss = loss_fn(logits, y_tensor)
        loss.backward()
        optimizer.step()

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pt"
    config_path = out_dir / "config.json"
    metrics_path = out_dir / "metrics.json"
    torch.save(model.state_dict(), model_path)
    write_json(
        config_path,
        {
            "model_type": "linear_reranker_with_chemical_bias",
            "feature_columns": FEATURE_COLUMNS,
            "feature_mean": x_mean.tolist(),
            "feature_std": x_std.tolist(),
            "seed": int(seed),
        },
    )
    write_json(
        metrics_path,
        {
            "created_utc": utc_now_iso(),
            "train_rows": int(features_df.height),
            "loss_final": float(loss.detach().cpu().item()),
        },
    )
    return RerankerTrainResult(model_path=model_path, config_path=config_path, metrics_path=metrics_path)


def score_template_reranker(
    *,
    repo_root: Path,
    candidates_path: Path,
    chemical_features_path: Path,
    model_path: Path,
    config_path: Path,
    out_path: Path,
    top_k: int | None,
) -> RerankerScoreResult:
    stage = "SCORE_RERANKER"
    location = "src/rna3d_local/reranker.py:score_template_reranker"
    if top_k is not None and top_k <= 0:
        raise_error(stage, location, "top_k deve ser > 0", impact="1", examples=[str(top_k)])
    candidates = read_table(candidates_path, stage=stage, location=location)
    require_columns(candidates, ["target_id", "template_uid"], stage=stage, location=location, label="candidates")
    chemical = read_table(chemical_features_path, stage=stage, location=location)
    require_columns(chemical, ["target_id", "p_open", "p_paired"], stage=stage, location=location, label="chemical_features")
    if not model_path.exists() or not config_path.exists():
        raise_error(stage, location, "model/config ausente", impact="1", examples=[str(model_path), str(config_path)])

    config = json.loads(config_path.read_text(encoding="utf-8"))
    feature_columns = config.get("feature_columns")
    if feature_columns != FEATURE_COLUMNS:
        raise_error(stage, location, "feature_columns do config invalido", impact="1", examples=[str(feature_columns)])
    mean = np.asarray(config.get("feature_mean", []), dtype=np.float32)
    std = np.asarray(config.get("feature_std", []), dtype=np.float32)
    if mean.shape[0] != len(FEATURE_COLUMNS) or std.shape[0] != len(FEATURE_COLUMNS):
        raise_error(stage, location, "feature_mean/std invalidos", impact="1", examples=[str(mean.shape), str(std.shape)])

    features_df = _prepare_features(candidates, chemical, stage=stage, location=location)
    x_np = features_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
    x_norm = (x_np - mean) / np.where(std <= 1e-8, 1.0, std)
    model = LinearReranker(input_dim=len(FEATURE_COLUMNS))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        scores = model(torch.tensor(x_norm, dtype=torch.float32)).detach().cpu().numpy().astype(np.float64)

    scored = features_df.with_columns(pl.Series(name="reranker_score", values=scores))
    scored = scored.sort(["target_id", "reranker_score"], descending=[False, True]).with_columns(
        pl.int_range(1, pl.len() + 1).over("target_id").alias("rerank_rank")
    )
    if top_k is not None:
        scored = scored.filter(pl.col("rerank_rank") <= int(top_k))
    write_table(scored, out_path)

    manifest_path = out_path.parent / "reranker_scoring_manifest.json"
    manifest = {
        "created_utc": utc_now_iso(),
        "paths": {
            "candidates": rel_or_abs(candidates_path, repo_root),
            "chemical_features": rel_or_abs(chemical_features_path, repo_root),
            "model": rel_or_abs(model_path, repo_root),
            "config": rel_or_abs(config_path, repo_root),
            "scored": rel_or_abs(out_path, repo_root),
        },
        "stats": {"n_rows": int(scored.height)},
        "sha256": {"scored.parquet": sha256_file(out_path)},
    }
    write_json(manifest_path, manifest)
    return RerankerScoreResult(scored_path=out_path, manifest_path=manifest_path)
