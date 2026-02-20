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
    templates: pl.DataFrame,
    *,
    stage: str,
    location: str,
) -> pl.DataFrame:
    require_columns(candidates, ["target_id", "template_uid"], stage=stage, location=location, label="candidates")
    require_columns(
        chemical_features,
        ["target_id", "resid", "p_open", "p_paired"],
        stage=stage,
        location=location,
        label="chemical_features",
    )
    require_columns(templates, ["template_uid", "resid", "x", "y", "z"], stage=stage, location=location, label="templates")

    chem = chemical_features.select(
        pl.col("target_id").cast(pl.Utf8),
        pl.col("resid").cast(pl.Int32),
        pl.col("p_open").cast(pl.Float64),
        pl.col("p_paired").cast(pl.Float64),
    )
    chem_dup = chem.group_by(["target_id", "resid"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if chem_dup.height > 0:
        examples = (
            chem_dup.with_columns((pl.col("target_id") + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k"))
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(stage, location, "chemical_features com chave duplicada", impact=str(int(chem_dup.height)), examples=[str(item) for item in examples])
    chem_bad = chem.filter(
        pl.col("p_open").is_null()
        | pl.col("p_paired").is_null()
        | (pl.col("p_open") < 0.0)
        | (pl.col("p_open") > 1.0)
        | (pl.col("p_paired") < 0.0)
        | (pl.col("p_paired") > 1.0)
    )
    if chem_bad.height > 0:
        examples = (
            chem_bad.with_columns((pl.col("target_id") + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k"))
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(
            stage,
            location,
            "chemical_features com p_open/p_paired fora de [0,1] ou nulos",
            impact=str(int(chem_bad.height)),
            examples=[str(item) for item in examples],
        )

    tpl = templates.select(
        pl.col("template_uid").cast(pl.Utf8),
        pl.col("resid").cast(pl.Int32),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
    )
    tpl_dup = tpl.group_by(["template_uid", "resid"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if tpl_dup.height > 0:
        examples = (
            tpl_dup.with_columns((pl.col("template_uid") + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k"))
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(stage, location, "templates com chave duplicada", impact=str(int(tpl_dup.height)), examples=[str(item) for item in examples])

    centers = tpl.group_by("template_uid").agg(
        pl.col("x").mean().alias("_cx"),
        pl.col("y").mean().alias("_cy"),
        pl.col("z").mean().alias("_cz"),
    )
    tpl_with_radius = tpl.join(centers, on="template_uid", how="left").with_columns(
        ((((pl.col("x") - pl.col("_cx")) ** 2) + ((pl.col("y") - pl.col("_cy")) ** 2) + ((pl.col("z") - pl.col("_cz")) ** 2)).sqrt()).alias("_radius")
    )
    min_resid = tpl_with_radius.group_by("template_uid").agg(pl.col("resid").min().alias("_min_resid"))
    tpl_with_norm_resid = tpl_with_radius.join(min_resid, on="template_uid", how="left").with_columns(
        (pl.col("resid") - pl.col("_min_resid") + 1).cast(pl.Int32).alias("resid_norm")
    )
    radius_stats = tpl_with_norm_resid.group_by("template_uid").agg(
        pl.col("_radius").min().alias("_r_min"),
        pl.col("_radius").max().alias("_r_max"),
    )
    tpl_profile = (
        tpl_with_norm_resid.join(radius_stats, on="template_uid", how="left")
        .with_columns(
            pl.when((pl.col("_r_max") - pl.col("_r_min")) <= 1e-8)
            .then(pl.lit(0.5))
            .otherwise((pl.col("_radius") - pl.col("_r_min")) / (pl.col("_r_max") - pl.col("_r_min")))
            .alias("geom_p_open")
        )
        .with_columns((1.0 - pl.col("geom_p_open")).alias("geom_p_paired"))
        .select(
            pl.col("template_uid"),
            pl.col("resid_norm").cast(pl.Int32).alias("resid"),
            pl.col("geom_p_open").cast(pl.Float64),
            pl.col("geom_p_paired").cast(pl.Float64),
        )
    )

    candidate_keys = candidates.select(pl.col("target_id").cast(pl.Utf8), pl.col("template_uid").cast(pl.Utf8)).unique()
    chem_vs_template = (
        candidate_keys.join(chem, on="target_id", how="inner")
        .join(tpl_profile, on=["template_uid", "resid"], how="left")
        .group_by(["target_id", "template_uid"])
        .agg(
            pl.len().alias("_target_len"),
            pl.col("geom_p_open").is_not_null().sum().alias("_overlap_len"),
            (pl.col("p_open") - pl.col("geom_p_open")).abs().mean().alias("_open_mae"),
            (pl.col("p_paired") - pl.col("geom_p_paired")).abs().mean().alias("_paired_mae"),
        )
        .with_columns((pl.col("_overlap_len").cast(pl.Float64) / pl.col("_target_len").cast(pl.Float64)).alias("_coverage"))
    )

    no_overlap = chem_vs_template.filter(pl.col("_overlap_len") <= 0)
    if no_overlap.height > 0:
        examples = (
            no_overlap.with_columns((pl.col("target_id") + pl.lit(":") + pl.col("template_uid")).alias("k"))
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(
            stage,
            location,
            "candidates sem sobreposicao target-template para feicoes quimico-geometricas",
            impact=str(int(no_overlap.height)),
            examples=[str(item) for item in examples],
        )

    chem_features_by_candidate = chem_vs_template.with_columns(
        ((1.0 - pl.col("_open_mae").fill_null(1.0)).clip(0.0, 1.0) * pl.col("_coverage")).alias("chem_p_open_mean"),
        ((1.0 - pl.col("_paired_mae").fill_null(1.0)).clip(0.0, 1.0) * pl.col("_coverage")).alias("chem_p_paired_mean"),
    ).select("target_id", "template_uid", "chem_p_open_mean", "chem_p_paired_mean")

    base = candidates.join(chem_features_by_candidate, on=["target_id", "template_uid"], how="left")
    missing = base.filter(pl.col("chem_p_open_mean").is_null() | pl.col("chem_p_paired_mean").is_null())
    if missing.height > 0:
        examples = (
            missing.select((pl.col("target_id") + pl.lit(":") + pl.col("template_uid")).alias("k"))
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(stage, location, "falha ao montar feicoes quimicas por candidato", impact=str(int(missing.height)), examples=[str(item) for item in examples])

    require_columns(base, FEATURE_COLUMNS, stage=stage, location=location, label="training_features")
    return base


def train_template_reranker(
    *,
    repo_root: Path,
    candidates_path: Path,
    chemical_features_path: Path,
    templates_path: Path,
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
    require_columns(chemical, ["target_id", "resid", "p_open", "p_paired"], stage=stage, location=location, label="chemical_features")
    templates = read_table(templates_path, stage=stage, location=location)
    require_columns(templates, ["template_uid", "resid", "x", "y", "z"], stage=stage, location=location, label="templates")

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

    features_df = _prepare_features(labeled, chemical, templates, stage=stage, location=location)
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
    templates_path: Path,
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
    require_columns(chemical, ["target_id", "resid", "p_open", "p_paired"], stage=stage, location=location, label="chemical_features")
    templates = read_table(templates_path, stage=stage, location=location)
    require_columns(templates, ["template_uid", "resid", "x", "y", "z"], stage=stage, location=location, label="templates")
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

    features_df = _prepare_features(candidates, chemical, templates, stage=stage, location=location)
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
            "templates": rel_or_abs(templates_path, repo_root),
            "model": rel_or_abs(model_path, repo_root),
            "config": rel_or_abs(config_path, repo_root),
            "scored": rel_or_abs(out_path, repo_root),
        },
        "stats": {"n_rows": int(scored.height)},
        "sha256": {"scored.parquet": sha256_file(out_path)},
    }
    write_json(manifest_path, manifest)
    return RerankerScoreResult(scored_path=out_path, manifest_path=manifest_path)
