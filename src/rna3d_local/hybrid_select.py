from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .contracts import require_columns
from .errors import raise_error
from .io_tables import read_table, write_table
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class HybridTop5Result:
    predictions_path: Path
    manifest_path: Path


def select_top5_hybrid(
    *,
    repo_root: Path,
    candidates_path: Path,
    out_path: Path,
    n_models: int = 5,
) -> HybridTop5Result:
    stage = "SELECT_TOP5_HYBRID"
    location = "src/rna3d_local/hybrid_select.py:select_top5_hybrid"
    if n_models <= 0:
        raise_error(stage, location, "n_models deve ser > 0", impact="1", examples=[str(n_models)])

    candidates = read_table(candidates_path, stage=stage, location=location)
    require_columns(
        candidates,
        ["target_id", "model_id", "resid", "resname", "x", "y", "z", "source", "confidence"],
        stage=stage,
        location=location,
        label="candidates",
    )
    key_dup = candidates.group_by(["target_id", "source", "model_id", "resid"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if key_dup.height > 0:
        examples = (
            key_dup.with_columns(
                (
                    pl.col("target_id")
                    + pl.lit(":")
                    + pl.col("source")
                    + pl.lit(":")
                    + pl.col("model_id").cast(pl.Utf8)
                    + pl.lit(":")
                    + pl.col("resid").cast(pl.Utf8)
                ).alias("k")
            )
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(stage, location, "candidates com chave duplicada", impact=str(key_dup.height), examples=[str(x) for x in examples])

    model_scores = (
        candidates.group_by(["target_id", "source", "model_id"])
        .agg(
            pl.col("confidence").mean().alias("model_confidence"),
            pl.len().alias("n_rows"),
        )
        .sort(["target_id", "model_confidence", "source", "model_id"], descending=[False, True, False, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("target_id").alias("rank"))
    )
    selected = model_scores.filter(pl.col("rank") <= int(n_models))
    per_target = selected.group_by("target_id").agg(pl.len().alias("n_selected")).filter(pl.col("n_selected") != int(n_models))
    if per_target.height > 0:
        examples = per_target.with_columns((pl.col("target_id") + pl.lit(":") + pl.col("n_selected").cast(pl.Utf8)).alias("k")).get_column("k").head(8).to_list()
        raise_error(stage, location, "alvos com numero insuficiente de modelos selecionados", impact=str(per_target.height), examples=[str(x) for x in examples])

    selected_keys = selected.select("target_id", "source", "model_id", "rank").rename({"rank": "final_model_id"})
    out = (
        candidates.join(selected_keys, on=["target_id", "source", "model_id"], how="inner")
        .select(
            pl.col("target_id"),
            pl.col("final_model_id").cast(pl.Int32).alias("model_id"),
            pl.col("resid").cast(pl.Int32),
            pl.col("resname").cast(pl.Utf8),
            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
            pl.col("z").cast(pl.Float64),
            pl.col("source").cast(pl.Utf8),
            pl.col("confidence").cast(pl.Float64),
        )
        .sort(["target_id", "model_id", "resid"])
    )
    write_table(out, out_path)
    manifest_path = out_path.parent / "select_top5_hybrid_manifest.json"
    write_json(
        manifest_path,
        {
            "created_utc": utc_now_iso(),
            "params": {"n_models": int(n_models)},
            "paths": {
                "candidates": rel_or_abs(candidates_path, repo_root),
                "predictions": rel_or_abs(out_path, repo_root),
            },
            "stats": {
                "n_rows": int(out.height),
                "n_targets": int(out.get_column("target_id").n_unique()),
            },
            "sha256": {"predictions.parquet": sha256_file(out_path)},
        },
    )
    return HybridTop5Result(predictions_path=out_path, manifest_path=manifest_path)
