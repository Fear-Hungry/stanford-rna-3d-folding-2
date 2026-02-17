from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .contracts import require_columns
from .errors import raise_error
from .io_tables import read_table, write_table
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class ChemicalFeaturesResult:
    features_path: Path
    manifest_path: Path


def _resolve_column(columns: list[str], candidates: list[str], *, stage: str, location: str, label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise_error(stage, location, f"coluna obrigatoria nao encontrada para {label}", impact="1", examples=candidates)


def _normalize_and_finalize(df: pl.DataFrame, *, stage: str, location: str) -> pl.DataFrame:
    require_columns(
        df,
        ["target_id", "resid", "reactivity_dms", "reactivity_2a3"],
        stage=stage,
        location=location,
        label="chemical_features_raw",
    )
    bad = df.filter(
        pl.col("target_id").is_null()
        | (pl.col("target_id").str.strip_chars() == "")
        | pl.col("resid").is_null()
        | pl.col("reactivity_dms").is_null()
        | pl.col("reactivity_2a3").is_null()
    )
    if bad.height > 0:
        examples = bad.select("target_id", "resid").head(8).rows()
        raise_error(stage, location, "valores nulos/invalidos em chemical_features", impact=str(bad.height), examples=[str(item) for item in examples])

    dup = df.group_by(["target_id", "resid"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if dup.height > 0:
        examples = (
            dup.with_columns((pl.col("target_id") + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k"))
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(stage, location, "chaves duplicadas em chemical_features", impact=str(dup.height), examples=[str(item) for item in examples])

    stats = df.group_by("target_id").agg(
        pl.col("reactivity_dms").min().alias("dms_min"),
        pl.col("reactivity_dms").max().alias("dms_max"),
        pl.col("reactivity_2a3").min().alias("a3_min"),
        pl.col("reactivity_2a3").max().alias("a3_max"),
    )
    return (
        df.join(stats, on="target_id", how="left")
        .with_columns(
            pl.when((pl.col("dms_max") - pl.col("dms_min")) > 0)
            .then((pl.col("reactivity_dms") - pl.col("dms_min")) / (pl.col("dms_max") - pl.col("dms_min")))
            .otherwise(0.5)
            .alias("_dms_norm"),
            pl.when((pl.col("a3_max") - pl.col("a3_min")) > 0)
            .then((pl.col("reactivity_2a3") - pl.col("a3_min")) / (pl.col("a3_max") - pl.col("a3_min")))
            .otherwise(0.5)
            .alias("_a3_norm"),
        )
        .with_columns(((pl.col("_dms_norm") + pl.col("_a3_norm")) / 2.0).alias("p_open"))
        .with_columns((1.0 - pl.col("p_open")).alias("p_paired"))
        .select("target_id", "resid", "reactivity_dms", "reactivity_2a3", "p_open", "p_paired")
        .sort(["target_id", "resid"])
    )


def _prepare_from_reactivity_table(table: pl.DataFrame, *, stage: str, location: str) -> tuple[pl.DataFrame, str]:
    require_columns(table, ["target_id", "resid"], stage=stage, location=location, label="ribonanza_quickstart")
    dms_col = _resolve_column(table.columns, ["dms", "dms_reactivity"], stage=stage, location=location, label="dms")
    twoa3_col = _resolve_column(
        table.columns,
        ["2a3", "twoa3", "a3", "reactivity_2a3", "2a3_reactivity"],
        stage=stage,
        location=location,
        label="2a3",
    )
    df = table.select(
        pl.col("target_id").cast(pl.Utf8),
        pl.col("resid").cast(pl.Int32),
        pl.col(dms_col).cast(pl.Float64).alias("reactivity_dms"),
        pl.col(twoa3_col).cast(pl.Float64).alias("reactivity_2a3"),
    )
    return _normalize_and_finalize(df, stage=stage, location=location), "reactivity_columns"


def _prepare_from_template_coordinates(table: pl.DataFrame, *, stage: str, location: str) -> tuple[pl.DataFrame, str]:
    require_columns(table, ["ID", "resid"], stage=stage, location=location, label="ribonanza_template_quickstart")
    has_plain_xyz = all(column in table.columns for column in ["x", "y", "z"])
    suffixes = sorted(
        {
            column.split("_", 1)[1]
            for column in table.columns
            if ("_" in column) and (column.split("_", 1)[0] in {"x", "y", "z"}) and column.split("_", 1)[1].isdigit()
        },
        key=lambda item: int(item),
    )
    valid_suffixes = [suffix for suffix in suffixes if f"x_{suffix}" in table.columns and f"y_{suffix}" in table.columns and f"z_{suffix}" in table.columns]
    if not has_plain_xyz and len(valid_suffixes) < 1:
        raise_error(
            stage,
            location,
            "schema de template quickstart invalido: tripletos x_i,y_i,z_i ausentes",
            impact=str(len(valid_suffixes)),
            examples=valid_suffixes[:8],
        )

    if has_plain_xyz:
        x_cols = ["x"]
        y_cols = ["y"]
        z_cols = ["z"]
        schema_mode = "template_coordinates_triplets=1_plain_xyz"
    else:
        x_cols = [f"x_{suffix}" for suffix in valid_suffixes]
        y_cols = [f"y_{suffix}" for suffix in valid_suffixes]
        z_cols = [f"z_{suffix}" for suffix in valid_suffixes]
        schema_mode = f"template_coordinates_triplets={len(valid_suffixes)}"
    view = table.select(
        pl.col("ID").cast(pl.Utf8).alias("_id"),
        pl.col("resid").cast(pl.Int32),
        *[pl.col(column).cast(pl.Float64) for column in x_cols + y_cols + z_cols],
    )
    x = np.asarray(view.select(x_cols).to_numpy(), dtype=np.float64)
    y = np.asarray(view.select(y_cols).to_numpy(), dtype=np.float64)
    z = np.asarray(view.select(z_cols).to_numpy(), dtype=np.float64)
    missing_mask = np.isnan(x).any(axis=1) | np.isnan(y).any(axis=1) | np.isnan(z).any(axis=1)
    if bool(missing_mask.any()):
        bad_ids = view.select("_id").to_series().to_list()
        examples = [str(bad_ids[idx]) for idx in np.flatnonzero(missing_mask)[:8]]
        raise_error(stage, location, "coordenadas nulas no quickstart de templates", impact=str(int(missing_mask.sum())), examples=examples)

    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)
    mean_z = z.mean(axis=1)
    spread = (x.std(axis=1) + y.std(axis=1) + z.std(axis=1)) / 3.0
    ids = view.select("_id").to_series().to_list()
    resids = view.select("resid").to_series().to_list()
    target_ids: list[str] = []
    for raw_id, resid in zip(ids, resids, strict=True):
        text = str(raw_id).strip()
        if ":" in text:
            text = text.split(":", 1)[0].strip()
        elif "_" in text:
            prefix, suffix = text.rsplit("_", 1)
            if suffix.isdigit():
                try:
                    if int(suffix) == int(resid):
                        text = str(prefix).strip()
                except Exception:
                    pass
        target_ids.append(text)
    blank_targets = [str(ids[idx]) for idx, target_id in enumerate(target_ids) if not target_id]
    if blank_targets:
        raise_error(stage, location, "ID invalido sem target_id", impact=str(len(blank_targets)), examples=blank_targets[:8])

    base = pl.DataFrame(
        {
            "target_id": target_ids,
            "resid": view.select("resid").to_series().to_list(),
            "_mx": mean_x.tolist(),
            "_my": mean_y.tolist(),
            "_mz": mean_z.tolist(),
            "reactivity_dms": spread.tolist(),
        }
    )
    centroid = base.group_by("target_id").agg(
        pl.col("_mx").mean().alias("_cx"),
        pl.col("_my").mean().alias("_cy"),
        pl.col("_mz").mean().alias("_cz"),
    )
    with_centroid = base.join(centroid, on="target_id", how="left")
    enriched = with_centroid.with_columns(
        (
            (
                (pl.col("_mx") - pl.col("_cx")) * (pl.col("_mx") - pl.col("_cx"))
                + (pl.col("_my") - pl.col("_cy")) * (pl.col("_my") - pl.col("_cy"))
                + (pl.col("_mz") - pl.col("_cz")) * (pl.col("_mz") - pl.col("_cz"))
            ).sqrt()
        ).alias("reactivity_2a3")
    )
    if has_plain_xyz or len(valid_suffixes) == 1:
        enriched = enriched.with_columns(pl.col("reactivity_2a3").alias("reactivity_dms"))
    enriched = enriched.select("target_id", "resid", "reactivity_dms", "reactivity_2a3")
    return _normalize_and_finalize(enriched, stage=stage, location=location), schema_mode


def prepare_chemical_features(
    *,
    repo_root: Path,
    quickstart_path: Path,
    out_path: Path,
) -> ChemicalFeaturesResult:
    stage = "CHEMICAL_FEATURES"
    location = "src/rna3d_local/chemical_features.py:prepare_chemical_features"
    table = read_table(quickstart_path, stage=stage, location=location)
    if "target_id" in table.columns:
        enriched, schema_mode = _prepare_from_reactivity_table(table, stage=stage, location=location)
    elif "ID" in table.columns:
        enriched, schema_mode = _prepare_from_template_coordinates(table, stage=stage, location=location)
    else:
        raise_error(
            stage,
            location,
            "schema quickstart nao suportado: esperado target_id/resid ou ID/resid",
            impact="1",
            examples=table.columns[:8],
        )
    write_table(enriched, out_path)

    manifest_path = out_path.parent / "chemical_features_manifest.json"
    manifest = {
        "created_utc": utc_now_iso(),
        "paths": {
            "quickstart": rel_or_abs(quickstart_path, repo_root),
            "chemical_features": rel_or_abs(out_path, repo_root),
        },
        "stats": {
            "n_rows": int(enriched.height),
            "n_targets": int(enriched.get_column("target_id").n_unique()),
            "schema_mode": schema_mode,
        },
        "sha256": {"chemical_features.parquet": sha256_file(out_path)},
    }
    write_json(manifest_path, manifest)
    return ChemicalFeaturesResult(features_path=out_path, manifest_path=manifest_path)
