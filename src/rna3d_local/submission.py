from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .contracts import validate_submission_against_sample
from .errors import raise_error
from .io_tables import read_table


@dataclass(frozen=True)
class SubmissionExportResult:
    submission_path: Path


def _target_id_from_key(key: str) -> str:
    if "_" not in key:
        return key
    return key.rsplit("_", 1)[0]


def _resid_from_key(key: str) -> int:
    if "_" not in key:
        raise ValueError(key)
    return int(key.rsplit("_", 1)[1])


def export_submission(
    *,
    sample_path: Path,
    predictions_long_path: Path,
    out_path: Path,
) -> SubmissionExportResult:
    stage = "EXPORT"
    location = "src/rna3d_local/submission.py:export_submission"
    sample = read_table(sample_path, stage=stage, location=location)
    out_cols = list(sample.columns)
    pred = read_table(predictions_long_path, stage=stage, location=location)
    required = ["target_id", "model_id", "resid", "resname", "x", "y", "z"]
    missing = [column for column in required if column not in pred.columns]
    if missing:
        raise_error(stage, location, "predictions long sem coluna obrigatoria", impact=str(len(missing)), examples=missing[:8])

    model_cols = [column for column in sample.columns if column.startswith("x_")]
    model_ids = sorted(int(column.split("_", 1)[1]) for column in model_cols)
    if not model_ids:
        raise_error(stage, location, "sample sem colunas de modelo", impact="1", examples=sample.columns[:8])

    pred = pred.select(
        pl.col("target_id").cast(pl.Utf8),
        pl.col("model_id").cast(pl.Int32),
        pl.col("resid").cast(pl.Int32),
        pl.col("resname").cast(pl.Utf8),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
    )
    key_dup = pred.group_by(["target_id", "model_id", "resid"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if key_dup.height > 0:
        examples = (
            key_dup.select(
                (pl.col("target_id") + pl.lit(":") + pl.col("model_id").cast(pl.Utf8) + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k")
            )
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(stage, location, "predictions long com chave duplicada", impact=str(int(key_dup.height)), examples=[str(x) for x in examples])

    # Vectorize ID parsing to avoid Python loops (RAM-safe for large hidden datasets).
    sample_work = sample.with_row_index("_row").with_columns(
        [
            pl.col("ID").cast(pl.Utf8),
            pl.col("ID").cast(pl.Utf8).str.extract(r"^(.*)_(\d+)$", 1).alias("_target_id"),
            pl.col("ID").cast(pl.Utf8).str.extract(r"^(.*)_(\d+)$", 2).cast(pl.Int32).alias("_resid"),
        ]
    )
    bad_keys = sample_work.filter(pl.col("_target_id").is_null() | pl.col("_resid").is_null())
    if bad_keys.height > 0:
        examples = bad_keys.get_column("ID").head(8).to_list()
        raise_error(stage, location, "ID invalido (esperado <target>_<resid>)", impact=str(int(bad_keys.height)), examples=[str(x) for x in examples])

    sample_keys = sample_work.select(
        pl.col("_row").cast(pl.Int64),
        pl.col("ID").cast(pl.Utf8),
        pl.col("resname").cast(pl.Utf8),
        pl.col("resid").cast(pl.Int32),
        pl.col("_target_id").cast(pl.Utf8).alias("target_id"),
        pl.col("_resid").cast(pl.Int32).alias("resid_key"),
    )

    joined = sample_keys.join(pred, left_on=["target_id", "resid_key"], right_on=["target_id", "resid"], how="left")
    model_count = joined.group_by("_row", maintain_order=True).agg(pl.col("model_id").drop_nulls().n_unique().alias("n_models"))
    expected = int(len(model_ids))
    missing_rows = model_count.filter(pl.col("n_models") != expected)
    if missing_rows.height > 0:
        examples = (
            sample_keys.join(missing_rows.select("_row"), on="_row", how="inner")
            .select("ID")
            .get_column("ID")
            .head(8)
            .to_list()
        )
        raise_error(
            stage,
            location,
            "predictions long com chave faltante para sample",
            impact=str(int(missing_rows.height)),
            examples=[str(x) for x in examples],
        )

    aggs: list[pl.Expr] = [pl.first("ID").alias("ID"), pl.first("resname").alias("resname"), pl.first("resid").alias("resid")]
    for mid in model_ids:
        k = int(mid)
        aggs.extend(
            [
                pl.when(pl.col("model_id") == k).then(pl.col("x")).max().alias(f"x_{k}"),
                pl.when(pl.col("model_id") == k).then(pl.col("y")).max().alias(f"y_{k}"),
                pl.when(pl.col("model_id") == k).then(pl.col("z")).max().alias(f"z_{k}"),
            ]
        )
    out = joined.group_by("_row", maintain_order=True).agg(aggs).sort("_row").drop("_row").select(out_cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_path)
    validate_submission_against_sample(sample_path=sample_path, submission_path=out_path)
    return SubmissionExportResult(submission_path=out_path)


def check_submission(*, sample_path: Path, submission_path: Path) -> None:
    validate_submission_against_sample(sample_path=sample_path, submission_path=submission_path)
