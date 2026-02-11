from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .bigdata import (
    DEFAULT_MAX_ROWS_IN_MEMORY,
    DEFAULT_MEMORY_BUDGET_MB,
    LabelStoreConfig,
    TableReadConfig,
    assert_memory_budget,
    assert_row_budget,
    collect_streaming,
    scan_labels,
    scan_table,
)
from .errors import raise_error
from .utils import sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class TemplateDbBuildResult:
    manifest_path: Path
    templates_path: Path
    index_path: Path


def _read_frame(path: Path, *, location: str) -> pl.DataFrame:
    return collect_streaming(
        lf=scan_table(
            config=TableReadConfig(
                path=path,
                stage="TEMPLATE_DB",
                location=location,
            )
        ),
        stage="TEMPLATE_DB",
        location=location,
    )


def _validate_columns(df: pl.DataFrame, required: list[str], *, location: str, stage: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise_error(
            stage,
            location,
            "colunas obrigatorias ausentes",
            impact=str(len(missing)),
            examples=missing[:8],
        )


def _cast_release_date(df: pl.DataFrame, *, col: str, location: str, stage: str) -> pl.DataFrame:
    out = df.with_columns(pl.col(col).cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias(col))
    bad = out.filter(pl.col(col).is_null())
    if bad.height > 0:
        examples = bad.select(pl.all().first()).row(0)
        raise_error(
            stage,
            location,
            "data invalida; esperado YYYY-MM-DD",
            impact=str(bad.height),
            examples=[str(examples)],
        )
    return out


def _validate_non_nulls(df: pl.DataFrame, columns: list[str], *, location: str, stage: str) -> None:
    bad: list[str] = []
    for col in columns:
        n = int(df.get_column(col).null_count())
        if n > 0:
            bad.append(f"{col}:{n}")
    if bad:
        raise_error(
            stage,
            location,
            "valores nulos em colunas obrigatorias",
            impact=str(len(bad)),
            examples=bad[:8],
        )


def _local_templates_from_train(
    *,
    train_sequences_path: Path,
    train_labels_parquet_dir: Path,
    max_train_templates: int | None,
    location: str,
    max_rows_in_memory: int,
) -> pl.DataFrame:
    seq_df = pl.read_csv(train_sequences_path, infer_schema_length=1000)
    assert_row_budget(
        stage="TEMPLATE_DB",
        location=location,
        rows=int(seq_df.height),
        max_rows_in_memory=max_rows_in_memory,
        label="train_sequences",
    )
    _validate_columns(
        seq_df,
        ["target_id", "sequence", "temporal_cutoff"],
        location=location,
        stage="TEMPLATE_DB",
    )
    seq_df = _cast_release_date(seq_df, col="temporal_cutoff", location=location, stage="TEMPLATE_DB").rename(
        {"temporal_cutoff": "release_date"}
    )
    seq_df = seq_df.select(
        pl.col("target_id").cast(pl.Utf8),
        pl.col("sequence").cast(pl.Utf8),
        pl.col("release_date"),
    )
    seq_df = seq_df.filter(pl.col("target_id").is_not_null() & pl.col("sequence").is_not_null())
    if seq_df.height == 0:
        raise_error("TEMPLATE_DB", location, "train_sequences sem linhas validas", impact="0", examples=[])

    if max_train_templates is not None:
        if max_train_templates <= 0:
            raise_error(
                "TEMPLATE_DB",
                location,
                "max_train_templates deve ser > 0",
                impact="1",
                examples=[str(max_train_templates)],
            )
        seq_df = seq_df.sort("target_id").head(max_train_templates)

    train_ids = seq_df.get_column("target_id").to_list()
    scan = scan_labels(
        config=LabelStoreConfig(
            labels_parquet_dir=train_labels_parquet_dir,
            required_columns=("ID", "resname", "resid", "x_1", "y_1", "z_1"),
            stage="TEMPLATE_DB",
            location=location,
        )
    )

    labels = (
        scan.with_columns(pl.col("ID").cast(pl.Utf8).str.extract(r"^(.*)_\d+$", 1).alias("template_id"))
        .filter(pl.col("template_id").is_in(train_ids))
        .select(
            pl.col("template_id").cast(pl.Utf8),
            pl.col("resid").cast(pl.Int32),
            pl.col("resname").cast(pl.Utf8),
            pl.col("x_1").cast(pl.Float64).alias("x"),
            pl.col("y_1").cast(pl.Float64).alias("y"),
            pl.col("z_1").cast(pl.Float64).alias("z"),
        )
        .collect(engine="streaming")
    )
    assert_row_budget(
        stage="TEMPLATE_DB",
        location=location,
        rows=int(labels.height),
        max_rows_in_memory=max_rows_in_memory,
        label="train_labels_filtered",
    )
    if labels.height == 0:
        raise_error("TEMPLATE_DB", location, "nenhuma coordenada local encontrada", impact="0", examples=[])

    out = labels.join(seq_df.rename({"target_id": "template_id"}), on="template_id", how="inner")
    if out.height == 0:
        raise_error(
            "TEMPLATE_DB",
            location,
            "falha ao reconciliar labels com sequences",
            impact="0",
            examples=[],
        )
    out = out.with_columns(
        pl.lit("train_kaggle").alias("source"),
        (pl.lit("train_kaggle:") + pl.col("template_id")).alias("template_uid"),
    )
    return out.select(
        "template_uid",
        "template_id",
        "source",
        "sequence",
        "release_date",
        "resid",
        "resname",
        "x",
        "y",
        "z",
    )


def _external_templates(
    *,
    external_templates_path: Path,
    location: str,
    max_rows_in_memory: int,
) -> pl.DataFrame:
    df = _read_frame(external_templates_path, location=location)
    assert_row_budget(
        stage="TEMPLATE_DB",
        location=location,
        rows=int(df.height),
        max_rows_in_memory=max_rows_in_memory,
        label="external_templates",
    )
    required = ["template_id", "sequence", "release_date", "resid", "resname", "x", "y", "z"]
    _validate_columns(df, required, location=location, stage="TEMPLATE_DB")
    if "source" not in df.columns:
        df = df.with_columns(pl.lit("external").alias("source"))

    df = _cast_release_date(df, col="release_date", location=location, stage="TEMPLATE_DB")
    df = df.select(
        pl.col("template_id").cast(pl.Utf8),
        pl.col("source").cast(pl.Utf8),
        pl.col("sequence").cast(pl.Utf8),
        pl.col("release_date"),
        pl.col("resid").cast(pl.Int32),
        pl.col("resname").cast(pl.Utf8),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
    ).with_columns((pl.col("source") + pl.lit(":") + pl.col("template_id")).alias("template_uid"))
    return df.select(
        "template_uid",
        "template_id",
        "source",
        "sequence",
        "release_date",
        "resid",
        "resname",
        "x",
        "y",
        "z",
    )


def build_template_db(
    *,
    repo_root: Path,
    train_sequences_path: Path,
    external_templates_path: Path,
    out_dir: Path,
    train_labels_parquet_dir: Path,
    max_train_templates: int | None = None,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    max_rows_in_memory: int = DEFAULT_MAX_ROWS_IN_MEMORY,
) -> TemplateDbBuildResult:
    """
    Build a strict, temporal-aware template database combining local Kaggle train and external templates.
    """
    location = "src/rna3d_local/template_db.py:build_template_db"
    assert_memory_budget(stage="TEMPLATE_DB", location=location, budget_mb=memory_budget_mb)
    for p in (train_sequences_path, external_templates_path):
        if not p.exists():
            raise_error("TEMPLATE_DB", location, "arquivo obrigatorio ausente", impact="1", examples=[str(p)])

    local_df = _local_templates_from_train(
        train_sequences_path=train_sequences_path,
        train_labels_parquet_dir=train_labels_parquet_dir,
        max_train_templates=max_train_templates,
        location=location,
        max_rows_in_memory=max_rows_in_memory,
    )
    ext_df = _external_templates(
        external_templates_path=external_templates_path,
        location=location,
        max_rows_in_memory=max_rows_in_memory,
    )

    all_df = pl.concat([local_df, ext_df], how="vertical_relaxed")
    assert_row_budget(
        stage="TEMPLATE_DB",
        location=location,
        rows=int(all_df.height),
        max_rows_in_memory=max_rows_in_memory,
        label="all_templates",
    )
    _validate_non_nulls(
        all_df,
        ["template_uid", "template_id", "source", "sequence", "release_date", "resid", "resname", "x", "y", "z"],
        location=location,
        stage="TEMPLATE_DB",
    )

    dup = (
        all_df.group_by(["template_uid", "resid"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
        .sort(["template_uid", "resid"])
    )
    if dup.height > 0:
        ex = dup.select((pl.col("template_uid") + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k")).get_column("k")
        raise_error(
            "TEMPLATE_DB",
            location,
            "duplicata de residuo no template",
            impact=str(dup.height),
            examples=ex.head(8).to_list(),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    templates_path = out_dir / "templates.parquet"
    all_df.sort(["template_uid", "resid"]).write_parquet(templates_path)

    idx = (
        all_df.group_by(["template_uid", "template_id", "source", "sequence", "release_date"])
        .agg(pl.len().alias("n_residues"))
        .sort(["release_date", "template_uid"])
    )
    index_path = out_dir / "template_index.parquet"
    idx.write_parquet(index_path)
    assert_memory_budget(stage="TEMPLATE_DB", location=location, budget_mb=memory_budget_mb)

    manifest = {
        "created_utc": _utc_now(),
        "template_count": int(idx.height),
        "residue_rows": int(all_df.height),
        "paths": {
            "templates": _rel(templates_path, repo_root),
            "template_index": _rel(index_path, repo_root),
            "train_sequences": _rel(train_sequences_path, repo_root),
            "train_labels_parquet_dir": _rel(train_labels_parquet_dir, repo_root),
            "external_templates": _rel(external_templates_path, repo_root),
        },
        "sha256": {
            "templates.parquet": sha256_file(templates_path),
            "template_index.parquet": sha256_file(index_path),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TemplateDbBuildResult(
        manifest_path=manifest_path,
        templates_path=templates_path,
        index_path=index_path,
    )
