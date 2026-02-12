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


def _count_rows(*, lf: pl.LazyFrame, stage: str, location: str) -> int:
    return int(
        collect_streaming(
            lf=lf.select(pl.len().alias("n")),
            stage=stage,
            location=location,
        ).get_column("n")[0]
    )


def _validate_columns(*, lf: pl.LazyFrame, required: list[str], location: str, stage: str) -> None:
    cols = set(lf.collect_schema().names())
    missing = [c for c in required if c not in cols]
    if missing:
        raise_error(
            stage,
            location,
            "colunas obrigatorias ausentes",
            impact=str(len(missing)),
            examples=missing[:8],
        )


def _validate_non_nulls(*, lf: pl.LazyFrame, columns: list[str], location: str, stage: str) -> None:
    null_counts = collect_streaming(
        lf=lf.select([pl.col(c).is_null().sum().alias(c) for c in columns]),
        stage=stage,
        location=location,
    ).row(0, named=True)
    bad = [f"{c}:{int(n)}" for c, n in null_counts.items() if int(n) > 0]
    if bad:
        raise_error(
            stage,
            location,
            "valores nulos em colunas obrigatorias",
            impact=str(len(bad)),
            examples=bad[:8],
        )


def _validate_duplicate_residues(*, lf: pl.LazyFrame, location: str, stage: str) -> None:
    dup_lf = (
        lf.group_by(["template_uid", "resid"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    )
    dup_n = _count_rows(lf=dup_lf, stage=stage, location=location)
    if dup_n <= 0:
        return
    examples = collect_streaming(
        lf=dup_lf.select(
            (pl.col("template_uid") + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k")
        ).head(8),
        stage=stage,
        location=location,
    ).get_column("k").to_list()
    raise_error(
        stage,
        location,
        "duplicata de residuo no template",
        impact=str(dup_n),
        examples=examples,
    )


def _load_train_sequences(
    *,
    train_sequences_path: Path,
    max_train_templates: int | None,
    location: str,
    max_rows_in_memory: int,
) -> pl.DataFrame:
    seq_df = collect_streaming(
        lf=scan_table(
            config=TableReadConfig(
                path=train_sequences_path,
                stage="TEMPLATE_DB",
                location=location,
                columns=("target_id", "sequence", "temporal_cutoff"),
            )
        ).select(
            pl.col("target_id").cast(pl.Utf8),
            pl.col("sequence").cast(pl.Utf8),
            pl.col("temporal_cutoff").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("release_date"),
        ),
        stage="TEMPLATE_DB",
        location=location,
    )
    assert_row_budget(
        stage="TEMPLATE_DB",
        location=location,
        rows=int(seq_df.height),
        max_rows_in_memory=max_rows_in_memory,
        label="train_sequences",
    )

    bad_dates = seq_df.filter(pl.col("release_date").is_null())
    if bad_dates.height > 0:
        examples = bad_dates.get_column("target_id").head(8).to_list()
        raise_error(
            "TEMPLATE_DB",
            location,
            "data invalida; esperado YYYY-MM-DD em temporal_cutoff",
            impact=str(int(bad_dates.height)),
            examples=examples,
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

    return seq_df


def _local_templates_from_train_lazy(
    *,
    train_sequences_df: pl.DataFrame,
    train_labels_parquet_dir: Path,
    location: str,
) -> pl.LazyFrame:
    train_ids = train_sequences_df.get_column("target_id").to_list()
    labels_lf = scan_labels(
        config=LabelStoreConfig(
            labels_parquet_dir=train_labels_parquet_dir,
            required_columns=("ID", "resname", "resid", "x_1", "y_1", "z_1"),
            stage="TEMPLATE_DB",
            location=location,
        )
    )
    release_lf = train_sequences_df.lazy().select(
        pl.col("target_id").cast(pl.Utf8).alias("template_id"),
        pl.col("release_date"),
    )

    return (
        labels_lf.with_columns(pl.col("ID").cast(pl.Utf8).str.extract(r"^(.*)_\d+$", 1).alias("template_id"))
        .filter(pl.col("template_id").is_in(train_ids))
        .select(
            pl.col("template_id").cast(pl.Utf8),
            pl.col("resid").cast(pl.Int32),
            pl.col("resname").cast(pl.Utf8),
            pl.col("x_1").cast(pl.Float64).alias("x"),
            pl.col("y_1").cast(pl.Float64).alias("y"),
            pl.col("z_1").cast(pl.Float64).alias("z"),
        )
        .join(release_lf, on="template_id", how="inner")
        .with_columns(
            pl.lit("train_kaggle").alias("source"),
            (pl.lit("train_kaggle:") + pl.col("template_id")).alias("template_uid"),
        )
        .select(
            "template_uid",
            "template_id",
            "source",
            "release_date",
            "resid",
            "resname",
            "x",
            "y",
            "z",
        )
    )


def _external_templates_and_index_lazy(*, external_templates_path: Path, location: str) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    raw = scan_table(
        config=TableReadConfig(
            path=external_templates_path,
            stage="TEMPLATE_DB",
            location=location,
        )
    )

    required = ["template_id", "sequence", "release_date", "resid", "resname", "x", "y", "z"]
    _validate_columns(lf=raw, required=required, location=location, stage="TEMPLATE_DB")

    if "source" not in raw.collect_schema().names():
        raw = raw.with_columns(pl.lit("external").alias("source"))

    ext_with_seq_lf = raw.select(
        pl.col("template_id").cast(pl.Utf8),
        pl.col("source").cast(pl.Utf8),
        pl.col("sequence").cast(pl.Utf8),
        pl.col("release_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("release_date"),
        pl.col("resid").cast(pl.Int32),
        pl.col("resname").cast(pl.Utf8),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
    ).with_columns((pl.col("source") + pl.lit(":") + pl.col("template_id")).alias("template_uid"))

    bad_date_n = _count_rows(lf=ext_with_seq_lf.filter(pl.col("release_date").is_null()), stage="TEMPLATE_DB", location=location)
    if bad_date_n > 0:
        examples = collect_streaming(
            lf=ext_with_seq_lf.filter(pl.col("release_date").is_null()).select("template_id").head(8),
            stage="TEMPLATE_DB",
            location=location,
        ).get_column("template_id").to_list()
        raise_error(
            "TEMPLATE_DB",
            location,
            "data invalida; esperado YYYY-MM-DD em release_date",
            impact=str(bad_date_n),
            examples=examples,
        )

    ext_templates_lf = ext_with_seq_lf.select(
        "template_uid",
        "template_id",
        "source",
        "release_date",
        "resid",
        "resname",
        "x",
        "y",
        "z",
    )
    ext_index_lf = (
        ext_with_seq_lf.group_by(["template_uid", "template_id", "source", "sequence", "release_date"])
        .agg(pl.len().alias("n_residues"))
        .select("template_uid", "template_id", "source", "sequence", "release_date", "n_residues")
    )
    return ext_templates_lf, ext_index_lf


def _local_index_lazy(*, train_sequences_df: pl.DataFrame, local_templates_lf: pl.LazyFrame, location: str) -> pl.LazyFrame:
    seq_meta_lf = train_sequences_df.lazy().select(
        (pl.lit("train_kaggle:") + pl.col("target_id").cast(pl.Utf8)).alias("template_uid"),
        pl.col("target_id").cast(pl.Utf8).alias("template_id"),
        pl.lit("train_kaggle").alias("source"),
        pl.col("sequence").cast(pl.Utf8),
        pl.col("release_date"),
    )
    local_counts_lf = local_templates_lf.group_by("template_uid").agg(pl.len().alias("n_residues"))

    missing_seq_lf = seq_meta_lf.join(local_counts_lf, on="template_uid", how="anti")
    missing_n = _count_rows(lf=missing_seq_lf, stage="TEMPLATE_DB", location=location)
    if missing_n > 0:
        examples = collect_streaming(
            lf=missing_seq_lf.select("template_id").head(8),
            stage="TEMPLATE_DB",
            location=location,
        ).get_column("template_id").to_list()
        raise_error(
            "TEMPLATE_DB",
            location,
            "targets sem residuos locais apos reconciliacao labels/sequences",
            impact=str(missing_n),
            examples=examples,
        )

    return seq_meta_lf.join(local_counts_lf, on="template_uid", how="inner").select(
        "template_uid",
        "template_id",
        "source",
        "sequence",
        "release_date",
        "n_residues",
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

    train_sequences_df = _load_train_sequences(
        train_sequences_path=train_sequences_path,
        max_train_templates=max_train_templates,
        location=location,
        max_rows_in_memory=max_rows_in_memory,
    )

    local_templates_lf = _local_templates_from_train_lazy(
        train_sequences_df=train_sequences_df,
        train_labels_parquet_dir=train_labels_parquet_dir,
        location=location,
    )
    ext_templates_lf, ext_index_lf = _external_templates_and_index_lazy(
        external_templates_path=external_templates_path,
        location=location,
    )
    local_index_lf = _local_index_lazy(
        train_sequences_df=train_sequences_df,
        local_templates_lf=local_templates_lf,
        location=location,
    )

    local_rows = _count_rows(lf=local_templates_lf, stage="TEMPLATE_DB", location=location)
    if local_rows == 0:
        raise_error("TEMPLATE_DB", location, "nenhuma coordenada local encontrada", impact="0", examples=[])

    ext_rows = _count_rows(lf=ext_templates_lf, stage="TEMPLATE_DB", location=location)
    if ext_rows == 0:
        raise_error("TEMPLATE_DB", location, "external_templates sem linhas validas", impact="0", examples=[])

    all_templates_lf = pl.concat([local_templates_lf, ext_templates_lf], how="vertical_relaxed")
    all_index_lf = pl.concat([local_index_lf, ext_index_lf], how="vertical_relaxed")

    _validate_non_nulls(
        lf=all_templates_lf,
        columns=["template_uid", "template_id", "source", "release_date", "resid", "resname", "x", "y", "z"],
        location=location,
        stage="TEMPLATE_DB",
    )
    _validate_duplicate_residues(lf=all_templates_lf, location=location, stage="TEMPLATE_DB")
    _validate_non_nulls(
        lf=all_index_lf,
        columns=["template_uid", "template_id", "source", "sequence", "release_date", "n_residues"],
        location=location,
        stage="TEMPLATE_DB",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    templates_path = out_dir / "templates.parquet"
    index_path = out_dir / "template_index.parquet"
    tmp_templates_path = out_dir / "_tmp_templates.parquet"
    tmp_index_path = out_dir / "_tmp_template_index.parquet"

    for p in (tmp_templates_path, tmp_index_path):
        if p.exists():
            p.unlink()

    all_templates_lf.sink_parquet(tmp_templates_path, compression="zstd")
    all_index_lf.sort(["release_date", "template_uid"]).sink_parquet(tmp_index_path, compression="zstd")

    if templates_path.exists():
        templates_path.unlink()
    if index_path.exists():
        index_path.unlink()
    tmp_templates_path.replace(templates_path)
    tmp_index_path.replace(index_path)

    template_count = _count_rows(
        lf=scan_table(
            config=TableReadConfig(
                path=index_path,
                stage="TEMPLATE_DB",
                location=location,
                columns=("template_uid",),
            )
        ),
        stage="TEMPLATE_DB",
        location=location,
    )
    residue_rows = _count_rows(
        lf=scan_table(
            config=TableReadConfig(
                path=templates_path,
                stage="TEMPLATE_DB",
                location=location,
                columns=("template_uid",),
            )
        ),
        stage="TEMPLATE_DB",
        location=location,
    )

    assert_memory_budget(stage="TEMPLATE_DB", location=location, budget_mb=memory_budget_mb)

    manifest = {
        "created_utc": _utc_now(),
        "template_count": int(template_count),
        "residue_rows": int(residue_rows),
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
