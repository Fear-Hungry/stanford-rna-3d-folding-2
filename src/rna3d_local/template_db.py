from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .contracts import parse_date_column, require_columns, require_non_null
from .errors import raise_error
from .io_tables import read_table, write_table
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class TemplateDbResult:
    templates_path: Path
    template_index_path: Path
    manifest_path: Path


def build_template_db(
    *,
    repo_root: Path,
    external_templates_path: Path,
    out_dir: Path,
) -> TemplateDbResult:
    stage = "TEMPLATE_DB"
    location = "src/rna3d_local/template_db.py:build_template_db"
    df = read_table(external_templates_path, stage=stage, location=location)
    required = ["template_id", "sequence", "release_date", "resid", "resname", "x", "y", "z"]
    require_columns(df, required, stage=stage, location=location, label="external_templates")
    if "source" not in df.columns:
        df = df.with_columns(pl.lit("external").alias("source"))

    df = df.select(
        pl.col("template_id").cast(pl.Utf8).str.strip_chars(),
        pl.col("sequence").cast(pl.Utf8).str.strip_chars(),
        pl.col("release_date").cast(pl.Utf8),
        pl.col("resid").cast(pl.Int32),
        pl.col("resname").cast(pl.Utf8).str.strip_chars(),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
        pl.col("source").cast(pl.Utf8).str.strip_chars(),
    )
    require_non_null(
        df,
        ["template_id", "sequence", "release_date", "resid", "resname", "x", "y", "z", "source"],
        stage=stage,
        location=location,
        label="external_templates",
    )
    df = parse_date_column(df, "release_date", stage=stage, location=location, label="external_templates")
    df = df.with_columns((pl.col("source") + pl.lit(":") + pl.col("template_id")).alias("template_uid"))

    duplicates = df.group_by(["template_uid", "resid"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if duplicates.height > 0:
        examples = (
            duplicates.with_columns((pl.col("template_uid") + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k"))
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(stage, location, "duplicata de residuo por template", impact=str(duplicates.height), examples=examples)

    seq_conflicts = (
        df.group_by("template_uid")
        .agg(pl.col("sequence").n_unique().alias("n_seq"))
        .filter(pl.col("n_seq") > 1)
    )
    if seq_conflicts.height > 0:
        examples = seq_conflicts.get_column("template_uid").head(8).to_list()
        raise_error(stage, location, "template com multiplas sequences", impact=str(seq_conflicts.height), examples=examples)

    non_contiguous = (
        df.group_by("template_uid")
        .agg(
            pl.col("resid").min().alias("min_resid"),
            pl.col("resid").max().alias("max_resid"),
            pl.col("resid").n_unique().alias("n_unique"),
        )
        .with_columns((pl.col("max_resid") - pl.col("min_resid") + 1).alias("expected"))
        .filter(pl.col("n_unique") != pl.col("expected"))
    )
    if non_contiguous.height > 0:
        examples = non_contiguous.get_column("template_uid").head(8).to_list()
        raise_error(stage, location, "residuos nao contiguos no template", impact=str(non_contiguous.height), examples=examples)

    templates = df.select(
        "template_uid",
        "template_id",
        "source",
        "release_date",
        "resid",
        "resname",
        "x",
        "y",
        "z",
    ).sort(["template_uid", "resid"])
    template_index = (
        df.group_by(["template_uid", "template_id", "source", "sequence", "release_date"])
        .agg(pl.len().alias("n_residues"))
        .sort(["release_date", "template_uid"])
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    templates_path = out_dir / "templates.parquet"
    template_index_path = out_dir / "template_index.parquet"
    manifest_path = out_dir / "manifest.json"
    write_table(templates, templates_path)
    write_table(template_index, template_index_path)

    manifest = {
        "created_utc": utc_now_iso(),
        "paths": {
            "external_templates": rel_or_abs(external_templates_path, repo_root),
            "templates": rel_or_abs(templates_path, repo_root),
            "template_index": rel_or_abs(template_index_path, repo_root),
        },
        "stats": {
            "n_templates": int(template_index.height),
            "n_residue_rows": int(templates.height),
        },
        "sha256": {
            "templates.parquet": sha256_file(templates_path),
            "template_index.parquet": sha256_file(template_index_path),
        },
    }
    write_json(manifest_path, manifest)
    return TemplateDbResult(templates_path=templates_path, template_index_path=template_index_path, manifest_path=manifest_path)
