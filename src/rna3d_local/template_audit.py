from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from .bigdata import (
    DEFAULT_MAX_ROWS_IN_MEMORY,
    DEFAULT_MEMORY_BUDGET_MB,
    TableReadConfig,
    assert_memory_budget,
    assert_row_budget,
    collect_streaming,
    scan_table,
)
from .errors import raise_error
from .utils import sha256_file

_ALLOWED_BASES = {"A", "C", "G", "U"}
_MIN_STEP_ANGSTROM = 1.0
_MAX_STEP_ANGSTROM = 15.0
_MAX_ABS_COORD_ANGSTROM = 1000.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _count_rows(*, lf: pl.LazyFrame, location: str) -> int:
    return int(
        collect_streaming(
            lf=lf.select(pl.len().alias("n")),
            stage="TEMPLATE_AUDIT",
            location=location,
        ).get_column("n")[0]
    )


def _raise_if_bad_rows(*, df: pl.DataFrame, cause: str, location: str, example_expr: pl.Expr) -> None:
    if df.height <= 0:
        return
    examples = (
        collect_streaming(
            lf=df.lazy().select(example_expr.alias("ex")).head(8),
            stage="TEMPLATE_AUDIT",
            location=location,
        )
        .get_column("ex")
        .to_list()
    )
    raise_error(
        "TEMPLATE_AUDIT",
        location,
        cause,
        impact=str(int(df.height)),
        examples=[str(x) for x in examples],
    )


def _as_template_uid_expr() -> pl.Expr:
    return (pl.col("source") + pl.lit(":") + pl.col("template_id")).alias("template_uid")


def audit_external_templates(
    *,
    external_templates_path: Path,
    out_report_path: Path | None = None,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    max_rows_in_memory: int = DEFAULT_MAX_ROWS_IN_MEMORY,
) -> dict:
    location = "src/rna3d_local/template_audit.py:audit_external_templates"
    assert_memory_budget(stage="TEMPLATE_AUDIT", location=location, budget_mb=memory_budget_mb)
    if not external_templates_path.exists():
        raise_error("TEMPLATE_AUDIT", location, "arquivo obrigatorio ausente", impact="1", examples=[str(external_templates_path)])

    raw = scan_table(
        config=TableReadConfig(
            path=external_templates_path,
            stage="TEMPLATE_AUDIT",
            location=location,
        )
    )
    cols = set(raw.collect_schema().names())
    required = ("template_id", "sequence", "release_date", "resid", "resname", "x", "y", "z")
    missing = [c for c in required if c not in cols]
    if missing:
        raise_error(
            "TEMPLATE_AUDIT",
            location,
            "external_templates sem coluna obrigatoria",
            impact=str(len(missing)),
            examples=missing[:8],
        )

    source_expr = pl.col("source").cast(pl.Utf8) if "source" in cols else pl.lit("external", dtype=pl.Utf8)
    df = collect_streaming(
        lf=raw.select(
            pl.col("template_id").cast(pl.Utf8).str.strip_chars().alias("template_id"),
            source_expr.str.strip_chars().alias("source"),
            pl.col("sequence").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("sequence"),
            pl.col("release_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("release_date"),
            pl.col("resid").cast(pl.Int32).alias("resid"),
            pl.col("resname").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("resname"),
            pl.col("x").cast(pl.Float64).alias("x"),
            pl.col("y").cast(pl.Float64).alias("y"),
            pl.col("z").cast(pl.Float64).alias("z"),
        ),
        stage="TEMPLATE_AUDIT",
        location=location,
    ).with_columns(_as_template_uid_expr())

    assert_row_budget(
        stage="TEMPLATE_AUDIT",
        location=location,
        rows=int(df.height),
        max_rows_in_memory=max_rows_in_memory,
        label="external_templates",
    )
    if df.height <= 0:
        raise_error("TEMPLATE_AUDIT", location, "external_templates vazio", impact="0", examples=[str(external_templates_path)])

    null_expr = (
        pl.col("template_id").is_null()
        | (pl.col("template_id").str.len_chars() == 0)
        | pl.col("source").is_null()
        | (pl.col("source").str.len_chars() == 0)
        | pl.col("sequence").is_null()
        | (pl.col("sequence").str.len_chars() == 0)
        | pl.col("release_date").is_null()
        | pl.col("resid").is_null()
        | pl.col("resname").is_null()
        | (pl.col("resname").str.len_chars() == 0)
        | pl.col("x").is_null()
        | pl.col("y").is_null()
        | pl.col("z").is_null()
    )
    _raise_if_bad_rows(
        df=df.filter(null_expr),
        cause="campos obrigatorios nulos ou vazios",
        location=location,
        example_expr=pl.col("template_uid") + pl.lit(":") + pl.col("resid").cast(pl.Utf8),
    )

    seq_bad = df.filter(~pl.col("sequence").str.contains(r"^[ACGU]+$"))
    _raise_if_bad_rows(
        df=seq_bad,
        cause="sequence invalida (permitido somente A,C,G,U)",
        location=location,
        example_expr=pl.col("template_uid") + pl.lit(":") + pl.col("sequence"),
    )

    resname_bad = df.filter(~pl.col("resname").is_in(list(_ALLOWED_BASES)))
    _raise_if_bad_rows(
        df=resname_bad,
        cause="resname invalido (permitido somente A,C,G,U)",
        location=location,
        example_expr=pl.col("template_uid") + pl.lit(":") + pl.col("resname"),
    )

    today = date.today()
    future = df.filter(pl.col("release_date") > pl.lit(today))
    _raise_if_bad_rows(
        df=future,
        cause="release_date futura nao permitida",
        location=location,
        example_expr=pl.col("template_uid")
        + pl.lit(":")
        + pl.col("release_date").cast(pl.Utf8),
    )

    finite_bad = df.filter(
        (~pl.col("x").is_finite())
        | (~pl.col("y").is_finite())
        | (~pl.col("z").is_finite())
    )
    _raise_if_bad_rows(
        df=finite_bad,
        cause="coordenadas nao finitas em templates externos",
        location=location,
        example_expr=pl.col("template_uid")
        + pl.lit(":")
        + pl.col("resid").cast(pl.Utf8),
    )

    scale_bad = df.filter(
        (pl.col("x").abs() > float(_MAX_ABS_COORD_ANGSTROM))
        | (pl.col("y").abs() > float(_MAX_ABS_COORD_ANGSTROM))
        | (pl.col("z").abs() > float(_MAX_ABS_COORD_ANGSTROM))
    )
    _raise_if_bad_rows(
        df=scale_bad,
        cause="coordenadas fora de escala plausivel (|coord|>1000A)",
        location=location,
        example_expr=pl.col("template_uid")
        + pl.lit(":")
        + pl.col("resid").cast(pl.Utf8),
    )

    dup = (
        df.lazy()
        .group_by(["template_uid", "resid"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
    )
    dup_n = _count_rows(lf=dup, location=location)
    if dup_n > 0:
        examples = (
            collect_streaming(
                lf=dup.select(
                    (
                        pl.col("template_uid")
                        + pl.lit(":")
                        + pl.col("resid").cast(pl.Utf8)
                    ).alias("ex")
                ).head(8),
                stage="TEMPLATE_AUDIT",
                location=location,
            )
            .get_column("ex")
            .to_list()
        )
        raise_error(
            "TEMPLATE_AUDIT",
            location,
            "duplicata de residuo por template",
            impact=str(dup_n),
            examples=[str(x) for x in examples],
        )

    group_df = collect_streaming(
        lf=df.lazy()
        .group_by("template_uid")
        .agg(
            pl.col("sequence").n_unique().alias("sequence_n_unique"),
            pl.col("sequence").first().alias("sequence"),
            pl.col("resid").min().alias("resid_min"),
            pl.col("resid").max().alias("resid_max"),
            pl.col("resid").n_unique().alias("resid_n_unique"),
            pl.col("template_id").first().alias("template_id"),
            pl.col("source").first().alias("source"),
            pl.col("release_date").first().alias("release_date"),
            pl.col("resid").sort().alias("resids"),
            pl.col("x").sort_by("resid").alias("xs"),
            pl.col("y").sort_by("resid").alias("ys"),
            pl.col("z").sort_by("resid").alias("zs"),
        ),
        stage="TEMPLATE_AUDIT",
        location=location,
    )

    seq_consistency_bad = group_df.filter(pl.col("sequence_n_unique") != 1)
    _raise_if_bad_rows(
        df=seq_consistency_bad,
        cause="template com multiplas sequences divergentes",
        location=location,
        example_expr=pl.col("template_uid"),
    )

    contiguous_bad = group_df.filter(
        (pl.col("resid_min") != 1)
        | (pl.col("resid_max") != pl.col("resid_n_unique"))
    )
    _raise_if_bad_rows(
        df=contiguous_bad,
        cause="resids nao contiguos iniciando em 1",
        location=location,
        example_expr=pl.col("template_uid")
        + pl.lit(":")
        + pl.col("resid_min").cast(pl.Utf8)
        + pl.lit("-")
        + pl.col("resid_max").cast(pl.Utf8),
    )

    len_bad = group_df.filter(pl.col("sequence").str.len_chars() != pl.col("resid_n_unique"))
    _raise_if_bad_rows(
        df=len_bad,
        cause="comprimento da sequence diverge da contagem de residuos",
        location=location,
        example_expr=pl.col("template_uid")
        + pl.lit(":len_seq=")
        + pl.col("sequence").str.len_chars().cast(pl.Utf8)
        + pl.lit(":n_resid=")
        + pl.col("resid_n_unique").cast(pl.Utf8),
    )

    step_bad_count = 0
    step_bad_examples: list[str] = []
    for row in group_df.iter_rows(named=True):
        uid = str(row["template_uid"])
        resids = [int(v) for v in row["resids"]]
        xs = [float(v) for v in row["xs"]]
        ys = [float(v) for v in row["ys"]]
        zs = [float(v) for v in row["zs"]]
        for idx in range(1, len(resids)):
            rid_prev = resids[idx - 1]
            rid_cur = resids[idx]
            if rid_cur - rid_prev != 1:
                step_bad_count += 1
                if len(step_bad_examples) < 8:
                    step_bad_examples.append(f"{uid}:{rid_prev}-{rid_cur}:gap")
                continue
            dx = float(xs[idx] - xs[idx - 1])
            dy = float(ys[idx] - ys[idx - 1])
            dz = float(zs[idx] - zs[idx - 1])
            dist = float(math.sqrt((dx * dx) + (dy * dy) + (dz * dz)))
            if (not math.isfinite(dist)) or dist < float(_MIN_STEP_ANGSTROM) or dist > float(_MAX_STEP_ANGSTROM):
                step_bad_count += 1
                if len(step_bad_examples) < 8:
                    step_bad_examples.append(f"{uid}:{rid_prev}-{rid_cur}:dist={dist:.3f}")
    if step_bad_count > 0:
        raise_error(
            "TEMPLATE_AUDIT",
            location,
            "distancia consecutiva entre residuos fora de faixa plausivel [1A,15A]",
            impact=str(step_bad_count),
            examples=step_bad_examples,
        )

    report = {
        "created_utc": _utc_now(),
        "input_path": str(external_templates_path),
        "n_rows": int(df.height),
        "n_templates": int(group_df.height),
        "checks": {
            "allowed_sequence_alphabet": "ACGU",
            "allowed_resname": sorted(_ALLOWED_BASES),
            "max_abs_coord_angstrom": float(_MAX_ABS_COORD_ANGSTROM),
            "step_distance_range_angstrom": [float(_MIN_STEP_ANGSTROM), float(_MAX_STEP_ANGSTROM)],
            "release_date_not_future": True,
            "resid_contiguous_from_1": True,
            "sequence_length_matches_residues": True,
        },
        "sha256": {
            "external_templates": sha256_file(external_templates_path),
        },
    }
    if out_report_path is not None:
        out_report_path.parent.mkdir(parents=True, exist_ok=True)
        out_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["report_path"] = str(out_report_path)
    return report


__all__ = ["audit_external_templates"]
