from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from .bigdata import (
    DEFAULT_MAX_ROWS_IN_MEMORY,
    DEFAULT_MEMORY_BUDGET_MB,
    TableReadConfig,
    assert_memory_budget,
    collect_streaming,
    scan_table,
)
from .contracts import validate_submission_against_sample
from .errors import raise_error


def _count_rows(*, lf: pl.LazyFrame, location: str) -> int:
    return int(
        collect_streaming(
            lf=lf.select(pl.len().alias("n")),
            stage="EXPORT",
            location=location,
        ).get_column("n")[0]
    )


def _extract_model_ids(sample_cols: list[str], *, location: str) -> list[int]:
    model_ids: set[int] = set()
    for col in sample_cols:
        m = re.fullmatch(r"[xyz]_(\d+)", col)
        if not m:
            continue
        model_ids.add(int(m.group(1)))
    if not model_ids:
        raise_error(
            "EXPORT",
            location,
            "sample_submission sem colunas de coordenadas",
            impact="0",
            examples=sample_cols[:8],
        )
    for i in sorted(model_ids):
        for axis in ("x", "y", "z"):
            c = f"{axis}_{i}"
            if c not in sample_cols:
                raise_error("EXPORT", location, "sample_submission incompleto por modelo", impact="1", examples=[c])
    return sorted(model_ids)


def export_submission_from_long(
    *,
    sample_submission_path: Path,
    predictions_long_path: Path,
    out_submission_path: Path,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    max_rows_in_memory: int = DEFAULT_MAX_ROWS_IN_MEMORY,
) -> Path:
    """
    Export strict Kaggle-format submission from long predictions:
    expected columns in long file: ID,resid,resname,model_id,x,y,z
    """
    location = "src/rna3d_local/export.py:export_submission_from_long"
    if max_rows_in_memory <= 0:
        raise_error("EXPORT", location, "max_rows_in_memory invalido (deve ser > 0)", impact="1", examples=[str(max_rows_in_memory)])

    assert_memory_budget(stage="EXPORT", location=location, budget_mb=memory_budget_mb)

    sample_lf = scan_table(
        config=TableReadConfig(
            path=sample_submission_path,
            stage="EXPORT",
            location=location,
        )
    )
    sample_cols = list(sample_lf.collect_schema().names())

    required_sample = ["ID", "resname", "resid"]
    miss = [c for c in required_sample if c not in sample_cols]
    if miss:
        raise_error("EXPORT", location, "sample_submission sem colunas obrigatorias", impact=str(len(miss)), examples=miss)

    model_ids = _extract_model_ids(sample_cols, location=location)
    coord_cols = [f"{axis}_{mid}" for mid in model_ids for axis in ("x", "y", "z")]

    preds_lf = scan_table(
        config=TableReadConfig(
            path=predictions_long_path,
            stage="EXPORT",
            location=location,
            columns=("ID", "resid", "resname", "model_id", "x", "y", "z"),
        )
    ).select(
        pl.col("ID").cast(pl.Utf8),
        pl.col("resid").cast(pl.Int32),
        pl.col("resname").cast(pl.Utf8),
        pl.col("model_id").cast(pl.Int32),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
    )

    dup_lf = preds_lf.group_by(["ID", "model_id"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    dup_n = _count_rows(lf=dup_lf, location=location)
    if dup_n > 0:
        examples = collect_streaming(
            lf=dup_lf.select((pl.col("ID") + pl.lit(":") + pl.col("model_id").cast(pl.Utf8)).alias("k")).head(8),
            stage="EXPORT",
            location=location,
        ).get_column("k").to_list()
        raise_error("EXPORT", location, "predictions_long com chave duplicada", impact=str(dup_n), examples=examples)

    found_models = collect_streaming(
        lf=preds_lf.select(pl.col("model_id").unique().sort()),
        stage="EXPORT",
        location=location,
    ).get_column("model_id").to_list()
    found_models = [int(x) for x in found_models]
    if found_models != model_ids:
        raise_error(
            "EXPORT",
            location,
            "model_id da predicao nao bate com sample_submission",
            impact=f"expected={len(model_ids)} got={len(found_models)}",
            examples=[f"expected={model_ids}", f"found={found_models}"],
        )

    sample_ids_lf = sample_lf.select(pl.col("ID").cast(pl.Utf8)).unique()
    pred_ids_lf = preds_lf.select("ID").unique()

    missing_id_lf = sample_ids_lf.join(pred_ids_lf, on="ID", how="anti")
    extra_id_lf = pred_ids_lf.join(sample_ids_lf, on="ID", how="anti")
    missing_id_n = _count_rows(lf=missing_id_lf, location=location)
    extra_id_n = _count_rows(lf=extra_id_lf, location=location)
    if missing_id_n > 0 or extra_id_n > 0:
        examples = [f"missing_id:{r[0]}" for r in collect_streaming(lf=missing_id_lf.head(4), stage="EXPORT", location=location).iter_rows()]
        examples += [f"extra_id:{r[0]}" for r in collect_streaming(lf=extra_id_lf.head(4), stage="EXPORT", location=location).iter_rows()]
        raise_error(
            "EXPORT",
            location,
            "IDs da predicao nao batem com sample",
            impact=f"missing={missing_id_n} extra={extra_id_n}",
            examples=examples,
        )

    for mid in model_ids:
        pred_mid = preds_lf.filter(pl.col("model_id") == mid).select("ID").unique()
        missing_mid_lf = sample_ids_lf.join(pred_mid, on="ID", how="anti")
        extra_mid_lf = pred_mid.join(sample_ids_lf, on="ID", how="anti")
        missing_mid_n = _count_rows(lf=missing_mid_lf, location=location)
        extra_mid_n = _count_rows(lf=extra_mid_lf, location=location)
        if missing_mid_n > 0 or extra_mid_n > 0:
            examples = [f"missing:{r[0]}:{mid}" for r in collect_streaming(lf=missing_mid_lf.head(4), stage="EXPORT", location=location).iter_rows()]
            examples += [f"extra:{r[0]}:{mid}" for r in collect_streaming(lf=extra_mid_lf.head(4), stage="EXPORT", location=location).iter_rows()]
            raise_error(
                "EXPORT",
                location,
                "chaves da predicao nao batem com sample por modelo",
                impact=f"missing={missing_mid_n} extra={extra_mid_n} model_id={mid}",
                examples=examples,
            )

    sample_meta_lf = sample_lf.select(
        pl.col("ID").cast(pl.Utf8),
        pl.col("resname").cast(pl.Utf8),
        pl.col("resid").cast(pl.Int32),
    )
    mismatch_lf = preds_lf.join(sample_meta_lf, on="ID", how="inner", suffix="_sample").filter(
        (pl.col("resname") != pl.col("resname_sample")) | (pl.col("resid") != pl.col("resid_sample"))
    )
    mismatch_n = _count_rows(lf=mismatch_lf, location=location)
    if mismatch_n > 0:
        examples = collect_streaming(
            lf=mismatch_lf.select("ID").head(8),
            stage="EXPORT",
            location=location,
        ).get_column("ID").to_list()
        raise_error(
            "EXPORT",
            location,
            "resname/resid da predicao divergem do sample",
            impact=str(mismatch_n),
            examples=examples,
        )

    agg_exprs: list[pl.Expr] = []
    for mid in model_ids:
        agg_exprs.append(pl.when(pl.col("model_id") == mid).then(pl.col("x")).otherwise(None).max().alias(f"x_{mid}"))
        agg_exprs.append(pl.when(pl.col("model_id") == mid).then(pl.col("y")).otherwise(None).max().alias(f"y_{mid}"))
        agg_exprs.append(pl.when(pl.col("model_id") == mid).then(pl.col("z")).otherwise(None).max().alias(f"z_{mid}"))

    wide_lf = preds_lf.group_by("ID").agg(agg_exprs)

    sample_meta_ordered_lf = sample_lf.select(
        pl.col("ID").cast(pl.Utf8),
        pl.col("resname").cast(pl.Utf8),
        pl.col("resid").cast(pl.Int32),
    ).with_row_index("_ord")

    out_lf = (
        sample_meta_ordered_lf.join(wide_lf, on="ID", how="left")
        .sort("_ord")
        .drop("_ord")
        .select([pl.col(c) for c in sample_cols])
    )

    null_coord_n = int(
        collect_streaming(
            lf=out_lf.select(pl.any_horizontal([pl.col(c).is_null() for c in coord_cols]).sum().alias("n")),
            stage="EXPORT",
            location=location,
        ).get_column("n")[0]
    )
    if null_coord_n > 0:
        examples = collect_streaming(
            lf=out_lf.filter(pl.any_horizontal([pl.col(c).is_null() for c in coord_cols])).select("ID").head(8),
            stage="EXPORT",
            location=location,
        ).get_column("ID").to_list()
        raise_error(
            "EXPORT",
            location,
            "submissao exportada contem nulos",
            impact=str(null_coord_n),
            examples=examples,
        )

    out_submission_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out_path = out_submission_path.with_suffix(out_submission_path.suffix + ".tmp")
    if tmp_out_path.exists():
        tmp_out_path.unlink()

    out_lf.sink_csv(tmp_out_path)
    if out_submission_path.exists():
        out_submission_path.unlink()
    tmp_out_path.replace(out_submission_path)

    validate_submission_against_sample(sample_path=sample_submission_path, submission_path=out_submission_path)
    assert_memory_budget(stage="EXPORT", location=location, budget_mb=memory_budget_mb)
    return out_submission_path
