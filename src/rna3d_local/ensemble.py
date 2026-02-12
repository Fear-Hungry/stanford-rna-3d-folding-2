from __future__ import annotations

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
from .errors import raise_error


def _count_rows(*, lf: pl.LazyFrame, location: str) -> int:
    return int(
        collect_streaming(
            lf=lf.select(pl.len().alias("n")),
            stage="ENSEMBLE",
            location=location,
        ).get_column("n")[0]
    )


def _prediction_lf(*, path: Path, location: str, require_coverage: bool) -> pl.LazyFrame:
    cols = ("ID", "model_id", "resid", "resname", "x", "y", "z")
    if require_coverage:
        cols = cols + ("coverage",)
    select_exprs: list[pl.Expr] = [
        pl.col("ID").cast(pl.Utf8),
        pl.col("model_id").cast(pl.Int32),
        pl.col("resid").cast(pl.Int32),
        pl.col("resname").cast(pl.Utf8),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
    ]
    if require_coverage:
        select_exprs.append(pl.col("coverage").cast(pl.Float64))

    return scan_table(
        config=TableReadConfig(
            path=path,
            stage="ENSEMBLE",
            location=location,
            columns=cols,
        )
    ).select(*select_exprs)


def _validate_no_duplicate_keys(*, lf: pl.LazyFrame, label: str, location: str) -> None:
    dup_lf = lf.group_by(["ID", "model_id"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    dup_n = _count_rows(lf=dup_lf, location=location)
    if dup_n <= 0:
        return
    examples = collect_streaming(
        lf=dup_lf.select((pl.col("ID") + pl.lit(":") + pl.col("model_id").cast(pl.Utf8)).alias("k")).head(8),
        stage="ENSEMBLE",
        location=location,
    ).get_column("k").to_list()
    raise_error(
        "ENSEMBLE",
        location,
        f"{label} com chave duplicada",
        impact=str(dup_n),
        examples=examples,
    )


def blend_predictions(
    *,
    tbm_predictions_path: Path,
    rnapro_predictions_path: Path,
    out_path: Path,
    tbm_weight: float = 0.6,
    rnapro_weight: float = 0.4,
    dynamic_by_coverage: bool = False,
    coverage_power: float = 1.0,
    coverage_floor: float = 1e-6,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    max_rows_in_memory: int = DEFAULT_MAX_ROWS_IN_MEMORY,
) -> Path:
    """
    Deterministic blending of TBM and RNAPro predictions on exact (ID, model_id) keys.
    """
    location = "src/rna3d_local/ensemble.py:blend_predictions"
    if max_rows_in_memory <= 0:
        raise_error("ENSEMBLE", location, "max_rows_in_memory invalido (deve ser > 0)", impact="1", examples=[str(max_rows_in_memory)])
    if tbm_weight <= 0 or rnapro_weight <= 0:
        raise_error(
            "ENSEMBLE",
            location,
            "pesos devem ser > 0",
            impact="1",
            examples=[f"tbm={tbm_weight}", f"rnapro={rnapro_weight}"],
        )
    denom = tbm_weight + rnapro_weight
    if denom <= 0:
        raise_error("ENSEMBLE", location, "soma dos pesos invalida", impact="1", examples=[str(denom)])
    if dynamic_by_coverage and coverage_power < 0:
        raise_error("ENSEMBLE", location, "coverage_power invalido (deve ser >= 0)", impact="1", examples=[str(coverage_power)])
    if dynamic_by_coverage and coverage_floor <= 0:
        raise_error("ENSEMBLE", location, "coverage_floor invalido (deve ser > 0)", impact="1", examples=[str(coverage_floor)])

    assert_memory_budget(stage="ENSEMBLE", location=location, budget_mb=memory_budget_mb)

    tbm_lf = _prediction_lf(path=tbm_predictions_path, location=location, require_coverage=dynamic_by_coverage)
    rnp_lf = _prediction_lf(path=rnapro_predictions_path, location=location, require_coverage=dynamic_by_coverage)

    _validate_no_duplicate_keys(lf=tbm_lf, label="TBM", location=location)
    _validate_no_duplicate_keys(lf=rnp_lf, label="RNAPRO", location=location)

    tbm_keys_lf = tbm_lf.select("ID", "model_id").unique()
    rnp_keys_lf = rnp_lf.select("ID", "model_id").unique()

    missing_df = collect_streaming(
        lf=tbm_keys_lf.join(rnp_keys_lf, on=["ID", "model_id"], how="anti").head(4),
        stage="ENSEMBLE",
        location=location,
    )
    extra_df = collect_streaming(
        lf=rnp_keys_lf.join(tbm_keys_lf, on=["ID", "model_id"], how="anti").head(4),
        stage="ENSEMBLE",
        location=location,
    )
    missing_n = _count_rows(lf=tbm_keys_lf.join(rnp_keys_lf, on=["ID", "model_id"], how="anti"), location=location)
    extra_n = _count_rows(lf=rnp_keys_lf.join(tbm_keys_lf, on=["ID", "model_id"], how="anti"), location=location)
    if missing_n > 0 or extra_n > 0:
        missing = [f"missing:{r[0]}:{r[1]}" for r in missing_df.iter_rows()]
        extra = [f"extra:{r[0]}:{r[1]}" for r in extra_df.iter_rows()]
        raise_error(
            "ENSEMBLE",
            location,
            "chaves divergentes entre TBM e RNAPRO",
            impact=f"missing={missing_n} extra={extra_n}",
            examples=missing + extra,
        )

    tbm_select = [
        "ID",
        "model_id",
        "resid",
        "resname",
        pl.col("x").alias("tbm_x"),
        pl.col("y").alias("tbm_y"),
        pl.col("z").alias("tbm_z"),
    ]
    rnp_select = [
        "ID",
        "model_id",
        pl.col("resid").alias("rnp_resid"),
        pl.col("resname").alias("rnp_resname"),
        pl.col("x").alias("rnp_x"),
        pl.col("y").alias("rnp_y"),
        pl.col("z").alias("rnp_z"),
    ]
    if dynamic_by_coverage:
        tbm_select.append(pl.col("coverage").alias("tbm_cov"))
        rnp_select.append(pl.col("coverage").alias("rnp_cov"))

    merged_lf = tbm_lf.select(*tbm_select).join(
        rnp_lf.select(*rnp_select),
        on=["ID", "model_id"],
        how="inner",
    )

    mismatch_lf = merged_lf.filter((pl.col("resid") != pl.col("rnp_resid")) | (pl.col("resname") != pl.col("rnp_resname")))
    mismatch_n = _count_rows(lf=mismatch_lf, location=location)
    if mismatch_n > 0:
        examples = collect_streaming(
            lf=mismatch_lf.select((pl.col("ID") + pl.lit(":") + pl.col("model_id").cast(pl.Utf8)).alias("k")).head(8),
            stage="ENSEMBLE",
            location=location,
        ).get_column("k").to_list()
        raise_error(
            "ENSEMBLE",
            location,
            "metadata de residuo divergente entre TBM e RNAPRO",
            impact=str(mismatch_n),
            examples=examples,
        )

    if dynamic_by_coverage:
        bad_cov_lf = merged_lf.filter(
            pl.col("tbm_cov").is_null()
            | pl.col("rnp_cov").is_null()
            | (pl.col("tbm_cov") < 0.0)
            | (pl.col("tbm_cov") > 1.0)
            | (pl.col("rnp_cov") < 0.0)
            | (pl.col("rnp_cov") > 1.0)
        )
        bad_cov_n = _count_rows(lf=bad_cov_lf, location=location)
        if bad_cov_n > 0:
            examples = collect_streaming(
                lf=bad_cov_lf.select(
                    (
                        pl.col("ID")
                        + pl.lit(":")
                        + pl.col("model_id").cast(pl.Utf8)
                        + pl.lit(":tbm=")
                        + pl.col("tbm_cov").cast(pl.Utf8)
                        + pl.lit(":rnp=")
                        + pl.col("rnp_cov").cast(pl.Utf8)
                    ).alias("k")
                ).head(8),
                stage="ENSEMBLE",
                location=location,
            ).get_column("k").to_list()
            raise_error(
                "ENSEMBLE",
                location,
                "coverage invalida para blending dinamico",
                impact=str(bad_cov_n),
                examples=examples,
            )

        tbm_w = pl.lit(tbm_weight) * pl.max_horizontal(pl.col("tbm_cov"), pl.lit(coverage_floor)).pow(coverage_power)
        rnp_w = pl.lit(rnapro_weight) * pl.max_horizontal(pl.col("rnp_cov"), pl.lit(coverage_floor)).pow(coverage_power)
        dyn_den = (tbm_w + rnp_w)
        x_expr = ((pl.col("tbm_x") * tbm_w) + (pl.col("rnp_x") * rnp_w)) / dyn_den
        y_expr = ((pl.col("tbm_y") * tbm_w) + (pl.col("rnp_y") * rnp_w)) / dyn_den
        z_expr = ((pl.col("tbm_z") * tbm_w) + (pl.col("rnp_z") * rnp_w)) / dyn_den
    else:
        x_expr = ((pl.col("tbm_x") * tbm_weight + pl.col("rnp_x") * rnapro_weight) / denom)
        y_expr = ((pl.col("tbm_y") * tbm_weight + pl.col("rnp_y") * rnapro_weight) / denom)
        z_expr = ((pl.col("tbm_z") * tbm_weight + pl.col("rnp_z") * rnapro_weight) / denom)

    out_lf = merged_lf.select(
        pl.lit("ensemble").alias("branch"),
        pl.col("ID"),
        pl.col("model_id"),
        pl.col("resid"),
        pl.col("resname"),
        x_expr.alias("x"),
        y_expr.alias("y"),
        z_expr.alias("z"),
    ).with_columns(pl.col("ID").str.extract(r"^(.*)_\d+$", 1).alias("target_id"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_out_path.exists():
        tmp_out_path.unlink()

    out_lf.sink_parquet(tmp_out_path, compression="zstd")

    if out_path.exists():
        out_path.unlink()
    tmp_out_path.replace(out_path)

    assert_memory_budget(stage="ENSEMBLE", location=location, budget_mb=memory_budget_mb)
    return out_path
