from __future__ import annotations

from pathlib import Path

import polars as pl

from .errors import raise_error


def _read_predictions(path: Path, *, location: str) -> pl.DataFrame:
    if not path.exists():
        raise_error("ENSEMBLE", location, "arquivo nao encontrado", impact="1", examples=[str(path)])
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=1000)
    raise_error("ENSEMBLE", location, "formato nao suportado (csv/parquet)", impact="1", examples=[str(path)])
    raise AssertionError("unreachable")


def blend_predictions(
    *,
    tbm_predictions_path: Path,
    rnapro_predictions_path: Path,
    out_path: Path,
    tbm_weight: float = 0.6,
    rnapro_weight: float = 0.4,
) -> Path:
    """
    Deterministic blending of TBM and RNAPro predictions on exact (ID, model_id) keys.
    """
    location = "src/rna3d_local/ensemble.py:blend_predictions"
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

    tbm = _read_predictions(tbm_predictions_path, location=location)
    rnp = _read_predictions(rnapro_predictions_path, location=location)
    required = ["ID", "model_id", "resid", "resname", "x", "y", "z"]
    for col in required:
        if col not in tbm.columns:
            raise_error("ENSEMBLE", location, "TBM sem coluna obrigatoria", impact="1", examples=[col])
        if col not in rnp.columns:
            raise_error("ENSEMBLE", location, "RNAPRO sem coluna obrigatoria", impact="1", examples=[col])

    key_cols = ["ID", "model_id"]
    dup_tbm = tbm.group_by(key_cols).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if dup_tbm.height > 0:
        ex = dup_tbm.select((pl.col("ID") + pl.lit(":") + pl.col("model_id").cast(pl.Utf8)).alias("k")).get_column("k")
        raise_error(
            "ENSEMBLE",
            location,
            "TBM com chave duplicada",
            impact=str(dup_tbm.height),
            examples=ex.head(8).to_list(),
        )
    dup_rnp = rnp.group_by(key_cols).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if dup_rnp.height > 0:
        ex = dup_rnp.select((pl.col("ID") + pl.lit(":") + pl.col("model_id").cast(pl.Utf8)).alias("k")).get_column("k")
        raise_error(
            "ENSEMBLE",
            location,
            "RNAPRO com chave duplicada",
            impact=str(dup_rnp.height),
            examples=ex.head(8).to_list(),
        )

    tbm_keys = set(tbm.select("ID", "model_id").iter_rows())
    rnp_keys = set(rnp.select("ID", "model_id").iter_rows())
    missing = sorted(tbm_keys - rnp_keys)
    extra = sorted(rnp_keys - tbm_keys)
    if missing or extra:
        examples = [f"missing:{k[0]}:{k[1]}" for k in missing[:4]] + [f"extra:{k[0]}:{k[1]}" for k in extra[:4]]
        raise_error(
            "ENSEMBLE",
            location,
            "chaves divergentes entre TBM e RNAPRO",
            impact=f"missing={len(missing)} extra={len(extra)}",
            examples=examples,
        )

    merged = tbm.select(
        "ID",
        "model_id",
        "resid",
        "resname",
        pl.col("x").alias("tbm_x"),
        pl.col("y").alias("tbm_y"),
        pl.col("z").alias("tbm_z"),
    ).join(
        rnp.select(
            "ID",
            "model_id",
            pl.col("resid").alias("rnp_resid"),
            pl.col("resname").alias("rnp_resname"),
            pl.col("x").alias("rnp_x"),
            pl.col("y").alias("rnp_y"),
            pl.col("z").alias("rnp_z"),
        ),
        on=["ID", "model_id"],
        how="inner",
    )
    mismatch = merged.filter((pl.col("resid") != pl.col("rnp_resid")) | (pl.col("resname") != pl.col("rnp_resname")))
    if mismatch.height > 0:
        ex = mismatch.select((pl.col("ID") + pl.lit(":") + pl.col("model_id").cast(pl.Utf8)).alias("k")).get_column("k")
        raise_error(
            "ENSEMBLE",
            location,
            "metadata de residuo divergente entre TBM e RNAPRO",
            impact=str(mismatch.height),
            examples=ex.head(8).to_list(),
        )

    out = merged.select(
        pl.lit("ensemble").alias("branch"),
        pl.col("ID"),
        pl.col("model_id"),
        pl.col("resid"),
        pl.col("resname"),
        ((pl.col("tbm_x") * tbm_weight + pl.col("rnp_x") * rnapro_weight) / denom).alias("x"),
        ((pl.col("tbm_y") * tbm_weight + pl.col("rnp_y") * rnapro_weight) / denom).alias("y"),
        ((pl.col("tbm_z") * tbm_weight + pl.col("rnp_z") * rnapro_weight) / denom).alias("z"),
    ).with_columns(pl.col("ID").str.extract(r"^(.*)_\d+$", 1).alias("target_id"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.sort(["target_id", "model_id", "resid"]).write_parquet(out_path)
    return out_path

