from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from .contracts import validate_submission_against_sample
from .errors import raise_error


def _read_frame(path: Path, *, location: str) -> pl.DataFrame:
    if not path.exists():
        raise_error("EXPORT", location, "arquivo nao encontrado", impact="1", examples=[str(path)])
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=1000)
    raise_error("EXPORT", location, "formato nao suportado (csv/parquet)", impact="1", examples=[str(path)])
    raise AssertionError("unreachable")


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
) -> Path:
    """
    Export strict Kaggle-format submission from long predictions:
    expected columns in long file: ID,resid,resname,model_id,x,y,z
    """
    location = "src/rna3d_local/export.py:export_submission_from_long"
    sample = pl.read_csv(sample_submission_path, infer_schema_length=1000)
    required_sample = ["ID", "resname", "resid"]
    miss = [c for c in required_sample if c not in sample.columns]
    if miss:
        raise_error("EXPORT", location, "sample_submission sem colunas obrigatorias", impact=str(len(miss)), examples=miss)

    model_ids = _extract_model_ids(sample.columns, location=location)
    preds = _read_frame(predictions_long_path, location=location)
    required_preds = ["ID", "resid", "resname", "model_id", "x", "y", "z"]
    miss_preds = [c for c in required_preds if c not in preds.columns]
    if miss_preds:
        raise_error("EXPORT", location, "predictions_long sem coluna obrigatoria", impact=str(len(miss_preds)), examples=miss_preds)

    preds = preds.select(
        pl.col("ID").cast(pl.Utf8),
        pl.col("resid").cast(pl.Int32),
        pl.col("resname").cast(pl.Utf8),
        pl.col("model_id").cast(pl.Int32),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
    )

    dup = preds.group_by(["ID", "model_id"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if dup.height > 0:
        ex = dup.select((pl.col("ID") + pl.lit(":") + pl.col("model_id").cast(pl.Utf8)).alias("k")).get_column("k")
        raise_error("EXPORT", location, "predictions_long com chave duplicada", impact=str(dup.height), examples=ex.head(8).to_list())

    found_models = sorted(set(preds.get_column("model_id").unique().to_list()))
    if found_models != model_ids:
        raise_error(
            "EXPORT",
            location,
            "model_id da predicao nao bate com sample_submission",
            impact=f"expected={len(model_ids)} got={len(found_models)}",
            examples=[f"expected={model_ids}", f"found={found_models}"],
        )

    key_pairs = set(preds.select("ID", "model_id").iter_rows())
    expected_pairs = {(r["ID"], mid) for r in sample.select("ID").to_dicts() for mid in model_ids}
    missing = sorted(expected_pairs - key_pairs)
    extra = sorted(key_pairs - expected_pairs)
    if missing or extra:
        examples = [f"missing:{x[0]}:{x[1]}" for x in missing[:4]] + [f"extra:{x[0]}:{x[1]}" for x in extra[:4]]
        raise_error(
            "EXPORT",
            location,
            "chaves da predicao nao batem com sample",
            impact=f"missing={len(missing)} extra={len(extra)}",
            examples=examples,
        )

    sample_meta = sample.select(pl.col("ID").cast(pl.Utf8), pl.col("resname").cast(pl.Utf8), pl.col("resid").cast(pl.Int32))
    merged = preds.join(sample_meta, on="ID", how="inner", suffix="_sample")
    mismatch = merged.filter((pl.col("resname") != pl.col("resname_sample")) | (pl.col("resid") != pl.col("resid_sample")))
    if mismatch.height > 0:
        ex = mismatch.select("ID").get_column("ID").head(8).to_list()
        raise_error(
            "EXPORT",
            location,
            "resname/resid da predicao divergem do sample",
            impact=str(mismatch.height),
            examples=ex,
        )

    xw = preds.pivot(values="x", index="ID", on="model_id").rename({str(i): f"x_{i}" for i in model_ids})
    yw = preds.pivot(values="y", index="ID", on="model_id").rename({str(i): f"y_{i}" for i in model_ids})
    zw = preds.pivot(values="z", index="ID", on="model_id").rename({str(i): f"z_{i}" for i in model_ids})
    out = sample_meta.join(xw, on="ID", how="left").join(yw, on="ID", how="left").join(zw, on="ID", how="left")

    null_rows = out.select(pl.any_horizontal(pl.all().is_null()).alias("_has_null")).get_column("_has_null")
    if int(null_rows.sum()) > 0:
        bad = out.filter(null_rows).get_column("ID").head(8).to_list()
        raise_error(
            "EXPORT",
            location,
            "submissao exportada contem nulos",
            impact=str(int(null_rows.sum())),
            examples=bad,
        )

    out = out.select(sample.columns)
    out_submission_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_submission_path)
    validate_submission_against_sample(sample_path=sample_submission_path, submission_path=out_submission_path)
    return out_submission_path
