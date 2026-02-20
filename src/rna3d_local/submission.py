from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .contracts import validate_submission_against_sample
from .errors import raise_error
from .io_tables import read_table


@dataclass(frozen=True)
class SubmissionExportResult:
    submission_path: Path


def _model_ids_from_sample_columns(columns: list[str], *, stage: str, location: str) -> list[int]:
    model_cols = [column for column in columns if column.startswith("x_")]
    model_ids = sorted(int(column.split("_", 1)[1]) for column in model_cols)
    if not model_ids:
        raise_error(stage, location, "sample sem colunas de modelo", impact="1", examples=columns[:8])
    for mid in model_ids:
        for prefix in ("y_", "z_"):
            col = f"{prefix}{mid}"
            if col not in columns:
                raise_error(stage, location, "sample sem coluna obrigatoria de modelo", impact="1", examples=[col])
    return model_ids


def _should_use_streaming_export(*, sample_path: Path, predictions_long_path: Path) -> bool:
    if os.environ.get("RNA3D_EXPORT_STREAMING", "").strip() == "1":
        return True
    if predictions_long_path.is_dir():
        return True
    try:
        total = int(sample_path.stat().st_size) + int(predictions_long_path.stat().st_size)
    except OSError:
        return False
    # Force streaming for any non-empty file to prevent CPU OOM in hidden rerun.
    threshold = int(os.environ.get("RNA3D_EXPORT_STREAMING_THRESHOLD_BYTES", str(1024 * 1024)))
    return total >= threshold


def _partition_predictions_by_target(
    *,
    predictions_long_path: Path,
    out_dir: Path,
    stage: str,
    location: str,
) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Carrega em RAM e salva manualmente por target.
        df = pl.read_parquet(
            str(predictions_long_path),
            columns=["target_id", "model_id", "resid", "x", "y", "z"],
        )
        for target_id, part in df.group_by("target_id", maintain_order=False):
            tid = str(target_id[0] if isinstance(target_id, tuple) else target_id)
            target_dir = out_dir / f"target_id={tid}"
            target_dir.mkdir(parents=True, exist_ok=True)
            part.write_parquet(str(target_dir / "0.parquet"))
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "falha ao particionar predictions por target_id", impact="1", examples=[f"{type(exc).__name__}:{exc}"])


def _load_target_pred_map(
    *,
    pred_part_dir: Path,
    target_id: str,
    model_ids: list[int],
    stage: str,
    location: str,
) -> dict[int, dict[int, tuple[float, float, float]]]:
    tdir = pred_part_dir / f"target_id={target_id}"
    files = sorted(tdir.glob("*.parquet"))
    if not files:
        raise_error(
            stage,
            location,
            "predictions sem particao do target_id",
            impact="1",
            examples=[target_id],
        )
    df = pl.read_parquet([str(p) for p in files])
    required = ["target_id", "model_id", "resid", "x", "y", "z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise_error(stage, location, "predictions particionadas sem coluna obrigatoria", impact=str(len(missing)), examples=missing[:8])

    df = df.select(
        pl.col("target_id").cast(pl.Utf8),
        pl.col("model_id").cast(pl.Int32),
        pl.col("resid").cast(pl.Int32),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
    )
    dup = df.group_by(["model_id", "resid"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if dup.height > 0:
        examples = (
            dup.with_columns((pl.lit(target_id) + pl.lit(":") + pl.col("model_id").cast(pl.Utf8) + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k"))
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(stage, location, "predictions com chave duplicada no target", impact=str(int(dup.height)), examples=[str(x) for x in examples])

    coords: dict[int, dict[int, tuple[float, float, float]]] = {}
    for row in df.iter_rows(named=True):
        resid = int(row["resid"])
        mid = int(row["model_id"])
        if mid not in model_ids:
            continue
        x = float(row["x"])
        y = float(row["y"])
        z = float(row["z"])
        if (not math.isfinite(x)) or (not math.isfinite(y)) or (not math.isfinite(z)):
            raise_error(stage, location, "coordenadas nao-finitas nas predictions", impact="1", examples=[f"{target_id}:{mid}:{resid}"])
        if abs(x) > 1e6 or abs(y) > 1e6 or abs(z) > 1e6:
            raise_error(stage, location, "coordenadas fora do range nas predictions", impact="1", examples=[f"{target_id}:{mid}:{resid}"])
        per_resid = coords.setdefault(resid, {})
        if mid in per_resid:
            raise_error(stage, location, "predictions com chave duplicada no target", impact="1", examples=[f"{target_id}:{mid}:{resid}"])
        per_resid[mid] = (x, y, z)

    # Center coordinates per target/model (translation-only). This keeps values within Kaggle
    # validation bounds without changing relative geometry (TM-score is translation-invariant).
    sums: dict[int, list[float]] = {int(mid): [0.0, 0.0, 0.0] for mid in model_ids}
    counts: dict[int, int] = {int(mid): 0 for mid in model_ids}
    for per_resid in coords.values():
        for mid, (x, y, z) in per_resid.items():
            sums[int(mid)][0] += float(x)
            sums[int(mid)][1] += float(y)
            sums[int(mid)][2] += float(z)
            counts[int(mid)] += 1
    means: dict[int, tuple[float, float, float]] = {}
    for mid in model_ids:
        c = int(counts[int(mid)])
        if c <= 0:
            raise_error(
                stage,
                location,
                "predictions sem coordenadas validas para target_id/model_id",
                impact="1",
                examples=[f"{target_id}:{mid}"],
            )
        sx, sy, sz = sums[int(mid)]
        means[int(mid)] = (sx / c, sy / c, sz / c)
    for per_resid in coords.values():
        for mid, (x, y, z) in list(per_resid.items()):
            mean = means.get(int(mid))
            if mean is None:
                raise_error(
                    stage,
                    location,
                    "media ausente para target_id/model_id durante normalizacao",
                    impact="1",
                    examples=[f"{target_id}:{mid}"],
                )
            mx, my, mz = mean
            per_resid[int(mid)] = (float(x) - mx, float(y) - my, float(z) - mz)
    return coords


def _export_submission_streaming(
    *,
    sample_path: Path,
    predictions_long_path: Path,
    out_path: Path,
) -> SubmissionExportResult:
    stage = "EXPORT"
    location = "src/rna3d_local/submission.py:_export_submission_streaming"

    pred_part_dir: Path | None = None
    if predictions_long_path.is_dir():
        pred_part_dir = predictions_long_path
    else:
        pred_part_dir = out_path.parent / f"{predictions_long_path.stem}_by_target"
        _partition_predictions_by_target(predictions_long_path=predictions_long_path, out_dir=pred_part_dir, stage=stage, location=location)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sample_path.open("r", encoding="utf-8", newline="") as f_sample, out_path.open("w", encoding="utf-8", newline="") as f_out:
            reader = csv.DictReader(f_sample)
            if reader.fieldnames is None:
                raise_error(stage, location, "sample vazio", impact="1", examples=[str(sample_path)])
            header = list(reader.fieldnames)
            if "ID" not in header:
                raise_error(stage, location, "sample sem coluna ID", impact="1", examples=["ID"])
            if "resid" not in header:
                raise_error(stage, location, "sample sem coluna resid", impact="1", examples=["resid"])
            model_ids = _model_ids_from_sample_columns(header, stage=stage, location=location)
            writer = csv.DictWriter(f_out, fieldnames=header)
            writer.writeheader()

            current_target: str | None = None
            current_coords: dict[int, dict[int, tuple[float, float, float]]] | None = None
            for row_index, row in enumerate(reader):
                key = str(row.get("ID", ""))
                if "_" not in key:
                    raise_error(stage, location, "ID invalido (esperado <target>_<resid>)", impact="1", examples=[key])
                target_id = _target_id_from_key(key)
                resid_key = _resid_from_key(key)
                resid_col = int(str(row.get("resid", "")).strip())
                if resid_col != resid_key:
                    raise_error(
                        stage,
                        location,
                        "resid divergente no sample",
                        impact="1",
                        examples=[f"{key}:resid={resid_col}"],
                    )

                if target_id != current_target:
                    current_target = target_id
                    if pred_part_dir is None:
                        raise_error(stage, location, "pred_part_dir ausente no export streaming", impact="1", examples=[target_id])
                    current_coords = _load_target_pred_map(
                        pred_part_dir=pred_part_dir,
                        target_id=target_id,
                        model_ids=model_ids,
                        stage=stage,
                        location=location,
                    )

                assert current_coords is not None
                per_resid = current_coords.get(int(resid_key))
                if per_resid is None:
                    raise_error(
                        stage,
                        location,
                        "predictions sem cobertura de residuo para target",
                        impact="1",
                        examples=[f"{target_id}:{resid_key}", f"row={row_index}"],
                    )
                for mid in model_ids:
                    xyz = per_resid.get(int(mid))
                    if xyz is None:
                        raise_error(
                            stage,
                            location,
                            "predictions sem cobertura de modelo para target/resid",
                            impact="1",
                            examples=[f"{target_id}:{resid_key}:model={mid}"],
                        )
                    x, y, z = xyz
                    row[f"x_{mid}"] = str(float(x))
                    row[f"y_{mid}"] = str(float(y))
                    row[f"z_{mid}"] = str(float(z))
                writer.writerow(row)
    except OSError as exc:
        raise_error(stage, location, "falha de IO ao exportar submissao (streaming)", impact="1", examples=[f"{type(exc).__name__}:{exc}"])

    validate_submission_against_sample(sample_path=sample_path, submission_path=out_path)
    return SubmissionExportResult(submission_path=out_path)


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
    if _should_use_streaming_export(sample_path=sample_path, predictions_long_path=predictions_long_path):
        return _export_submission_streaming(sample_path=sample_path, predictions_long_path=predictions_long_path, out_path=out_path)

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

    # Center per (target_id, model_id) to avoid out-of-bounds translations in strict Kaggle validators.
    means = pred.group_by(["target_id", "model_id"]).agg(
        pl.mean("x").alias("_mx"),
        pl.mean("y").alias("_my"),
        pl.mean("z").alias("_mz"),
    )
    pred = (
        pred.join(means, on=["target_id", "model_id"], how="left")
        .with_columns(
            (pl.col("x") - pl.col("_mx")).alias("x"),
            (pl.col("y") - pl.col("_my")).alias("y"),
            (pl.col("z") - pl.col("_mz")).alias("z"),
        )
        .drop(["_mx", "_my", "_mz"])
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
    coord_cols = [column for column in out.columns if column.startswith(("x_", "y_", "z_"))]
    if coord_cols:
        null_expr = pl.any_horizontal([pl.col(column).is_null() for column in coord_cols]).alias("_has_null")
        bad_coords = out.with_columns(null_expr).filter(pl.col("_has_null")).select("ID")
        if bad_coords.height > 0:
            examples = bad_coords.get_column("ID").head(8).to_list()
            raise_error(
                stage,
                location,
                "submissao com coordenadas nulas apos agregacao",
                impact=str(int(bad_coords.height)),
                examples=[str(x) for x in examples],
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_path)
    validate_submission_against_sample(sample_path=sample_path, submission_path=out_path)
    return SubmissionExportResult(submission_path=out_path)


def check_submission(*, sample_path: Path, submission_path: Path) -> None:
    validate_submission_against_sample(sample_path=sample_path, submission_path=submission_path)
