from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .contracts import validate_submission_against_sample
from .errors import raise_error
from .io_tables import read_table


@dataclass(frozen=True)
class SubmissionExportResult:
    submission_path: Path


_DUMMY_RESID_MOD = 300
_DUMMY_COORD_SPACING = 3.0


def _dummy_coords_for_resid(resid_key: int) -> tuple[float, float, float]:
    # Deterministic "linha reta" fallback keeps CSV valid when a target/model is missing.
    resid_bucket = int(abs(int(resid_key))) % int(_DUMMY_RESID_MOD)
    return (float(resid_bucket) * float(_DUMMY_COORD_SPACING), 0.0, 0.0)


def _warn_submission_survival(*, stage: str, location: str, cause: str, impact: str, examples: list[str]) -> None:
    print(
        f"[{stage}] [{location}] {cause} | impacto={impact} | exemplos={','.join(examples[:8])}",
        file=sys.stderr,
    )


def _safe_resid_for_dummy(*, row: dict[str, str], key: str, row_index: int) -> int:
    try:
        if "_" in key:
            return int(key.rsplit("_", 1)[1])
    except Exception:
        pass
    try:
        return int(str(row.get("resid", "")).strip())
    except Exception:
        return int(row_index + 1)


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
        raise_error(stage, location, "diretorio de particionamento nao-vazio", impact="1", examples=[str(out_dir)])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        lf = pl.scan_parquet(str(predictions_long_path)).select(
            pl.col("target_id").cast(pl.Utf8),
            pl.col("model_id").cast(pl.Int32),
            pl.col("resid").cast(pl.Int32),
            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
            pl.col("z").cast(pl.Float64),
        )
        scheme = pl.PartitionByKey(out_dir, by="target_id", include_key=True)
        lf.sink_parquet(scheme, mkdir=True, engine="streaming")
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
        _warn_submission_survival(
            stage=stage,
            location=location,
            cause="predictions sem particao do target_id; aplicando coordenadas dummy",
            impact="1",
            examples=[target_id],
        )
        return {}
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
            continue
        sx, sy, sz = sums[int(mid)]
        means[int(mid)] = (sx / c, sy / c, sz / c)
    if not means:
        _warn_submission_survival(
            stage=stage,
            location=location,
            cause="predictions sem coordenadas validas para target_id; aplicando coordenadas dummy",
            impact="1",
            examples=[target_id],
        )
        return {}
    for per_resid in coords.values():
        for mid, (x, y, z) in list(per_resid.items()):
            mean = means.get(int(mid))
            if mean is None:
                continue
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
    force_dummy_all_targets = False
    if predictions_long_path.is_dir():
        pred_part_dir = predictions_long_path
    else:
        pred_part_dir = out_path.parent / f"{predictions_long_path.stem}_by_target"
        try:
            _partition_predictions_by_target(predictions_long_path=predictions_long_path, out_dir=pred_part_dir, stage=stage, location=location)
        except Exception as exc:  # noqa: BLE001
            force_dummy_all_targets = True
            _warn_submission_survival(
                stage=stage,
                location=location,
                cause="falha ao particionar predictions; forçando dummy para todos os alvos",
                impact="1",
                examples=[f"{type(exc).__name__}:{exc}"],
            )

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
            dummy_rows = 0
            dummy_cells = 0
            dummy_examples: list[str] = []
            for row_index, row in enumerate(reader):
                try:
                    key = str(row.get("ID", ""))
                    if "_" not in key:
                        raise ValueError(f"ID invalido: {key}")
                    target_id = _target_id_from_key(key)
                    resid_key = _resid_from_key(key)
                    resid_col = int(str(row.get("resid", "")).strip())
                    if resid_col != resid_key:
                        raise ValueError(f"resid divergente no sample: {key}:resid={resid_col}")

                    if target_id != current_target:
                        current_target = target_id
                        if force_dummy_all_targets:
                            current_coords = {}
                        else:
                            if pred_part_dir is None:
                                current_coords = {}
                            else:
                                try:
                                    current_coords = _load_target_pred_map(
                                        pred_part_dir=pred_part_dir,
                                        target_id=target_id,
                                        model_ids=model_ids,
                                        stage=stage,
                                        location=location,
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    current_coords = {}
                                    _warn_submission_survival(
                                        stage=stage,
                                        location=location,
                                        cause="falha ao carregar predicoes do alvo; aplicando dummy no alvo",
                                        impact="1",
                                        examples=[f"{target_id}:{type(exc).__name__}:{exc}"],
                                    )

                    assert current_coords is not None
                    per_resid = current_coords.get(int(resid_key))
                    used_dummy = False
                    if per_resid is None:
                        per_resid = {}
                        used_dummy = True
                    for mid in model_ids:
                        if mid in per_resid:
                            x, y, z = per_resid[mid]
                        else:
                            x, y, z = _dummy_coords_for_resid(int(resid_key))
                            dummy_cells += 1
                            used_dummy = True
                            if len(dummy_examples) < 8:
                                dummy_examples.append(f"{target_id}:{mid}:{resid_key}")
                        row[f"x_{mid}"] = str(float(x))
                        row[f"y_{mid}"] = str(float(y))
                        row[f"z_{mid}"] = str(float(z))
                    if used_dummy:
                        dummy_rows += 1
                    writer.writerow(row)
                except Exception as exc:  # noqa: BLE001
                    key = str(row.get("ID", ""))
                    resid_key = _safe_resid_for_dummy(row=row, key=key, row_index=row_index)
                    target_id = _target_id_from_key(key) if "_" in key else f"row_{int(row_index)}"
                    for mid in model_ids:
                        x, y, z = _dummy_coords_for_resid(int(resid_key))
                        row[f"x_{mid}"] = str(float(x))
                        row[f"y_{mid}"] = str(float(y))
                        row[f"z_{mid}"] = str(float(z))
                        dummy_cells += 1
                        if len(dummy_examples) < 8:
                            dummy_examples.append(f"{target_id}:{mid}:{resid_key}")
                    dummy_rows += 1
                    writer.writerow(row)
                    _warn_submission_survival(
                        stage=stage,
                        location=location,
                        cause="erro por linha durante export; linha preenchida com dummy",
                        impact="1",
                        examples=[f"{target_id}:{type(exc).__name__}:{exc}"],
                    )
            if dummy_cells > 0:
                _warn_submission_survival(
                    stage=stage,
                    location=location,
                    cause="submissao exportada com lacunas preenchidas por coordenadas dummy",
                    impact=str(dummy_cells),
                    examples=dummy_examples[:8] + [f"rows={dummy_rows}"],
                )
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
    # Kaggle hidden sets can contain pathological targets; default to per-target fail-safe export.
    if os.environ.get("RNA3D_FAILSAFE_PER_TARGET", "1").strip() != "0":
        return _export_submission_streaming(sample_path=sample_path, predictions_long_path=predictions_long_path, out_path=out_path)
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
        _warn_submission_survival(
            stage=stage,
            location=location,
            cause="predictions long com chave faltante para sample; aplicando coordenadas dummy",
            impact=str(int(missing_rows.height)),
            examples=[str(x) for x in examples],
        )

    aggs: list[pl.Expr] = [pl.first("ID").alias("ID"), pl.first("resname").alias("resname"), pl.first("resid").alias("resid")]
    for mid in model_ids:
        k = int(mid)
        resid_dummy_x = (
            (pl.first("resid_key").cast(pl.Int64).abs() % pl.lit(int(_DUMMY_RESID_MOD))).cast(pl.Float64) * pl.lit(float(_DUMMY_COORD_SPACING))
        )
        aggs.extend(
            [
                pl.when(pl.col("model_id") == k).then(pl.col("x")).max().fill_null(resid_dummy_x).alias(f"x_{k}"),
                pl.when(pl.col("model_id") == k).then(pl.col("y")).max().fill_null(pl.lit(0.0)).alias(f"y_{k}"),
                pl.when(pl.col("model_id") == k).then(pl.col("z")).max().fill_null(pl.lit(0.0)).alias(f"z_{k}"),
            ]
        )
    out = joined.group_by("_row", maintain_order=True).agg(aggs).sort("_row").drop("_row").select(out_cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_path)
    validate_submission_against_sample(sample_path=sample_path, submission_path=out_path)
    return SubmissionExportResult(submission_path=out_path)


def check_submission(*, sample_path: Path, submission_path: Path) -> None:
    validate_submission_against_sample(sample_path=sample_path, submission_path=submission_path)
