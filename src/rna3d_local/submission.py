from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .contracts import validate_submission_against_sample
from .errors import raise_error
from .io_tables import read_table


@dataclass(frozen=True)
class SubmissionExportResult:
    submission_path: Path


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
    stage = "EXPORT"
    location = "src/rna3d_local/submission.py:export_submission"
    sample = read_table(sample_path, stage=stage, location=location)
    pred = read_table(predictions_long_path, stage=stage, location=location)
    required = ["target_id", "model_id", "resid", "resname", "x", "y", "z"]
    missing = [column for column in required if column not in pred.columns]
    if missing:
        raise_error(stage, location, "predictions long sem coluna obrigatoria", impact=str(len(missing)), examples=missing[:8])

    model_cols = [column for column in sample.columns if column.startswith("x_")]
    model_ids = sorted(int(column.split("_", 1)[1]) for column in model_cols)
    if not model_ids:
        raise_error(stage, location, "sample sem colunas de modelo", impact="1", examples=sample.columns[:8])

    key_map: dict[tuple[str, int, int], tuple[float, float, float, str]] = {}
    for row in pred.select("target_id", "model_id", "resid", "x", "y", "z", "resname").iter_rows():
        key = (str(row[0]), int(row[1]), int(row[2]))
        if key in key_map:
            raise_error(stage, location, "predictions long com chave duplicada", impact="1", examples=[f"{key[0]}:{key[1]}:{key[2]}"])
        key_map[key] = (float(row[3]), float(row[4]), float(row[5]), str(row[6]))

    out_rows: list[dict[str, object]] = []
    for sample_row in sample.to_dicts():
        key = str(sample_row["ID"])
        target_id = _target_id_from_key(key)
        resid = _resid_from_key(key)
        row = {"ID": key, "resname": sample_row["resname"], "resid": sample_row["resid"]}
        for model_id in model_ids:
            k = (target_id, model_id, resid)
            if k not in key_map:
                raise_error(stage, location, "predictions long com chave faltante para sample", impact="1", examples=[f"{target_id}:{model_id}:{resid}"])
            x, y, z, _pred_resname = key_map[k]
            row[f"x_{model_id}"] = x
            row[f"y_{model_id}"] = y
            row[f"z_{model_id}"] = z
        out_rows.append(row)

    out = pl.DataFrame(out_rows).select(sample.columns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_path)
    validate_submission_against_sample(sample_path=sample_path, submission_path=out_path)
    return SubmissionExportResult(submission_path=out_path)


def check_submission(*, sample_path: Path, submission_path: Path) -> None:
    validate_submission_against_sample(sample_path=sample_path, submission_path=submission_path)
