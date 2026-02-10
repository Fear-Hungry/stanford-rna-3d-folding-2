from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.export import export_submission_from_long


def _write_csv(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_csv(path)


def test_export_fails_on_missing_model_rows(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    preds = tmp_path / "preds.parquet"
    out = tmp_path / "submission.csv"

    _write_csv(
        sample,
        [
            {"ID": "A_1", "resname": "A", "resid": 1, "x_1": 0, "y_1": 0, "z_1": 0, "x_2": 0, "y_2": 0, "z_2": 0},
            {"ID": "A_2", "resname": "C", "resid": 2, "x_1": 0, "y_1": 0, "z_1": 0, "x_2": 0, "y_2": 0, "z_2": 0},
        ],
    )
    pl.DataFrame(
        [
            {"ID": "A_1", "resname": "A", "resid": 1, "model_id": 1, "x": 0.1, "y": 0.1, "z": 0.1},
            {"ID": "A_2", "resname": "C", "resid": 2, "model_id": 1, "x": 0.2, "y": 0.2, "z": 0.2},
        ]
    ).write_parquet(preds)

    with pytest.raises(PipelineError) as e:
        export_submission_from_long(sample_submission_path=sample, predictions_long_path=preds, out_submission_path=out)
    assert "model_id da predicao nao bate com sample_submission" in str(e.value)


def test_export_fails_on_duplicate_id_model(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    preds = tmp_path / "preds.parquet"
    out = tmp_path / "submission.csv"

    _write_csv(
        sample,
        [{"ID": "A_1", "resname": "A", "resid": 1, "x_1": 0, "y_1": 0, "z_1": 0}],
    )
    pl.DataFrame(
        [
            {"ID": "A_1", "resname": "A", "resid": 1, "model_id": 1, "x": 0.1, "y": 0.1, "z": 0.1},
            {"ID": "A_1", "resname": "A", "resid": 1, "model_id": 1, "x": 0.2, "y": 0.2, "z": 0.2},
        ]
    ).write_parquet(preds)

    with pytest.raises(PipelineError) as e:
        export_submission_from_long(sample_submission_path=sample, predictions_long_path=preds, out_submission_path=out)
    assert "predictions_long com chave duplicada" in str(e.value)

