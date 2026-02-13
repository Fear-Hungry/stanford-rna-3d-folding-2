from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.gating import assert_submission_allowed


def _write_submission(path: Path) -> None:
    pl.DataFrame(
        [
            {"ID": "Q1_1", "resname": "A", "resid": 1, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0},
            {"ID": "Q1_2", "resname": "C", "resid": 2, "x_1": 1.0, "y_1": 0.0, "z_1": 0.0},
        ]
    ).write_csv(path)


def test_assert_submission_allowed_rejects_negative_min_improvement(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    score_json = tmp_path / "score.json"
    report_path = tmp_path / "gating_report.json"

    _write_submission(sample)
    _write_submission(submission)
    score_json.write_text(json.dumps({"score": 0.40}), encoding="utf-8")

    with pytest.raises(PipelineError, match="min_improvement invalido"):
        assert_submission_allowed(
            sample_path=sample,
            submission_path=submission,
            report_path=report_path,
            is_smoke=False,
            is_partial=False,
            score_json_path=score_json,
            baseline_score=0.40,
            min_improvement=-0.001,
            allow_regression=False,
        )
