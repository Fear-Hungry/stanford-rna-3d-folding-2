from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.evaluation.kaggle_oracle import score_local_kaggle_official


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("rows vazio")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_kaggle_oracle_imports_metric_and_scores(tmp_path: Path) -> None:
    metric_py = tmp_path / "metric.py"
    metric_py.write_text(
        "def score(sol_df, sub_df, row_id_column_name='ID'):\n"
        "    assert row_id_column_name == 'ID'\n"
        "    return float(len(sol_df)) / 10.0\n",
        encoding="utf-8",
    )
    gt = tmp_path / "gt.csv"
    sub = tmp_path / "sub.csv"
    _write_csv(gt, [{"ID": "T1_1", "x": 0.0, "y": 0.0, "z": 0.0}])
    _write_csv(sub, [{"ID": "T1_1", "x_1": 0.0, "y_1": 0.0, "z_1": 0.0}])
    score_json = tmp_path / "score.json"
    report = tmp_path / "report.json"
    out = score_local_kaggle_official(
        ground_truth_path=gt,
        submission_path=sub,
        score_json_path=score_json,
        report_path=report,
        metric_path=metric_py,
    )
    assert float(out.score) == pytest.approx(0.1)
    assert score_json.exists()
    assert report.exists()


def test_kaggle_oracle_requires_metric_py(tmp_path: Path) -> None:
    gt = tmp_path / "gt.csv"
    sub = tmp_path / "sub.csv"
    _write_csv(gt, [{"ID": "T1_1", "x": 0.0, "y": 0.0, "z": 0.0}])
    _write_csv(sub, [{"ID": "T1_1", "x_1": 0.0, "y_1": 0.0, "z_1": 0.0}])
    with pytest.raises(PipelineError, match="metric.py"):
        score_local_kaggle_official(
            ground_truth_path=gt,
            submission_path=sub,
            score_json_path=tmp_path / "score.json",
            report_path=tmp_path / "report.json",
            metric_path=tmp_path / "missing_metric.py",
        )

