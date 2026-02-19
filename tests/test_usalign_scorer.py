from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.evaluation import score_local_bestof5


def _write_fake_usalign(path: Path) -> None:
    content = """#!/usr/bin/env python3
import os
import re
import sys

pred = os.path.basename(sys.argv[1])
match = re.match(r"(.+)_pred_(\\d+)\\.pdb$", pred)
if match is None:
    print("invalid pred filename", file=sys.stderr)
    sys.exit(2)
target = match.group(1)
model_id = int(match.group(2))
scores = {
    "T1": [0.10, 0.20, 0.30, 0.40, 0.50],
    "T2": [0.90, 0.20, 0.10, 0.05, 0.01],
    "QX": [0.61, 0.62, 0.63, 0.64, 0.65],
}
if target not in scores or model_id < 1 or model_id > 5:
    print("invalid target/model", file=sys.stderr)
    sys.exit(3)
score = scores[target][model_id - 1]
print("pred\\ttrue\\trmsd\\ttm2")
print(f"{pred}\\tgt\\t0.0\\t{score:.6f}")
"""
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _write_fake_usalign_with_gt_copy(path: Path) -> None:
    content = """#!/usr/bin/env python3
import os
import re
import sys

pred = os.path.basename(sys.argv[1])
true = os.path.basename(sys.argv[2])
pm = re.match(r"(.+)_pred_(\\d+)\\.pdb$", pred)
tm = re.match(r"(.+)_gt_(\\d+)\\.pdb$", true)
if pm is None or tm is None:
    print("invalid filenames", file=sys.stderr)
    sys.exit(2)
target = pm.group(1)
model_id = int(pm.group(2))
gt_copy = int(tm.group(2))
scores = {
    "T1": {
        1: [0.10, 0.20, 0.30, 0.40, 0.50],
        2: [0.90, 0.20, 0.10, 0.05, 0.01],
    }
}
if target not in scores or gt_copy not in scores[target] or model_id < 1 or model_id > 5:
    print("invalid target/model/gt_copy", file=sys.stderr)
    sys.exit(3)
score = scores[target][gt_copy][model_id - 1]
print("pred\\ttrue\\trmsd\\ttm2")
print(f"{pred}\\t{true}\\t0.0\\t{score:.6f}")
"""
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _write_submission(path: Path, ids: list[str]) -> None:
    rows: list[dict[str, object]] = []
    for index, key in enumerate(ids, start=1):
        row: dict[str, object] = {"ID": key}
        for model_id in [1, 2, 3, 4, 5]:
            row[f"x_{model_id}"] = float(index + model_id)
            row[f"y_{model_id}"] = float(index + model_id + 1)
            row[f"z_{model_id}"] = float(index + model_id + 2)
        rows.append(row)
    pl.DataFrame(rows).write_csv(path)


def test_score_local_bestof5_success(tmp_path: Path) -> None:
    usalign_bin = tmp_path / "USalign"
    _write_fake_usalign(usalign_bin)
    ground_truth = tmp_path / "ground_truth.csv"
    submission = tmp_path / "submission.csv"
    score_json = tmp_path / "score.json"
    report = tmp_path / "report.json"
    pl.DataFrame(
        [
            {"ID": "T1_1", "resname": "A", "x": 0.0, "y": 0.0, "z": 0.0},
            {"ID": "T1_2", "resname": "C", "x": 1.0, "y": 0.0, "z": 0.0},
            {"ID": "T2_1", "resname": "G", "x": 0.0, "y": 1.0, "z": 0.0},
            {"ID": "T2_2", "resname": "U", "x": 1.0, "y": 1.0, "z": 0.0},
        ]
    ).write_csv(ground_truth)
    _write_submission(submission, ids=["T1_1", "T1_2", "T2_1", "T2_2"])
    result = score_local_bestof5(
        ground_truth_path=ground_truth,
        submission_path=submission,
        usalign_path=usalign_bin,
        score_json_path=score_json,
        report_path=report,
    )
    assert result.n_targets == 2
    assert result.score == pytest.approx(0.70, abs=1e-6)
    score_payload = json.loads(score_json.read_text(encoding="utf-8"))
    assert float(score_payload["score"]) == pytest.approx(0.70, abs=1e-6)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert len(report_payload["targets"]) == 2


def test_score_local_bestof5_fails_on_key_mismatch(tmp_path: Path) -> None:
    usalign_bin = tmp_path / "USalign"
    _write_fake_usalign(usalign_bin)
    ground_truth = tmp_path / "ground_truth.csv"
    submission = tmp_path / "submission.csv"
    score_json = tmp_path / "score.json"
    pl.DataFrame(
        [
            {"ID": "T1_1", "x": 0.0, "y": 0.0, "z": 0.0},
            {"ID": "T1_2", "x": 1.0, "y": 0.0, "z": 0.0},
        ]
    ).write_csv(ground_truth)
    _write_submission(submission, ids=["T1_1"])
    with pytest.raises(PipelineError, match="chaves da submissao nao batem com ground_truth"):
        score_local_bestof5(
            ground_truth_path=ground_truth,
            submission_path=submission,
            usalign_path=usalign_bin,
            score_json_path=score_json,
            report_path=None,
        )


def test_score_local_bestof5_accepts_target_id_resid_and_x1(tmp_path: Path) -> None:
    usalign_bin = tmp_path / "USalign"
    _write_fake_usalign(usalign_bin)
    ground_truth = tmp_path / "ground_truth.csv"
    submission = tmp_path / "submission.csv"
    score_json = tmp_path / "score.json"
    pl.DataFrame(
        [
            {"target_id": "QX", "resid": 1, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0},
            {"target_id": "QX", "resid": 2, "x_1": 1.0, "y_1": 0.0, "z_1": 0.0},
        ]
    ).write_csv(ground_truth)
    _write_submission(submission, ids=["QX_1", "QX_2"])
    result = score_local_bestof5(
        ground_truth_path=ground_truth,
        submission_path=submission,
        usalign_path=usalign_bin,
        score_json_path=score_json,
        report_path=None,
    )
    assert result.n_targets == 1
    assert result.score == pytest.approx(0.65, abs=1e-6)


def test_score_local_bestof5_best_of_gt_copies(tmp_path: Path) -> None:
    usalign_bin = tmp_path / "USalign"
    _write_fake_usalign_with_gt_copy(usalign_bin)
    ground_truth = tmp_path / "ground_truth.csv"
    submission = tmp_path / "submission.csv"
    score_json = tmp_path / "score.json"
    pl.DataFrame(
        [
            {"target_id": "T1", "resid": 1, "resname": "A", "x_1": 0.0, "y_1": 0.0, "z_1": 0.0, "x_2": 1.0, "y_2": 0.0, "z_2": 0.0},
            {"target_id": "T1", "resid": 2, "resname": "C", "x_1": 1.0, "y_1": 0.0, "z_1": 0.0, "x_2": 2.0, "y_2": 0.0, "z_2": 0.0},
        ]
    ).write_csv(ground_truth)
    _write_submission(submission, ids=["T1_1", "T1_2"])
    result = score_local_bestof5(
        ground_truth_path=ground_truth,
        submission_path=submission,
        usalign_path=usalign_bin,
        score_json_path=score_json,
        report_path=None,
        ground_truth_mode="best_of_gt_copies",
    )
    assert result.n_targets == 1
    assert result.score == pytest.approx(0.90, abs=1e-6)


def test_score_local_bestof5_recovers_missing_execute_permission(tmp_path: Path) -> None:
    usalign_bin = tmp_path / "USalign"
    _write_fake_usalign(usalign_bin)
    usalign_bin.chmod(0o644)
    assert os.access(usalign_bin, os.X_OK) is False

    ground_truth = tmp_path / "ground_truth.csv"
    submission = tmp_path / "submission.csv"
    score_json = tmp_path / "score.json"
    pl.DataFrame(
        [
            {"ID": "T1_1", "resname": "A", "x": 0.0, "y": 0.0, "z": 0.0},
            {"ID": "T1_2", "resname": "C", "x": 1.0, "y": 0.0, "z": 0.0},
        ]
    ).write_csv(ground_truth)
    _write_submission(submission, ids=["T1_1", "T1_2"])
    result = score_local_bestof5(
        ground_truth_path=ground_truth,
        submission_path=submission,
        usalign_path=usalign_bin,
        score_json_path=score_json,
        report_path=None,
    )
    assert result.n_targets == 1
    assert os.access(usalign_bin, os.X_OK) is True
