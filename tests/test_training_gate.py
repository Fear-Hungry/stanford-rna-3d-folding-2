from __future__ import annotations

import json
from pathlib import Path

import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.training_gate import (
    evaluate_training_gate_from_model_json,
    evaluate_training_overfit_gate,
    write_training_gate_report,
)


def _metrics(*, mae: float, rmse: float, r2: float, spearman: float, pearson: float, n_samples: int) -> dict:
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "spearman": float(spearman),
        "pearson": float(pearson),
        "n_samples": int(n_samples),
    }


def test_training_gate_allows_balanced_model() -> None:
    report = evaluate_training_overfit_gate(
        train_metrics=_metrics(mae=0.12, rmse=0.18, r2=0.62, spearman=0.71, pearson=0.73, n_samples=1200),
        val_metrics=_metrics(mae=0.13, rmse=0.20, r2=0.55, spearman=0.64, pearson=0.66, n_samples=220),
        min_val_samples=64,
    )
    assert report["allowed"] is True
    assert report["reasons"] == []


def test_training_gate_blocks_overfit_rmse_gap() -> None:
    report = evaluate_training_overfit_gate(
        train_metrics=_metrics(mae=0.05, rmse=0.06, r2=0.95, spearman=0.97, pearson=0.96, n_samples=1400),
        val_metrics=_metrics(mae=0.16, rmse=0.19, r2=0.42, spearman=0.40, pearson=0.39, n_samples=180),
        max_rmse_gap_ratio=0.40,
        max_r2_drop=0.30,
    )
    assert report["allowed"] is False
    assert any("rmse_gap_ratio" in x for x in report["reasons"])
    assert any("r2_drop" in x for x in report["reasons"])


def test_training_gate_from_model_json_requires_train_val_blocks(tmp_path: Path) -> None:
    model_path = tmp_path / "qa_model.json"
    model_path.write_text(json.dumps({"version": 1}), encoding="utf-8")
    with pytest.raises(PipelineError):
        evaluate_training_gate_from_model_json(model_json_path=model_path)


def test_training_gate_from_model_json_roundtrip(tmp_path: Path) -> None:
    model_path = tmp_path / "qa_model.json"
    model_path.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "qa_linear",
                "train_metrics": _metrics(mae=0.11, rmse=0.17, r2=0.66, spearman=0.70, pearson=0.72, n_samples=1000),
                "val_metrics": _metrics(mae=0.12, rmse=0.19, r2=0.58, spearman=0.63, pearson=0.64, n_samples=180),
            }
        ),
        encoding="utf-8",
    )
    report = evaluate_training_gate_from_model_json(model_json_path=model_path)
    out = tmp_path / "train_gate_report.json"
    write_training_gate_report(report=report, out_path=out)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["allowed"] is True
    assert payload["model_type"] == "qa_linear"
