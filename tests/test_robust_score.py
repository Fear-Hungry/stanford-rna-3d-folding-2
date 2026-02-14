from __future__ import annotations

import json
from pathlib import Path

import pytest

from rna3d_local.errors import PipelineError
from rna3d_local import robust_score as rs


def test_summarize_scores_uses_public_cv_and_p25_components() -> None:
    summary = rs.summarize_scores(
        named_scores={
            "public_validation": 0.31,
            "cv:fold0": 0.29,
            "cv:fold1": 0.27,
            "aux": 0.35,
        },
        public_score_name="public_validation",
    )
    assert summary["cv_count"] == 2
    assert summary["cv_mean_score"] == pytest.approx(0.28)
    assert summary["p25_score"] == pytest.approx(0.285)
    # robust score is conservative min(public, cv_mean, p25)
    assert summary["robust_score"] == pytest.approx(0.28)


def test_evaluate_robust_gate_blocks_without_strict_improvement() -> None:
    report = rs.evaluate_robust_gate(
        named_scores={"public_validation": 0.30, "cv:fold0": 0.29, "cv:fold1": 0.31},
        baseline_robust_score=0.30,
        min_robust_improvement=0.0,
    )
    assert report["allowed"] is False
    assert "melhora estrita" in report["reasons"][0]


def test_evaluate_robust_gate_applies_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_build_calibration(*, competition: str, page_size: int, local_overrides_path: Path | None = None) -> dict:
        assert competition == "stanford-rna-3d-folding-2"
        assert page_size == 10
        assert local_overrides_path is None
        return {"stats": {"n_pairs": 5}}

    def _fake_alignment(**kwargs) -> dict:
        assert kwargs["local_score"] == pytest.approx(0.31)
        assert kwargs["baseline_public_score"] == pytest.approx(0.30)
        return {"allowed": True, "expected_public_score": 0.32, "required_threshold": 0.30}

    monkeypatch.setattr(rs, "build_kaggle_local_calibration", _fake_build_calibration)
    monkeypatch.setattr(rs, "build_alignment_decision", _fake_alignment)

    report = rs.evaluate_robust_gate(
        named_scores={"public_validation": 0.31, "cv:fold0": 0.29, "cv:fold1": 0.30},
        baseline_robust_score=0.28,
        competition="stanford-rna-3d-folding-2",
        baseline_public_score=0.30,
        calibration_page_size=10,
    )
    assert report["allowed"] is True
    assert report["alignment_decision"]["allowed"] is True


def test_evaluate_robust_gate_blocks_when_cv_count_below_min() -> None:
    report = rs.evaluate_robust_gate(
        named_scores={"public_validation": 0.31, "cv:fold0": 0.29},
        min_cv_count=2,
    )
    assert report["allowed"] is False
    assert any("cv_count insuficiente" in r for r in report["reasons"])


def test_evaluate_robust_gate_blocks_public_validation_without_cv() -> None:
    report = rs.evaluate_robust_gate(
        named_scores={"public_validation": 0.31},
        min_cv_count=0,
        block_public_validation_without_cv=True,
    )
    assert report["allowed"] is False
    assert any("public_validation sem evidencias de CV" in r for r in report["reasons"])


def test_read_score_json_fail_fast(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"foo": 1}), encoding="utf-8")
    with pytest.raises(PipelineError):
        rs.read_score_json(score_json_path=bad)


def test_read_score_json_requires_metadata_when_competitive(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"score": 0.123, "meta": {"dataset_dir": "x"}}), encoding="utf-8")
    with pytest.raises(PipelineError):
        rs.read_score_json(score_json_path=legacy, require_metadata=True)


def test_read_score_artifact_with_required_metadata_ok(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            {
                "score": 0.321,
                "meta": {
                    "dataset_type": "public_validation",
                    "sample_columns": ["ID", "resname", "resid", "x_1", "y_1", "z_1"],
                    "sample_schema_sha": "abc123",
                    "n_models": 1,
                    "metric_sha256": "m123",
                    "usalign_sha256": "u123",
                    "regime_id": "kaggle_official_5model",
                },
            }
        ),
        encoding="utf-8",
    )
    art = rs.read_score_artifact(score_json_path=good, require_metadata=True)
    assert float(art["score"]) == pytest.approx(0.321)
    assert art["meta"]["regime_id"] == "kaggle_official_5model"
