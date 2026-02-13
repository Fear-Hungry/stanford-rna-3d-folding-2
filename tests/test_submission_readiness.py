from __future__ import annotations

import pytest

from rna3d_local import submission_readiness as sr


def test_readiness_allows_when_cv_and_robust_improve(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_build_calibration(*, competition: str, page_size: int) -> dict:
        assert competition == "stanford-rna-3d-folding-2"
        assert page_size == 25
        return {
            "stats": {
                "n_pairs": 8,
                "pearson_local_public": 0.25,
                "spearman_local_public": 0.20,
            }
        }

    def _fake_alignment(**kwargs) -> dict:
        assert kwargs["baseline_public_score"] == pytest.approx(0.26)
        return {"allowed": True, "expected_public_score": 0.29, "required_threshold": 0.26}

    monkeypatch.setattr(sr, "build_kaggle_local_calibration", _fake_build_calibration)
    monkeypatch.setattr(sr, "build_alignment_decision", _fake_alignment)

    report = sr.evaluate_submit_readiness(
        candidate_scores={
            "public_validation": 0.31,
            "cv:fold0": 0.95,
            "cv:fold1": 0.96,
            "cv:fold2": 0.94,
        },
        baseline_scores={
            "public_validation": 0.30,
            "cv:fold0": 0.93,
            "cv:fold1": 0.95,
            "cv:fold2": 0.92,
        },
        min_cv_count=3,
        min_cv_improvement_count=2,
        competition="stanford-rna-3d-folding-2",
        baseline_public_score=0.26,
        calibration_page_size=25,
        min_calibration_pearson=0.1,
        min_calibration_spearman=0.1,
    )
    assert report["allowed"] is True
    assert report["fold_gate"]["allowed"] is True
    assert report["robust_gate"]["allowed"] is True


def test_readiness_blocks_on_cv_regression() -> None:
    report = sr.evaluate_submit_readiness(
        candidate_scores={
            "public_validation": 0.31,
            "cv:fold0": 0.90,
            "cv:fold1": 0.92,
            "cv:fold2": 0.90,
        },
        baseline_scores={
            "public_validation": 0.30,
            "cv:fold0": 0.93,
            "cv:fold1": 0.91,
            "cv:fold2": 0.89,
        },
        min_cv_count=3,
        min_cv_improvement_count=2,
        max_cv_regression=0.0,
    )
    assert report["allowed"] is False
    assert any("regressao CV acima do limite" in r for r in report["reasons"])


def test_readiness_blocks_when_calibration_health_is_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_build_calibration(*, competition: str, page_size: int) -> dict:
        return {
            "stats": {
                "n_pairs": 7,
                "pearson_local_public": -0.50,
                "spearman_local_public": -0.40,
            }
        }

    monkeypatch.setattr(sr, "build_kaggle_local_calibration", _fake_build_calibration)

    report = sr.evaluate_submit_readiness(
        candidate_scores={
            "public_validation": 0.31,
            "cv:fold0": 0.95,
            "cv:fold1": 0.96,
            "cv:fold2": 0.97,
        },
        baseline_scores={
            "public_validation": 0.30,
            "cv:fold0": 0.94,
            "cv:fold1": 0.95,
            "cv:fold2": 0.96,
        },
        min_cv_count=3,
        min_cv_improvement_count=2,
        competition="stanford-rna-3d-folding-2",
        baseline_public_score=0.26,
        min_calibration_pearson=0.0,
        min_calibration_spearman=0.0,
    )
    assert report["allowed"] is False
    assert any("pearson_local_public abaixo do minimo" in r for r in report["reasons"])
    assert any("spearman_local_public abaixo do minimo" in r for r in report["reasons"])
