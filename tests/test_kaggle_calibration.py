from __future__ import annotations

import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.kaggle_calibration import build_alignment_decision, estimate_public_from_local


def _base_calibration() -> dict:
    return {
        "stats": {
            "n_pairs": 5,
            "median_delta": 0.02,
            "p10_delta": 0.01,
            "min_delta": -0.03,
            "min_local_score": 0.20,
            "max_local_score": 0.34,
            "linear_fit_public_from_local": {
                "slope": 0.9,
                "intercept": 0.05,
                "r2": 0.8,
            },
        }
    }


def test_estimate_public_from_local_exposes_linear_fit() -> None:
    estimate = estimate_public_from_local(local_score=0.30, calibration=_base_calibration())
    assert estimate["expected_public_median"] == pytest.approx(0.32)
    assert estimate["expected_public_p10"] == pytest.approx(0.31)
    assert estimate["expected_public_worst_seen"] == pytest.approx(0.27)
    assert estimate["expected_public_linear_fit"] == pytest.approx(0.32)


def test_alignment_decision_allows_when_expected_public_beats_threshold() -> None:
    decision = build_alignment_decision(
        local_score=0.30,
        baseline_public_score=0.305,
        calibration=_base_calibration(),
        method="p10",
        min_public_improvement=0.0,
        min_pairs=3,
    )
    assert decision["allowed"] is True
    assert decision["expected_public_score"] == pytest.approx(0.31)
    assert decision["required_threshold"] == pytest.approx(0.305)
    assert decision["is_extrapolation"] is False


def test_alignment_decision_blocks_when_expected_public_not_enough() -> None:
    decision = build_alignment_decision(
        local_score=0.30,
        baseline_public_score=0.315,
        calibration=_base_calibration(),
        method="p10",
        min_public_improvement=0.0,
        min_pairs=3,
    )
    assert decision["allowed"] is False
    assert decision["expected_public_score"] == pytest.approx(0.31)
    assert decision["required_threshold"] == pytest.approx(0.315)


def test_alignment_decision_requires_min_pairs() -> None:
    calibration = _base_calibration()
    calibration["stats"]["n_pairs"] = 2
    with pytest.raises(PipelineError):
        build_alignment_decision(
            local_score=0.30,
            baseline_public_score=0.28,
            calibration=calibration,
            method="p10",
            min_pairs=3,
        )


def test_alignment_decision_linear_fit_requires_fit_data() -> None:
    calibration = _base_calibration()
    calibration["stats"]["linear_fit_public_from_local"] = None
    with pytest.raises(PipelineError):
        build_alignment_decision(
            local_score=0.30,
            baseline_public_score=0.28,
            calibration=calibration,
            method="linear_fit",
            min_pairs=3,
        )


def test_alignment_decision_blocks_on_extrapolation_by_default() -> None:
    decision = build_alignment_decision(
        local_score=0.50,
        baseline_public_score=0.28,
        calibration=_base_calibration(),
        method="p10",
        min_pairs=3,
    )
    assert decision["is_extrapolation"] is True
    assert decision["allowed"] is False
    assert "extrapolacao" in " ".join([str(x) for x in decision.get("reasons", [])]).lower()


def test_alignment_decision_can_allow_extrapolation_when_explicit() -> None:
    decision = build_alignment_decision(
        local_score=0.50,
        baseline_public_score=0.28,
        calibration=_base_calibration(),
        method="p10",
        min_pairs=3,
        allow_extrapolation=True,
    )
    assert decision["is_extrapolation"] is True
    assert decision["allow_extrapolation"] is True
    assert decision["allowed"] is True
