from __future__ import annotations

import sys
import types

import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.kaggle_calibration import build_alignment_decision, estimate_public_from_local
from rna3d_local.kaggle_calibration import build_kaggle_local_calibration


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


def _install_fake_kaggle_api(monkeypatch: pytest.MonkeyPatch, raw_submissions: list[dict]) -> None:
    class _FakeSub:
        def __init__(self, d: dict) -> None:
            self._d = d

        def to_dict(self) -> dict:
            return self._d

    class _FakeKaggleApi:
        def authenticate(self) -> None:  # noqa: D401
            return None

        def competition_submissions(self, competition: str, page_token: str, page_size: int) -> list[_FakeSub]:
            assert competition == "stanford-rna-3d-folding-2"
            assert page_size > 0
            assert page_token == ""
            return [_FakeSub(d) for d in raw_submissions]

    fake_ext = types.ModuleType("kaggle.api.kaggle_api_extended")
    fake_api_pkg = types.ModuleType("kaggle.api")
    fake_kaggle_pkg = types.ModuleType("kaggle")
    fake_ext.KaggleApi = _FakeKaggleApi
    fake_api_pkg.kaggle_api_extended = fake_ext
    fake_kaggle_pkg.api = fake_api_pkg
    monkeypatch.setitem(sys.modules, "kaggle", fake_kaggle_pkg)
    monkeypatch.setitem(sys.modules, "kaggle.api", fake_api_pkg)
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", fake_ext)


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


def test_build_kaggle_local_calibration_ignores_non_complete_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_kaggle_api(
        monkeypatch=monkeypatch,
        raw_submissions=[
            {"status": "error", "ref": "r1", "description": "local=0.10", "publicScore": 0.55, "date": "2026-01-01T00:00:00Z"},
            {"status": "complete", "ref": "r2", "description": "local=0.20", "publicScore": 0.75, "date": "2026-01-02T00:00:00Z"},
        ],
    )
    report = build_kaggle_local_calibration(competition="stanford-rna-3d-folding-2", page_size=10)
    assert report["stats"]["n_pairs"] == 1
    assert report["pairs"][0]["ref"] == "r2"
    assert report["excluded_by_status"] == 1


def test_build_kaggle_local_calibration_skips_pairs_without_local_or_public_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_kaggle_api(
        monkeypatch=monkeypatch,
        raw_submissions=[
            {"status": "complete", "ref": "r1", "description": "local=0.10", "publicScore": 0.55},
            {"status": "complete", "ref": "r2", "description": "no local score here", "publicScore": 0.60},
            {"status": "complete", "ref": "r3", "description": "local=0.30", "publicScore": None},
        ],
    )
    report = build_kaggle_local_calibration(competition="stanford-rna-3d-folding-2", page_size=10)
    assert report["stats"]["n_pairs"] == 1
    assert report["pairs"][0]["ref"] == "r1"
