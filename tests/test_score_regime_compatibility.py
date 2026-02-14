from __future__ import annotations

import pytest

from rna3d_local.cli_commands_common import _prepare_competitive_scores
from rna3d_local.errors import PipelineError


def _artifact(*, score: float, regime: str, schema_sha: str = "s1", n_models: int = 5, metric_sha: str = "m1", usalign_sha: str = "u1") -> dict:
    return {
        "score": float(score),
        "meta": {
            "dataset_type": "public_validation",
            "sample_columns": ["ID", "resname", "resid", "x_1", "y_1", "z_1"],
            "sample_schema_sha": schema_sha,
            "n_models": int(n_models),
            "metric_sha256": metric_sha,
            "usalign_sha256": usalign_sha,
            "regime_id": regime,
        },
        "path": "runs/x/score.json",
    }


def test_prepare_competitive_scores_excludes_other_regimes() -> None:
    named, bundle = _prepare_competitive_scores(
        named_artifacts={
            "public_validation": _artifact(score=0.31, regime="kaggle_official_5model"),
            "cv:fold0": _artifact(score=0.30, regime="kaggle_official_5model"),
            "cv40:fold0": _artifact(score=0.95, regime="custom_n40_deadbeef"),
        },
        public_score_name="public_validation",
        stage="ROBUST",
        location="tests/test_score_regime_compatibility.py:test_prepare_competitive_scores_excludes_other_regimes",
    )
    assert set(named.keys()) == {"public_validation", "cv:fold0"}
    assert bundle["regime_summary"]["competitive_regime_id"] == "kaggle_official_5model"
    assert "cv40:fold0" in bundle["regime_summary"]["excluded_by_regime"]


def test_prepare_competitive_scores_rejects_fingerprint_mismatch_in_same_regime() -> None:
    with pytest.raises(PipelineError):
        _prepare_competitive_scores(
            named_artifacts={
                "public_validation": _artifact(score=0.31, regime="kaggle_official_5model", schema_sha="s1"),
                "cv:fold0": _artifact(score=0.30, regime="kaggle_official_5model", schema_sha="s2"),
            },
            public_score_name="public_validation",
            stage="ROBUST",
            location="tests/test_score_regime_compatibility.py:test_prepare_competitive_scores_rejects_fingerprint_mismatch_in_same_regime",
        )


def test_prepare_competitive_scores_rejects_when_forced_regime_has_no_scores() -> None:
    with pytest.raises(PipelineError):
        _prepare_competitive_scores(
            named_artifacts={
                "cv40:fold0": _artifact(score=0.95, regime="custom_n40_deadbeef"),
            },
            public_score_name="public_validation",
            stage="READINESS",
            location="tests/test_score_regime_compatibility.py:test_prepare_competitive_scores_rejects_when_forced_regime_has_no_scores",
            forced_regime_id="kaggle_official_5model",
        )
