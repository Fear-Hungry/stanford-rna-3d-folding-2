from __future__ import annotations

from pathlib import Path

import pytest

from rna3d_local.cli import _enforce_submit_hardening
from rna3d_local.errors import PipelineError


def _base_payload() -> dict:
    return {
        "allowed": True,
        "summary": {
            "cv_count": 3,
            "public_score_name": "public_validation",
            "risk_flags": [],
        },
        "alignment_decision": {
            "is_extrapolation": False,
            "local_score": 0.30,
            "local_score_min_seen": 0.20,
            "local_score_max_seen": 0.35,
        },
    }


def test_hardening_requires_robust_report_by_default(tmp_path: Path) -> None:
    with pytest.raises(PipelineError):
        _enforce_submit_hardening(
            location="tests",
            allow_regression=False,
            require_robust_report=True,
            require_min_cv_count=2,
            block_public_validation_without_cv=True,
            block_target_patch=True,
            allow_calibration_extrapolation=False,
            robust_report_path=None,
            robust_payload=None,
            submission_path=tmp_path / "submission.csv",
            message="msg",
        )


def test_hardening_blocks_public_validation_without_cv(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["summary"]["cv_count"] = 0
    payload["summary"]["risk_flags"] = ["public_validation_without_cv"]
    with pytest.raises(PipelineError):
        _enforce_submit_hardening(
            location="tests",
            allow_regression=False,
            require_robust_report=True,
            require_min_cv_count=0,
            block_public_validation_without_cv=True,
            block_target_patch=False,
            allow_calibration_extrapolation=False,
            robust_report_path=tmp_path / "robust_eval.json",
            robust_payload=payload,
            submission_path=tmp_path / "submission.csv",
            message="msg",
        )


def test_hardening_blocks_extrapolation_when_not_allowed(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["alignment_decision"]["is_extrapolation"] = True
    with pytest.raises(PipelineError):
        _enforce_submit_hardening(
            location="tests",
            allow_regression=False,
            require_robust_report=True,
            require_min_cv_count=2,
            block_public_validation_without_cv=False,
            block_target_patch=False,
            allow_calibration_extrapolation=False,
            robust_report_path=tmp_path / "robust_eval.json",
            robust_payload=payload,
            submission_path=tmp_path / "submission.csv",
            message="msg",
        )


def test_hardening_allows_when_all_requirements_pass(tmp_path: Path) -> None:
    payload = _base_payload()
    _enforce_submit_hardening(
        location="tests",
        allow_regression=False,
        require_robust_report=True,
        require_min_cv_count=2,
        block_public_validation_without_cv=True,
        block_target_patch=True,
        allow_calibration_extrapolation=False,
        robust_report_path=tmp_path / "robust_eval.json",
        robust_payload=payload,
        submission_path=tmp_path / "submission.csv",
        message="generic_candidate",
    )


def test_hardening_requires_readiness_report_when_enabled(tmp_path: Path) -> None:
    payload = _base_payload()
    with pytest.raises(PipelineError):
        _enforce_submit_hardening(
            location="tests",
            allow_regression=False,
            require_robust_report=False,
            require_readiness_report=True,
            require_min_cv_count=0,
            block_public_validation_without_cv=False,
            block_target_patch=False,
            allow_calibration_extrapolation=False,
            robust_report_path=None,
            robust_payload=payload,
            readiness_report_path=None,
            readiness_payload=None,
            submission_path=tmp_path / "submission.csv",
            message="msg",
        )


def test_hardening_blocks_when_readiness_report_disallows(tmp_path: Path) -> None:
    payload = _base_payload()
    readiness_payload = {"allowed": False, "reasons": ["cv_count insuficiente"]}
    with pytest.raises(PipelineError):
        _enforce_submit_hardening(
            location="tests",
            allow_regression=False,
            require_robust_report=False,
            require_readiness_report=True,
            require_min_cv_count=0,
            block_public_validation_without_cv=False,
            block_target_patch=False,
            allow_calibration_extrapolation=False,
            robust_report_path=None,
            robust_payload=payload,
            readiness_report_path=tmp_path / "readiness.json",
            readiness_payload=readiness_payload,
            submission_path=tmp_path / "submission.csv",
            message="msg",
        )
