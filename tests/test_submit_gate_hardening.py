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


def test_hardening_requires_robust_report(tmp_path: Path) -> None:
    with pytest.raises(PipelineError):
        _enforce_submit_hardening(
            location="tests",
            require_min_cv_count=2,
            robust_report_path=None,
            robust_payload=None,
            readiness_report_path=tmp_path / "readiness.json",
            readiness_payload={"allowed": True},
            submission_path=tmp_path / "submission.csv",
            message="msg",
        )


def test_hardening_requires_readiness_report(tmp_path: Path) -> None:
    payload = _base_payload()
    with pytest.raises(PipelineError):
        _enforce_submit_hardening(
            location="tests",
            require_min_cv_count=2,
            robust_report_path=tmp_path / "robust_eval.json",
            robust_payload=payload,
            readiness_report_path=None,
            readiness_payload=None,
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
            require_min_cv_count=0,
            robust_report_path=tmp_path / "robust_eval.json",
            robust_payload=payload,
            readiness_report_path=tmp_path / "readiness.json",
            readiness_payload={"allowed": True},
            submission_path=tmp_path / "submission.csv",
            message="msg",
        )


def test_hardening_blocks_extrapolation(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["alignment_decision"]["is_extrapolation"] = True
    with pytest.raises(PipelineError):
        _enforce_submit_hardening(
            location="tests",
            require_min_cv_count=2,
            robust_report_path=tmp_path / "robust_eval.json",
            robust_payload=payload,
            readiness_report_path=tmp_path / "readiness.json",
            readiness_payload={"allowed": True},
            submission_path=tmp_path / "submission.csv",
            message="msg",
        )


def test_hardening_blocks_target_patch_pattern(tmp_path: Path) -> None:
    payload = _base_payload()
    with pytest.raises(PipelineError):
        _enforce_submit_hardening(
            location="tests",
            require_min_cv_count=2,
            robust_report_path=tmp_path / "robust_eval.json",
            robust_payload=payload,
            readiness_report_path=tmp_path / "readiness.json",
            readiness_payload={"allowed": True},
            submission_path=tmp_path / "submission_target_patch.csv",
            message="msg",
        )


def test_hardening_blocks_when_readiness_report_disallows(tmp_path: Path) -> None:
    payload = _base_payload()
    readiness_payload = {"allowed": False, "reasons": ["cv_count insuficiente"]}
    with pytest.raises(PipelineError):
        _enforce_submit_hardening(
            location="tests",
            require_min_cv_count=2,
            robust_report_path=tmp_path / "robust_eval.json",
            robust_payload=payload,
            readiness_report_path=tmp_path / "readiness.json",
            readiness_payload=readiness_payload,
            submission_path=tmp_path / "submission.csv",
            message="msg",
        )


def test_hardening_blocks_when_robust_report_disallows(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["allowed"] = False
    with pytest.raises(PipelineError):
        _enforce_submit_hardening(
            location="tests",
            require_min_cv_count=2,
            robust_report_path=tmp_path / "robust_eval.json",
            robust_payload=payload,
            readiness_report_path=tmp_path / "readiness.json",
            readiness_payload={"allowed": True},
            submission_path=tmp_path / "submission.csv",
            message="msg",
        )


def test_hardening_allows_when_all_requirements_pass(tmp_path: Path) -> None:
    payload = _base_payload()
    _enforce_submit_hardening(
        location="tests",
        require_min_cv_count=2,
        robust_report_path=tmp_path / "robust_eval.json",
        robust_payload=payload,
        readiness_report_path=tmp_path / "readiness.json",
        readiness_payload={"allowed": True},
        submission_path=tmp_path / "submission.csv",
        message="generic_candidate",
    )
