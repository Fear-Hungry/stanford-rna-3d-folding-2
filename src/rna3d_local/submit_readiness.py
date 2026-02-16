from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .contracts import validate_submission_against_sample
from .errors import raise_error
from .utils import rel_or_abs, utc_now_iso, write_json


@dataclass(frozen=True)
class SubmitReadinessResult:
    report_path: Path
    allowed: bool


def evaluate_submit_readiness(
    *,
    repo_root: Path,
    sample_path: Path,
    submission_path: Path,
    score_json_path: Path,
    baseline_score: float,
    report_path: Path,
    fail_on_disallow: bool = True,
) -> SubmitReadinessResult:
    stage = "READINESS"
    location = "src/rna3d_local/submit_readiness.py:evaluate_submit_readiness"
    validate_submission_against_sample(sample_path=sample_path, submission_path=submission_path)
    if not score_json_path.exists():
        raise_error(stage, location, "score_json ausente", impact="1", examples=[str(score_json_path)])
    payload = json.loads(score_json_path.read_text(encoding="utf-8"))
    if "score" not in payload:
        raise_error(stage, location, "score_json sem campo score", impact="1", examples=[str(score_json_path)])
    score = float(payload["score"])
    allowed = score > float(baseline_score)
    reason = "strict_improvement_ok" if allowed else "strict_improvement_failed"

    write_json(
        report_path,
        {
            "created_utc": utc_now_iso(),
            "paths": {
                "sample": rel_or_abs(sample_path, repo_root),
                "submission": rel_or_abs(submission_path, repo_root),
                "score_json": rel_or_abs(score_json_path, repo_root),
            },
            "baseline_score": float(baseline_score),
            "score": score,
            "allowed": bool(allowed),
            "reason": reason,
        },
    )
    if not allowed and fail_on_disallow:
        raise_error(
            stage,
            location,
            "candidato bloqueado por gate de melhoria estrita",
            impact="1",
            examples=[f"score={score}", f"baseline={baseline_score}"],
        )
    return SubmitReadinessResult(report_path=report_path, allowed=allowed)
