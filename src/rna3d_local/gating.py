from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .contracts import validate_submission_against_sample
from .errors import raise_error


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def assert_submission_allowed(
    *,
    sample_path: Path,
    submission_path: Path,
    report_path: Path,
    is_smoke: bool = False,
    is_partial: bool = False,
    score_json_path: Path | None = None,
    baseline_score: float | None = None,
    allow_regression: bool = False,
) -> Path:
    """
    Hard gate before Kaggle submit:
    - strict contract vs sample
    - blocks smoke/partial artifacts
    - optional regression gate against baseline_score
    """
    location = "src/rna3d_local/gating.py:assert_submission_allowed"
    validate_submission_against_sample(sample_path=sample_path, submission_path=submission_path)

    reasons: list[str] = []
    current_score: float | None = None
    if is_smoke:
        reasons.append("artefato de smoke test")
    if is_partial:
        reasons.append("execucao parcial")
    if score_json_path is not None:
        if not score_json_path.exists():
            raise_error("GATE", location, "score_json nao encontrado", impact="1", examples=[str(score_json_path)])
        try:
            payload = json.loads(score_json_path.read_text(encoding="utf-8"))
            current_score = float(payload["score"])
        except Exception as e:  # noqa: BLE001
            raise_error("GATE", location, "falha ao ler score_json", impact="1", examples=[f"{type(e).__name__}:{e}"])
    if baseline_score is not None and current_score is not None:
        if current_score < baseline_score and not allow_regression:
            reasons.append(f"score regressivo ({current_score:.6f} < {baseline_score:.6f})")

    report = {
        "created_utc": _utc_now(),
        "submission": str(submission_path),
        "sample_submission": str(sample_path),
        "is_smoke": bool(is_smoke),
        "is_partial": bool(is_partial),
        "baseline_score": baseline_score,
        "current_score": current_score,
        "allow_regression": bool(allow_regression),
        "allowed": len(reasons) == 0,
        "reasons": reasons,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if reasons:
        raise_error(
            "GATE",
            location,
            "submissao bloqueada pelo gating local",
            impact=str(len(reasons)),
            examples=reasons[:8],
        )
    return report_path

