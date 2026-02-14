from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .errors import raise_error
from .kaggle_calibration import build_alignment_decision, build_kaggle_local_calibration


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_score_json(*, score_json_path: Path, stage: str = "ROBUST", location: str = "src/rna3d_local/robust_score.py:read_score_json") -> float:
    if not score_json_path.exists():
        raise_error(stage, location, "score_json nao encontrado", impact="1", examples=[str(score_json_path)])
    try:
        payload = json.loads(score_json_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise_error(stage, location, "falha ao ler score_json", impact="1", examples=[f"{type(e).__name__}:{e}"])
    if "score" not in payload:
        raise_error(stage, location, "score_json sem campo score", impact="1", examples=[str(score_json_path)])
    try:
        return float(payload["score"])
    except (TypeError, ValueError):
        raise_error(stage, location, "campo score invalido no score_json", impact="1", examples=[str(payload.get("score"))])
    raise AssertionError("unreachable")


def _resolve_public_score(*, named_scores: dict[str, float], public_score_name: str, location: str) -> float | None:
    if public_score_name in named_scores:
        return float(named_scores[public_score_name])
    # Fallback deterministic: first key containing public_score_name.
    matches = [k for k in sorted(named_scores.keys()) if public_score_name in k]
    if not matches:
        return None
    return float(named_scores[matches[0]])


def summarize_scores(
    *,
    named_scores: dict[str, float],
    public_score_name: str = "public_validation",
) -> dict:
    location = "src/rna3d_local/robust_score.py:summarize_scores"
    if not named_scores:
        raise_error("ROBUST", location, "nenhum score informado", impact="0", examples=[])
    ordered = {str(k): float(v) for k, v in sorted(named_scores.items(), key=lambda kv: kv[0])}
    arr = np.asarray([ordered[k] for k in ordered], dtype=np.float64)
    if arr.size <= 0:
        raise_error("ROBUST", location, "nenhum score numerico valido", impact="0", examples=[])

    public_score = _resolve_public_score(named_scores=ordered, public_score_name=str(public_score_name), location=location)
    cv_vals = [float(v) for k, v in ordered.items() if k.startswith("cv:")]
    cv_mean = float(np.mean(np.asarray(cv_vals, dtype=np.float64))) if cv_vals else None
    p25 = float(np.percentile(arr, 25))
    components: list[float] = [float(p25)]
    if public_score is not None:
        components.append(float(public_score))
    if cv_mean is not None:
        components.append(float(cv_mean))
    robust_score = float(np.min(np.asarray(components, dtype=np.float64)))
    risk_flags: list[str] = []
    if int(len(cv_vals)) == 0:
        risk_flags.append("no_cv_scores")
    if public_score is not None and int(arr.size) == 1:
        risk_flags.append("public_only_score")
    if public_score is not None and int(len(cv_vals)) == 0 and str(public_score_name) == "public_validation":
        risk_flags.append("public_validation_without_cv")
    return {
        "n_scores": int(arr.size),
        "scores": ordered,
        "mean_score": float(np.mean(arr)),
        "median_score": float(np.median(arr)),
        "min_score": float(np.min(arr)),
        "max_score": float(np.max(arr)),
        "p25_score": float(p25),
        "std_score": float(np.std(arr)),
        "public_score_name": str(public_score_name),
        "public_score": None if public_score is None else float(public_score),
        "cv_count": int(len(cv_vals)),
        "cv_mean_score": cv_mean,
        "robust_components": components,
        "robust_score": float(robust_score),
        "risk_flags": risk_flags,
    }


def evaluate_robust_gate(
    *,
    named_scores: dict[str, float],
    public_score_name: str = "public_validation",
    baseline_robust_score: float | None = None,
    min_robust_improvement: float = 0.0,
    competition: str | None = None,
    baseline_public_score: float | None = None,
    calibration_method: str = "p10",
    calibration_page_size: int = 100,
    calibration_min_pairs: int = 3,
    calibration_overrides_path: Path | None = None,
    min_public_improvement: float = 0.0,
    min_cv_count: int = 2,
    block_public_validation_without_cv: bool = True,
    allow_calibration_extrapolation: bool = False,
) -> dict:
    location = "src/rna3d_local/robust_score.py:evaluate_robust_gate"
    summary = summarize_scores(named_scores=named_scores, public_score_name=public_score_name)
    reasons: list[str] = []
    robust_score = float(summary["robust_score"])

    try:
        min_cv_count_i = int(min_cv_count)
    except (TypeError, ValueError):
        raise_error("ROBUST", location, "min_cv_count invalido", impact="1", examples=[str(min_cv_count)])
    if min_cv_count_i < 0:
        raise_error("ROBUST", location, "min_cv_count deve ser >= 0", impact="1", examples=[str(min_cv_count_i)])
    cv_count = int(summary.get("cv_count") or 0)
    if cv_count < min_cv_count_i:
        reasons.append(f"cv_count insuficiente ({cv_count} < {min_cv_count_i})")
    risk_flags = [str(x) for x in (summary.get("risk_flags") or [])]
    if bool(block_public_validation_without_cv) and "public_validation_without_cv" in risk_flags:
        reasons.append("candidato depende de public_validation sem evidencias de CV")

    local_gate = None
    if baseline_robust_score is not None:
        threshold = float(baseline_robust_score) + float(min_robust_improvement)
        local_allowed = bool(robust_score > threshold)
        local_gate = {
            "baseline_robust_score": float(baseline_robust_score),
            "min_robust_improvement": float(min_robust_improvement),
            "required_threshold": float(threshold),
            "candidate_robust_score": float(robust_score),
            "allowed": bool(local_allowed),
        }
        if not local_allowed:
            reasons.append(f"robust_score sem melhora estrita ({robust_score:.6f} <= {threshold:.6f})")

    calibration = None
    alignment = None
    if baseline_public_score is not None:
        if competition is None or not str(competition).strip():
            raise_error(
                "ROBUST",
                location,
                "baseline_public_score requer competition para calibracao",
                impact="1",
                examples=[str(competition)],
            )
        local_for_public = summary.get("public_score")
        if local_for_public is None:
            local_for_public = robust_score
        calibration = build_kaggle_local_calibration(
            competition=str(competition),
            page_size=int(calibration_page_size),
            local_overrides_path=calibration_overrides_path,
        )
        alignment = build_alignment_decision(
            local_score=float(local_for_public),
            baseline_public_score=float(baseline_public_score),
            calibration=calibration,
            method=str(calibration_method),
            min_public_improvement=float(min_public_improvement),
            min_pairs=int(calibration_min_pairs),
            allow_extrapolation=bool(allow_calibration_extrapolation),
        )
        if not bool(alignment.get("allowed", False)):
            reasons.append(f"calibracao bloqueou (reasons={alignment.get('reasons')})")

    return {
        "created_utc": _utc_now(),
        "summary": summary,
        "local_gate": local_gate,
        "alignment_decision": alignment,
        "calibration_stats": None if calibration is None else calibration.get("stats"),
        "allowed": len(reasons) == 0,
        "reasons": reasons,
    }


def write_robust_report(*, report: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
