from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .errors import raise_error
from .kaggle_calibration import build_alignment_decision, build_kaggle_local_calibration
from .robust_score import summarize_scores


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _coerce_nonnegative_int(*, value: int, stage: str, location: str, field: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise_error(stage, location, f"{field} invalido", impact="1", examples=[str(value)])
    if out < 0:
        raise_error(stage, location, f"{field} deve ser >= 0", impact="1", examples=[str(out)])
    return out


def _coerce_nonnegative_float(*, value: float, stage: str, location: str, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise_error(stage, location, f"{field} invalido", impact="1", examples=[str(value)])
    if not np.isfinite(out) or out < 0.0:
        raise_error(stage, location, f"{field} deve ser >= 0 e finito", impact="1", examples=[str(out)])
    return out


def evaluate_submit_readiness(
    *,
    candidate_scores: dict[str, float],
    baseline_scores: dict[str, float] | None = None,
    public_score_name: str = "public_validation",
    require_baseline: bool = True,
    require_public_score: bool = True,
    min_cv_count: int = 3,
    min_cv_improvement_count: int = 2,
    min_fold_improvement: float = 0.0,
    max_cv_regression: float = 0.0,
    min_robust_improvement: float = 0.0,
    min_public_local_improvement: float = 0.0,
    max_cv_std: float = 0.03,
    max_cv_gap: float = 0.08,
    competition: str | None = None,
    baseline_public_score: float | None = None,
    calibration_method: str = "p10",
    calibration_page_size: int = 100,
    calibration_min_pairs: int = 3,
    calibration_overrides_path: Path | None = None,
    min_public_improvement: float = 0.0,
    allow_calibration_extrapolation: bool = False,
    min_calibration_pearson: float = 0.0,
    min_calibration_spearman: float = 0.0,
    block_public_validation_without_cv: bool = True,
) -> dict:
    location = "src/rna3d_local/submission_readiness.py:evaluate_submit_readiness"

    min_cv_count_i = _coerce_nonnegative_int(value=min_cv_count, stage="READINESS", location=location, field="min_cv_count")
    min_cv_improve_i = _coerce_nonnegative_int(
        value=min_cv_improvement_count,
        stage="READINESS",
        location=location,
        field="min_cv_improvement_count",
    )
    min_fold_improve_f = _coerce_nonnegative_float(
        value=min_fold_improvement,
        stage="READINESS",
        location=location,
        field="min_fold_improvement",
    )
    max_cv_reg_f = _coerce_nonnegative_float(
        value=max_cv_regression,
        stage="READINESS",
        location=location,
        field="max_cv_regression",
    )
    min_robust_improve_f = _coerce_nonnegative_float(
        value=min_robust_improvement,
        stage="READINESS",
        location=location,
        field="min_robust_improvement",
    )
    min_public_local_improve_f = _coerce_nonnegative_float(
        value=min_public_local_improvement,
        stage="READINESS",
        location=location,
        field="min_public_local_improvement",
    )
    max_cv_std_f = _coerce_nonnegative_float(value=max_cv_std, stage="READINESS", location=location, field="max_cv_std")
    max_cv_gap_f = _coerce_nonnegative_float(value=max_cv_gap, stage="READINESS", location=location, field="max_cv_gap")

    candidate_summary = summarize_scores(named_scores=candidate_scores, public_score_name=public_score_name)
    baseline_summary = (
        None
        if baseline_scores is None
        else summarize_scores(named_scores=baseline_scores, public_score_name=public_score_name)
    )

    reasons: list[str] = []

    candidate_cv = {k: float(v) for k, v in candidate_summary["scores"].items() if str(k).startswith("cv:")}
    candidate_cv_vals = np.asarray(list(candidate_cv.values()), dtype=np.float64) if candidate_cv else np.asarray([], dtype=np.float64)
    cv_count = int(candidate_cv_vals.size)
    if cv_count < min_cv_count_i:
        reasons.append(f"cv_count insuficiente ({cv_count} < {min_cv_count_i})")

    if cv_count > 0:
        cv_std = float(np.std(candidate_cv_vals))
        cv_gap = float(np.max(candidate_cv_vals) - np.min(candidate_cv_vals))
        if cv_std > max_cv_std_f:
            reasons.append(f"instabilidade CV alta (std={cv_std:.6f} > {max_cv_std_f:.6f})")
        if cv_gap > max_cv_gap_f:
            reasons.append(f"dispersao CV alta (gap={cv_gap:.6f} > {max_cv_gap_f:.6f})")
    else:
        cv_std = None
        cv_gap = None

    if bool(block_public_validation_without_cv) and ("public_validation_without_cv" in [str(x) for x in candidate_summary.get("risk_flags", [])]):
        reasons.append("candidato depende de public_validation sem evidencias de CV")

    if bool(require_public_score) and candidate_summary.get("public_score") is None:
        reasons.append(f"score publico local ausente ({public_score_name})")

    fold_gate: dict | None = None
    robust_gate: dict | None = None
    local_public_gate: dict | None = None

    if baseline_summary is None:
        if bool(require_baseline):
            reasons.append("baseline_scores ausente para validacao de melhoria")
    else:
        base_robust = float(baseline_summary["robust_score"])
        cand_robust = float(candidate_summary["robust_score"])
        robust_threshold = float(base_robust + min_robust_improve_f)
        robust_allowed = bool(cand_robust > robust_threshold)
        robust_gate = {
            "baseline_robust_score": base_robust,
            "candidate_robust_score": cand_robust,
            "min_robust_improvement": float(min_robust_improve_f),
            "required_threshold": robust_threshold,
            "allowed": robust_allowed,
        }
        if not robust_allowed:
            reasons.append(f"robust_score sem melhora estrita ({cand_robust:.6f} <= {robust_threshold:.6f})")

        baseline_cv = {k: float(v) for k, v in baseline_summary["scores"].items() if str(k).startswith("cv:")}
        missing_baseline_folds = sorted([k for k in candidate_cv.keys() if k not in baseline_cv])
        common_folds = sorted(set(candidate_cv.keys()) & set(baseline_cv.keys()))
        fold_deltas = {k: float(candidate_cv[k] - baseline_cv[k]) for k in common_folds}
        improved_folds = [k for k, d in fold_deltas.items() if d > min_fold_improve_f]
        regressed_folds = [k for k, d in fold_deltas.items() if d < (-max_cv_reg_f)]
        required_improved = min(min_cv_improve_i, int(len(common_folds)))
        fold_allowed = bool(len(improved_folds) >= required_improved and len(regressed_folds) == 0 and len(missing_baseline_folds) == 0)
        fold_gate = {
            "common_folds": common_folds,
            "missing_baseline_folds": missing_baseline_folds,
            "fold_deltas": fold_deltas,
            "improved_folds": improved_folds,
            "regressed_folds": regressed_folds,
            "min_fold_improvement": float(min_fold_improve_f),
            "max_cv_regression": float(max_cv_reg_f),
            "min_cv_improvement_count": int(min_cv_improve_i),
            "required_improved": int(required_improved),
            "allowed": bool(fold_allowed),
        }
        if missing_baseline_folds:
            reasons.append(
                f"baseline sem folds CV correspondentes | missing={len(missing_baseline_folds)}"
            )
        if len(improved_folds) < required_improved:
            reasons.append(
                f"melhora CV insuficiente ({len(improved_folds)} < {required_improved})"
            )
        if regressed_folds:
            samples = [f"{k}:{fold_deltas[k]:.6f}" for k in regressed_folds[:8]]
            reasons.append(f"regressao CV acima do limite | impacto={len(regressed_folds)} | exemplos={','.join(samples)}")

        cand_public = candidate_summary.get("public_score")
        base_public = baseline_summary.get("public_score")
        if cand_public is not None and base_public is not None:
            cand_public_f = float(cand_public)
            base_public_f = float(base_public)
            public_threshold = float(base_public_f + min_public_local_improve_f)
            public_allowed = bool(cand_public_f > public_threshold)
            local_public_gate = {
                "baseline_public_local_score": base_public_f,
                "candidate_public_local_score": cand_public_f,
                "min_public_local_improvement": float(min_public_local_improve_f),
                "required_threshold": public_threshold,
                "allowed": public_allowed,
            }
            if not public_allowed:
                reasons.append(f"public_validation local sem melhora estrita ({cand_public_f:.6f} <= {public_threshold:.6f})")

    calibration = None
    alignment = None
    calibration_health = None
    if baseline_public_score is not None:
        if competition is None or not str(competition).strip():
            raise_error(
                "READINESS",
                location,
                "baseline_public_score requer competition para calibracao",
                impact="1",
                examples=[str(competition)],
            )
        calibration = build_kaggle_local_calibration(
            competition=str(competition),
            page_size=int(calibration_page_size),
            local_overrides_path=calibration_overrides_path,
        )
        stats = calibration.get("stats")
        if not isinstance(stats, dict):
            raise_error("READINESS", location, "calibration sem bloco stats", impact="1", examples=[])
        n_pairs = int(stats.get("n_pairs") or 0)
        pearson = stats.get("pearson_local_public")
        spearman = stats.get("spearman_local_public")
        calibration_health = {
            "n_pairs": int(n_pairs),
            "pearson_local_public": None if pearson is None else float(pearson),
            "spearman_local_public": None if spearman is None else float(spearman),
            "min_calibration_pearson": float(min_calibration_pearson),
            "min_calibration_spearman": float(min_calibration_spearman),
            "allowed": True,
            "reasons": [],
        }
        if n_pairs < int(calibration_min_pairs):
            calibration_health["allowed"] = False
            calibration_health["reasons"].append(f"pares de calibracao insuficientes ({n_pairs} < {int(calibration_min_pairs)})")
        if pearson is None:
            calibration_health["allowed"] = False
            calibration_health["reasons"].append("pearson_local_public ausente")
        elif float(pearson) < float(min_calibration_pearson):
            calibration_health["allowed"] = False
            calibration_health["reasons"].append(
                f"pearson_local_public abaixo do minimo ({float(pearson):.6f} < {float(min_calibration_pearson):.6f})"
            )
        if spearman is None:
            calibration_health["allowed"] = False
            calibration_health["reasons"].append("spearman_local_public ausente")
        elif float(spearman) < float(min_calibration_spearman):
            calibration_health["allowed"] = False
            calibration_health["reasons"].append(
                f"spearman_local_public abaixo do minimo ({float(spearman):.6f} < {float(min_calibration_spearman):.6f})"
            )
        if not bool(calibration_health["allowed"]):
            reasons.extend([str(x) for x in calibration_health["reasons"]])

        local_for_public = candidate_summary.get("public_score")
        if local_for_public is None:
            local_for_public = candidate_summary.get("robust_score")
        if local_for_public is None:
            raise_error(
                "READINESS",
                location,
                "nao foi possivel obter local_score para calibracao",
                impact="1",
                examples=[],
            )
        if n_pairs >= int(calibration_min_pairs) and bool(calibration_health.get("allowed", False)):
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
        "candidate_summary": candidate_summary,
        "baseline_summary": baseline_summary,
        "cv_diagnostics": {
            "cv_count": int(cv_count),
            "cv_std": None if cv_std is None else float(cv_std),
            "cv_gap": None if cv_gap is None else float(cv_gap),
            "max_cv_std": float(max_cv_std_f),
            "max_cv_gap": float(max_cv_gap_f),
        },
        "fold_gate": fold_gate,
        "robust_gate": robust_gate,
        "local_public_gate": local_public_gate,
        "calibration_health": calibration_health,
        "alignment_decision": alignment,
        "calibration_stats": None if calibration is None else calibration.get("stats"),
        "allowed": len(reasons) == 0,
        "reasons": reasons,
    }


def write_submit_readiness_report(*, report: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
