from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .errors import raise_error

_LOCAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\blocal\s*=\s*([0-9]*\.?[0-9]+)\b", re.IGNORECASE),
    re.compile(r"\blocal_public\s*=\s*([0-9]*\.?[0-9]+)\b", re.IGNORECASE),
)
_ALLOWED_CALIBRATION_METHODS = ("median", "p10", "worst_seen", "linear_fit")


def _is_complete_status(*, status: str) -> bool:
    text = str(status or "").strip().lower()
    if not text:
        return True
    return "complete" in text


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_local_score(description: str) -> float | None:
    text = str(description or "")
    for pat in _LOCAL_PATTERNS:
        m = pat.search(text)
        if m is not None:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def _load_local_score_overrides(*, path: Path | None, location: str) -> tuple[dict[str, float], set[str], bool]:
    if path is None:
        return {}, set(), False
    if not path.exists():
        raise_error("CALIBRATION", location, "arquivo de overrides nao encontrado", impact="1", examples=[str(path)])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise_error("CALIBRATION", location, "falha ao ler overrides", impact="1", examples=[f"{type(e).__name__}:{e}"])
    if not isinstance(payload, dict):
        raise_error("CALIBRATION", location, "overrides deve ser objeto JSON", impact="1", examples=[str(path)])
    by_ref_raw = payload.get("by_ref", {})
    if by_ref_raw is None:
        by_ref_raw = {}
    if not isinstance(by_ref_raw, dict):
        raise_error("CALIBRATION", location, "campo by_ref invalido em overrides", impact="1", examples=[str(type(by_ref_raw))])
    by_ref: dict[str, float] = {}
    bad_refs: list[str] = []
    for raw_ref, raw_score in by_ref_raw.items():
        ref = str(raw_ref).strip()
        if not ref:
            bad_refs.append("<empty_ref>")
            continue
        try:
            by_ref[ref] = float(raw_score)
        except (TypeError, ValueError):
            bad_refs.append(ref)
    if bad_refs:
        examples = bad_refs[:8]
        raise_error("CALIBRATION", location, "valores invalidos em overrides.by_ref", impact=str(len(bad_refs)), examples=examples)
    exclude_refs_raw = payload.get("exclude_refs", [])
    if exclude_refs_raw is None:
        exclude_refs_raw = []
    if not isinstance(exclude_refs_raw, list):
        raise_error("CALIBRATION", location, "campo exclude_refs invalido em overrides", impact="1", examples=[str(type(exclude_refs_raw))])
    exclude_refs = {str(x).strip() for x in exclude_refs_raw if str(x).strip()}
    only_override_refs = bool(payload.get("only_override_refs", False))
    return by_ref, exclude_refs, only_override_refs


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if int(x.size) < 2 or int(y.size) < 2:
        return None
    xm = x - np.mean(x)
    ym = y - np.mean(y)
    den = float(np.sqrt(np.sum(xm * xm) * np.sum(ym * ym)))
    if den <= 0.0:
        return None
    return float(np.sum(xm * ym) / den)


def _rankdata_avg(x: np.ndarray) -> np.ndarray:
    n = int(x.size)
    order = np.argsort(x, kind="mergesort")
    sorted_vals = x[order]
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * float(i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if int(x.size) < 2 or int(y.size) < 2:
        return None
    return _pearson_corr(_rankdata_avg(x), _rankdata_avg(y))


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict | None:
    if int(x.size) < 2 or int(y.size) < 2:
        return None
    if float(np.var(x)) <= 0.0:
        return None
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:  # noqa: BLE001
        return None
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = None if ss_tot <= 0.0 else float(1.0 - (ss_res / ss_tot))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
    }


def build_kaggle_local_calibration(
    *,
    competition: str,
    page_size: int = 100,
    local_overrides_path: Path | None = None,
) -> dict:
    location = "src/rna3d_local/kaggle_calibration.py:build_kaggle_local_calibration"
    if int(page_size) <= 0:
        raise_error("CALIBRATION", location, "page_size invalido", impact="1", examples=[str(page_size)])
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:  # noqa: BLE001
        raise_error(
            "CALIBRATION",
            location,
            "falha ao importar KaggleApi",
            impact="1",
            examples=[f"{type(e).__name__}:{e}"],
        )
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:  # noqa: BLE001
        raise_error("CALIBRATION", location, "falha ao autenticar Kaggle API", impact="1", examples=[f"{type(e).__name__}:{e}"])
    try:
        subs = api.competition_submissions(competition=competition, page_token="", page_size=int(page_size))
    except Exception as e:  # noqa: BLE001
        raise_error(
            "CALIBRATION",
            location,
            "falha ao listar submissões da competicao",
            impact="1",
            examples=[f"{type(e).__name__}:{e}", str(competition)],
        )
    if subs is None:
        subs = []
    local_overrides_by_ref, exclude_refs, only_override_refs = _load_local_score_overrides(path=local_overrides_path, location=location)
    pairs: list[dict] = []
    total = 0
    excluded_by_ref = 0
    overridden_by_ref = 0
    excluded_by_scope = 0
    excluded_by_status = 0
    for sub in subs:
        d = sub.to_dict()
        total += 1
        status = str(d.get("status") or "")
        if not _is_complete_status(status=status):
            excluded_by_status += 1
            continue
        desc = str(d.get("description") or "")
        ref_str = str(d.get("ref") or "").strip()
        if only_override_refs and (not ref_str or ref_str not in local_overrides_by_ref):
            excluded_by_scope += 1
            continue
        if ref_str and ref_str in exclude_refs:
            excluded_by_ref += 1
            continue
        local_source = "description"
        local_score = _parse_local_score(desc)
        if ref_str and ref_str in local_overrides_by_ref:
            local_score = float(local_overrides_by_ref[ref_str])
            local_source = "override_ref"
            overridden_by_ref += 1
        pub_raw = d.get("publicScore")
        if local_score is None or pub_raw is None:
            continue
        try:
            public_score = float(pub_raw)
        except (TypeError, ValueError):
            continue
        pairs.append(
            {
                "ref": d.get("ref"),
                "date": d.get("date"),
                "status": status,
                "description": desc,
                "local_source": local_source,
                "local_score": float(local_score),
                "public_score": float(public_score),
                "delta_public_minus_local": float(public_score - local_score),
            }
        )
    deltas = np.asarray([float(p["delta_public_minus_local"]) for p in pairs], dtype=np.float64) if pairs else np.asarray([], dtype=np.float64)
    locals_arr = np.asarray([float(p["local_score"]) for p in pairs], dtype=np.float64) if pairs else np.asarray([], dtype=np.float64)
    public_arr = np.asarray([float(p["public_score"]) for p in pairs], dtype=np.float64) if pairs else np.asarray([], dtype=np.float64)
    if deltas.size == 0:
        stats = {
            "n_pairs": 0,
            "mean_delta": None,
            "median_delta": None,
            "min_delta": None,
            "max_delta": None,
            "p10_delta": None,
            "p90_delta": None,
            "pearson_local_public": None,
            "spearman_local_public": None,
            "linear_fit_public_from_local": None,
            "min_local_score": None,
            "max_local_score": None,
            "min_public_score": None,
            "max_public_score": None,
        }
    else:
        fit = _linear_fit(locals_arr, public_arr)
        stats = {
            "n_pairs": int(deltas.size),
            "mean_delta": float(np.mean(deltas)),
            "median_delta": float(np.median(deltas)),
            "min_delta": float(np.min(deltas)),
            "max_delta": float(np.max(deltas)),
            "p10_delta": float(np.percentile(deltas, 10)),
            "p90_delta": float(np.percentile(deltas, 90)),
            "pearson_local_public": _pearson_corr(locals_arr, public_arr),
            "spearman_local_public": _spearman_corr(locals_arr, public_arr),
            "linear_fit_public_from_local": fit,
            "min_local_score": float(np.min(locals_arr)),
            "max_local_score": float(np.max(locals_arr)),
            "min_public_score": float(np.min(public_arr)),
            "max_public_score": float(np.max(public_arr)),
        }
    return {
        "created_utc": _utc_now(),
        "competition": str(competition),
        "page_size": int(page_size),
        "local_overrides_path": None if local_overrides_path is None else str(local_overrides_path),
        "local_overrides_count": int(len(local_overrides_by_ref)),
        "exclude_refs_count": int(len(exclude_refs)),
        "only_override_refs": bool(only_override_refs),
        "excluded_by_ref": int(excluded_by_ref),
        "excluded_by_scope": int(excluded_by_scope),
        "excluded_by_status": int(excluded_by_status),
        "overridden_by_ref": int(overridden_by_ref),
        "submissions_scanned": int(total),
        "pairs": pairs,
        "stats": stats,
    }


def estimate_public_from_local(*, local_score: float, calibration: dict) -> dict:
    location = "src/rna3d_local/kaggle_calibration.py:estimate_public_from_local"
    try:
        x = float(local_score)
    except (TypeError, ValueError):
        raise_error("CALIBRATION", location, "local_score invalido", impact="1", examples=[str(local_score)])
    stats = calibration.get("stats")
    if not isinstance(stats, dict):
        raise_error("CALIBRATION", location, "calibration sem bloco stats", impact="1", examples=[])
    n_pairs = int(stats.get("n_pairs") or 0)
    if n_pairs <= 0:
        raise_error("CALIBRATION", location, "calibration sem pares validos", impact="0", examples=[])
    median_delta = stats.get("median_delta")
    min_delta = stats.get("min_delta")
    p10_delta = stats.get("p10_delta")
    if median_delta is None or min_delta is None or p10_delta is None:
        raise_error("CALIBRATION", location, "stats incompletos para estimativa", impact="1", examples=[])
    fit = stats.get("linear_fit_public_from_local")
    linear_fit_score = None
    if isinstance(fit, dict) and fit.get("slope") is not None and fit.get("intercept") is not None:
        linear_fit_score = float(float(fit["slope"]) * float(x) + float(fit["intercept"]))
    return {
        "local_score": float(x),
        "expected_public_median": float(x + float(median_delta)),
        "expected_public_p10": float(x + float(p10_delta)),
        "expected_public_worst_seen": float(x + float(min_delta)),
        "expected_public_linear_fit": linear_fit_score,
        "n_pairs": int(n_pairs),
    }


def build_alignment_decision(
    *,
    local_score: float,
    baseline_public_score: float,
    calibration: dict,
    method: str = "p10",
    min_public_improvement: float = 0.0,
    min_pairs: int = 3,
    allow_extrapolation: bool = False,
) -> dict:
    location = "src/rna3d_local/kaggle_calibration.py:build_alignment_decision"
    try:
        base_public = float(baseline_public_score)
    except (TypeError, ValueError):
        raise_error("CALIBRATION", location, "baseline_public_score invalido", impact="1", examples=[str(baseline_public_score)])
    try:
        min_imp = float(min_public_improvement)
    except (TypeError, ValueError):
        raise_error("CALIBRATION", location, "min_public_improvement invalido", impact="1", examples=[str(min_public_improvement)])
    try:
        min_pairs_i = int(min_pairs)
    except (TypeError, ValueError):
        raise_error("CALIBRATION", location, "min_pairs invalido", impact="1", examples=[str(min_pairs)])
    if min_pairs_i <= 0:
        raise_error("CALIBRATION", location, "min_pairs deve ser > 0", impact="1", examples=[str(min_pairs_i)])

    method_norm = str(method).strip().lower()
    if method_norm not in _ALLOWED_CALIBRATION_METHODS:
        raise_error(
            "CALIBRATION",
            location,
            "metodo de calibracao invalido",
            impact="1",
            examples=[method_norm, ",".join(_ALLOWED_CALIBRATION_METHODS)],
        )

    stats = calibration.get("stats")
    if not isinstance(stats, dict):
        raise_error("CALIBRATION", location, "calibration sem bloco stats", impact="1", examples=[])
    n_pairs = int(stats.get("n_pairs") or 0)
    if n_pairs < min_pairs_i:
        raise_error(
            "CALIBRATION",
            location,
            "pares insuficientes para gate calibrado",
            impact=f"n_pairs={n_pairs} min_pairs={min_pairs_i}",
            examples=[],
        )

    try:
        local_val = float(local_score)
    except (TypeError, ValueError):
        raise_error("CALIBRATION", location, "local_score invalido", impact="1", examples=[str(local_score)])

    estimate = estimate_public_from_local(local_score=local_score, calibration=calibration)
    key_map = {
        "median": "expected_public_median",
        "p10": "expected_public_p10",
        "worst_seen": "expected_public_worst_seen",
        "linear_fit": "expected_public_linear_fit",
    }
    chosen_key = key_map[method_norm]
    expected_public_raw = estimate.get(chosen_key)
    if expected_public_raw is None:
        raise_error(
            "CALIBRATION",
            location,
            "estimativa indisponivel para o metodo selecionado",
            impact="1",
            examples=[method_norm],
        )
    expected_public = float(expected_public_raw)
    threshold = float(base_public + min_imp)
    min_local_seen_raw = stats.get("min_local_score")
    max_local_seen_raw = stats.get("max_local_score")
    min_local_seen = None if min_local_seen_raw is None else float(min_local_seen_raw)
    max_local_seen = None if max_local_seen_raw is None else float(max_local_seen_raw)
    is_extrapolation = False
    extrapolation_direction = "none"
    if min_local_seen is not None and max_local_seen is not None:
        if local_val < min_local_seen:
            is_extrapolation = True
            extrapolation_direction = "below_min"
        elif local_val > max_local_seen:
            is_extrapolation = True
            extrapolation_direction = "above_max"
    allowed = bool(expected_public > threshold)
    reasons: list[str] = []
    if not allowed:
        reasons.append("expected_public_score abaixo do limiar")
    if is_extrapolation and not bool(allow_extrapolation):
        allowed = False
        reasons.append("local_score em extrapolacao fora do range historico")
    return {
        "method": method_norm,
        "estimate_key": chosen_key,
        "expected_public_score": float(expected_public),
        "baseline_public_score": float(base_public),
        "min_public_improvement": float(min_imp),
        "required_threshold": float(threshold),
        "allowed": bool(allowed),
        "n_pairs": int(n_pairs),
        "local_score": float(local_val),
        "local_score_min_seen": min_local_seen,
        "local_score_max_seen": max_local_seen,
        "is_extrapolation": bool(is_extrapolation),
        "extrapolation_direction": str(extrapolation_direction),
        "allow_extrapolation": bool(allow_extrapolation),
        "reasons": reasons,
        "estimate": estimate,
    }


def write_calibration_report(*, report: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
