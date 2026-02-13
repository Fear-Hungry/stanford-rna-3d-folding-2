from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .errors import raise_error


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_metric(*, payload: dict, key: str, location: str) -> float:
    if key not in payload:
        raise_error("TRAIN_GATE", location, "metrica ausente no modelo", impact="1", examples=[key])
    try:
        val = float(payload[key])
    except (TypeError, ValueError):
        raise_error("TRAIN_GATE", location, "metrica invalida no modelo", impact="1", examples=[f"{key}={payload.get(key)}"])
    if not math.isfinite(val):
        raise_error("TRAIN_GATE", location, "metrica nao-finita no modelo", impact="1", examples=[f"{key}={val}"])
    return float(val)


def _read_n_samples(*, payload: dict, location: str) -> int:
    if "n_samples" not in payload:
        raise_error("TRAIN_GATE", location, "metrica n_samples ausente no modelo", impact="1", examples=["n_samples"])
    try:
        n = int(payload["n_samples"])
    except (TypeError, ValueError):
        raise_error("TRAIN_GATE", location, "n_samples invalido no modelo", impact="1", examples=[str(payload.get("n_samples"))])
    if n <= 0:
        raise_error("TRAIN_GATE", location, "n_samples deve ser > 0", impact="1", examples=[str(n)])
    return int(n)


def _require_metrics_dict(*, payload: dict, key: str, location: str) -> dict:
    block = payload.get(key)
    if not isinstance(block, dict):
        raise_error("TRAIN_GATE", location, "modelo sem bloco de metricas obrigatorio", impact="1", examples=[key])
    return block


def evaluate_training_overfit_gate(
    *,
    train_metrics: dict,
    val_metrics: dict,
    min_val_samples: int = 32,
    max_mae_gap_ratio: float = 0.40,
    max_rmse_gap_ratio: float = 0.40,
    max_r2_drop: float = 0.30,
    max_spearman_drop: float = 0.30,
    max_pearson_drop: float = 0.30,
) -> dict:
    location = "src/rna3d_local/training_gate.py:evaluate_training_overfit_gate"
    try:
        min_val_samples_i = int(min_val_samples)
    except (TypeError, ValueError):
        raise_error("TRAIN_GATE", location, "min_val_samples invalido", impact="1", examples=[str(min_val_samples)])
    if min_val_samples_i < 0:
        raise_error("TRAIN_GATE", location, "min_val_samples deve ser >= 0", impact="1", examples=[str(min_val_samples_i)])

    train_mae = _read_metric(payload=train_metrics, key="mae", location=location)
    train_rmse = _read_metric(payload=train_metrics, key="rmse", location=location)
    train_r2 = _read_metric(payload=train_metrics, key="r2", location=location)
    train_sp = _read_metric(payload=train_metrics, key="spearman", location=location)
    train_pear = _read_metric(payload=train_metrics, key="pearson", location=location)
    train_n = _read_n_samples(payload=train_metrics, location=location)

    val_mae = _read_metric(payload=val_metrics, key="mae", location=location)
    val_rmse = _read_metric(payload=val_metrics, key="rmse", location=location)
    val_r2 = _read_metric(payload=val_metrics, key="r2", location=location)
    val_sp = _read_metric(payload=val_metrics, key="spearman", location=location)
    val_pear = _read_metric(payload=val_metrics, key="pearson", location=location)
    val_n = _read_n_samples(payload=val_metrics, location=location)

    eps = 1e-12
    mae_gap_ratio = float((val_mae - train_mae) / max(abs(train_mae), eps))
    rmse_gap_ratio = float((val_rmse - train_rmse) / max(abs(train_rmse), eps))
    r2_drop = float(train_r2 - val_r2)
    sp_drop = float(train_sp - val_sp)
    pear_drop = float(train_pear - val_pear)

    reasons: list[str] = []
    if val_n < min_val_samples_i:
        reasons.append(f"n_samples de validacao insuficiente ({val_n} < {min_val_samples_i})")
    if mae_gap_ratio > float(max_mae_gap_ratio):
        reasons.append(f"mae_gap_ratio acima do limite ({mae_gap_ratio:.6f} > {float(max_mae_gap_ratio):.6f})")
    if rmse_gap_ratio > float(max_rmse_gap_ratio):
        reasons.append(f"rmse_gap_ratio acima do limite ({rmse_gap_ratio:.6f} > {float(max_rmse_gap_ratio):.6f})")
    if r2_drop > float(max_r2_drop):
        reasons.append(f"r2_drop acima do limite ({r2_drop:.6f} > {float(max_r2_drop):.6f})")
    if sp_drop > float(max_spearman_drop):
        reasons.append(f"spearman_drop acima do limite ({sp_drop:.6f} > {float(max_spearman_drop):.6f})")
    if pear_drop > float(max_pearson_drop):
        reasons.append(f"pearson_drop acima do limite ({pear_drop:.6f} > {float(max_pearson_drop):.6f})")

    return {
        "created_utc": _utc_now(),
        "train_metrics": {
            "mae": float(train_mae),
            "rmse": float(train_rmse),
            "r2": float(train_r2),
            "spearman": float(train_sp),
            "pearson": float(train_pear),
            "n_samples": int(train_n),
        },
        "val_metrics": {
            "mae": float(val_mae),
            "rmse": float(val_rmse),
            "r2": float(val_r2),
            "spearman": float(val_sp),
            "pearson": float(val_pear),
            "n_samples": int(val_n),
        },
        "thresholds": {
            "min_val_samples": int(min_val_samples_i),
            "max_mae_gap_ratio": float(max_mae_gap_ratio),
            "max_rmse_gap_ratio": float(max_rmse_gap_ratio),
            "max_r2_drop": float(max_r2_drop),
            "max_spearman_drop": float(max_spearman_drop),
            "max_pearson_drop": float(max_pearson_drop),
        },
        "diagnostics": {
            "mae_gap_ratio": float(mae_gap_ratio),
            "rmse_gap_ratio": float(rmse_gap_ratio),
            "r2_drop": float(r2_drop),
            "spearman_drop": float(sp_drop),
            "pearson_drop": float(pear_drop),
        },
        "allowed": len(reasons) == 0,
        "reasons": reasons,
    }


def evaluate_training_gate_from_model_json(
    *,
    model_json_path: Path,
    min_val_samples: int = 32,
    max_mae_gap_ratio: float = 0.40,
    max_rmse_gap_ratio: float = 0.40,
    max_r2_drop: float = 0.30,
    max_spearman_drop: float = 0.30,
    max_pearson_drop: float = 0.30,
) -> dict:
    location = "src/rna3d_local/training_gate.py:evaluate_training_gate_from_model_json"
    if not model_json_path.exists():
        raise_error("TRAIN_GATE", location, "arquivo de modelo nao encontrado", impact="1", examples=[str(model_json_path)])
    try:
        payload = json.loads(model_json_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise_error("TRAIN_GATE", location, "falha ao ler modelo json", impact="1", examples=[f"{type(e).__name__}:{e}"])
    if not isinstance(payload, dict):
        raise_error("TRAIN_GATE", location, "modelo json invalido (esperado objeto)", impact="1", examples=[str(model_json_path)])

    train_metrics = _require_metrics_dict(payload=payload, key="train_metrics", location=location)
    val_metrics = _require_metrics_dict(payload=payload, key="val_metrics", location=location)
    report = evaluate_training_overfit_gate(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        min_val_samples=int(min_val_samples),
        max_mae_gap_ratio=float(max_mae_gap_ratio),
        max_rmse_gap_ratio=float(max_rmse_gap_ratio),
        max_r2_drop=float(max_r2_drop),
        max_spearman_drop=float(max_spearman_drop),
        max_pearson_drop=float(max_pearson_drop),
    )
    report["model_json"] = str(model_json_path)
    report["model_type"] = str(payload.get("model_type") or "unknown")
    return report


def write_training_gate_report(*, report: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
