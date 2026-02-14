from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .bigdata import TableReadConfig, collect_streaming, scan_table
from .errors import raise_error
from .robust_score import read_score_artifact, read_score_json

def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(20):
        if (cur / "pyproject.toml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise_error("CLI", "src/rna3d_local/cli.py:_find_repo_root", "repo_root nao encontrado (pyproject.toml ausente)", impact="1", examples=[str(start)])
    raise AssertionError("unreachable")

def _rel_or_abs(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)

def _parse_float_list_arg(*, raw: str, arg_name: str, location: str) -> tuple[float, ...]:
    items = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not items:
        raise_error("CLI", location, f"{arg_name} vazio", impact="1", examples=[str(raw)])
    out: list[float] = []
    for tok in items:
        try:
            out.append(float(tok))
        except ValueError:
            raise_error("CLI", location, f"{arg_name} contem valor invalido", impact="1", examples=[tok])
    return tuple(out)

def _parse_named_score_entries(*, raw_entries: list[str], repo: Path, location: str) -> dict[str, dict]:
    if not raw_entries:
        raise_error("ROBUST", location, "nenhum --score informado", impact="0", examples=[])
    named: dict[str, dict] = {}
    for raw in raw_entries:
        tok = str(raw).strip()
        if "=" not in tok:
            raise_error(
                "ROBUST",
                location,
                "formato invalido em --score (use nome=caminho_score_json)",
                impact="1",
                examples=[tok],
            )
        name, path_raw = tok.split("=", 1)
        score_name = str(name).strip()
        if not score_name:
            raise_error("ROBUST", location, "nome vazio em --score", impact="1", examples=[tok])
        score_path = Path(str(path_raw).strip())
        if not score_path.is_absolute():
            score_path = (repo / score_path).resolve()
        if score_name in named:
            raise_error("ROBUST", location, "nome duplicado em --score", impact="1", examples=[score_name])
        art = read_score_artifact(
            score_json_path=score_path,
            stage="ROBUST",
            location=location,
            require_metadata=True,
        )
        named[score_name] = art
    return named


def _resolve_competitive_regime_id(*, named_artifacts: dict[str, dict], public_score_name: str, stage: str, location: str) -> str:
    if not named_artifacts:
        raise_error(stage, location, "nenhum score informado para resolver regime competitivo", impact="0", examples=[])
    exact = named_artifacts.get(str(public_score_name))
    if isinstance(exact, dict):
        regime = str(exact.get("meta", {}).get("regime_id") or "").strip()
        if regime:
            return regime
    counts: dict[str, int] = {}
    for art in named_artifacts.values():
        regime = str(art.get("meta", {}).get("regime_id") or "").strip()
        if not regime:
            raise_error(stage, location, "score sem regime_id no meta", impact="1", examples=[str(art.get("path"))])
        counts[regime] = int(counts.get(regime, 0)) + 1
    best = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return str(best[0][0])


def _prepare_competitive_scores(
    *,
    named_artifacts: dict[str, dict],
    public_score_name: str,
    stage: str,
    location: str,
    forced_regime_id: str | None = None,
) -> tuple[dict[str, float], dict]:
    if not named_artifacts:
        raise_error(stage, location, "nenhum score informado", impact="0", examples=[])
    regime_id = str(forced_regime_id).strip() if forced_regime_id is not None else _resolve_competitive_regime_id(
        named_artifacts=named_artifacts,
        public_score_name=public_score_name,
        stage=stage,
        location=location,
    )
    if not regime_id:
        raise_error(stage, location, "regime competitivo invalido", impact="1", examples=[str(forced_regime_id)])

    included: dict[str, dict] = {}
    excluded: dict[str, str] = {}
    for name, art in sorted(named_artifacts.items(), key=lambda kv: kv[0]):
        art_regime = str(art.get("meta", {}).get("regime_id") or "").strip()
        if art_regime == regime_id:
            included[name] = art
        else:
            excluded[name] = art_regime
    if not included:
        raise_error(
            stage,
            location,
            "nenhum score no regime competitivo selecionado",
            impact=f"regime={regime_id} total={len(named_artifacts)}",
            examples=[f"{k}:{v}" for k, v in list(excluded.items())[:8]],
        )

    fingerprint_fields = ("sample_schema_sha", "n_models", "metric_sha256", "usalign_sha256")
    first_name = sorted(included.keys())[0]
    first_meta = included[first_name].get("meta", {})
    baseline_fp = {k: first_meta.get(k) for k in fingerprint_fields}
    mismatches: list[str] = []
    for name, art in included.items():
        meta = art.get("meta", {})
        for key in fingerprint_fields:
            if str(meta.get(key)) != str(baseline_fp.get(key)):
                mismatches.append(f"{name}:{key}")
    if mismatches:
        raise_error(
            stage,
            location,
            "scores do regime competitivo com fingerprint divergente",
            impact=str(len(mismatches)),
            examples=mismatches[:8],
        )

    named_scores: dict[str, float] = {name: float(art["score"]) for name, art in included.items()}
    compatibility = {
        "allowed": True,
        "compare_fields": list(fingerprint_fields),
        "fingerprint": {k: baseline_fp.get(k) for k in fingerprint_fields},
    }
    regime_summary = {
        "competitive_regime_id": regime_id,
        "public_score_name": str(public_score_name),
        "included_names": sorted(included.keys()),
        "excluded_by_regime": {k: str(v) for k, v in sorted(excluded.items(), key=lambda kv: kv[0])},
        "included_count": int(len(included)),
        "excluded_count": int(len(excluded)),
    }
    return named_scores, {"compatibility_checks": compatibility, "regime_summary": regime_summary}


def _assert_regime_bundle_matches(*, left: dict, right: dict, stage: str, location: str) -> None:
    left_comp = left.get("compatibility_checks")
    right_comp = right.get("compatibility_checks")
    if not isinstance(left_comp, dict) or not isinstance(right_comp, dict):
        raise_error(stage, location, "bundle de comparabilidade invalido", impact="1", examples=[])
    left_fp = left_comp.get("fingerprint")
    right_fp = right_comp.get("fingerprint")
    if not isinstance(left_fp, dict) or not isinstance(right_fp, dict):
        raise_error(stage, location, "fingerprint ausente no bundle de comparabilidade", impact="1", examples=[])
    mismatches: list[str] = []
    for key in ("sample_schema_sha", "n_models", "metric_sha256", "usalign_sha256"):
        if str(left_fp.get(key)) != str(right_fp.get(key)):
            mismatches.append(key)
    if mismatches:
        raise_error(
            stage,
            location,
            "candidate/baseline com fingerprint de score divergente",
            impact=str(len(mismatches)),
            examples=mismatches[:8],
        )
    left_regime = str(left.get("regime_summary", {}).get("competitive_regime_id") or "")
    right_regime = str(right.get("regime_summary", {}).get("competitive_regime_id") or "")
    if left_regime != right_regime:
        raise_error(
            stage,
            location,
            "candidate/baseline em regimes competitivos diferentes",
            impact="1",
            examples=[f"candidate={left_regime}", f"baseline={right_regime}"],
        )

def _read_robust_report(*, report_path: Path, location: str) -> dict:
    if not report_path.exists():
        raise_error("GATE", location, "robust_report nao encontrado", impact="1", examples=[str(report_path)])
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise_error("GATE", location, "falha ao ler robust_report", impact="1", examples=[f"{type(e).__name__}:{e}"])
    if not isinstance(payload, dict):
        raise_error("GATE", location, "robust_report invalido (esperado objeto JSON)", impact="1", examples=[str(report_path)])
    allowed = payload.get("allowed")
    if not isinstance(allowed, bool):
        raise_error("GATE", location, "robust_report sem campo booleano allowed", impact="1", examples=[str(report_path)])
    return payload

def _read_readiness_report(*, report_path: Path, location: str) -> dict:
    if not report_path.exists():
        raise_error("GATE", location, "readiness_report nao encontrado", impact="1", examples=[str(report_path)])
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise_error("GATE", location, "falha ao ler readiness_report", impact="1", examples=[f"{type(e).__name__}:{e}"])
    if not isinstance(payload, dict):
        raise_error("GATE", location, "readiness_report invalido (esperado objeto JSON)", impact="1", examples=[str(report_path)])
    allowed = payload.get("allowed")
    if not isinstance(allowed, bool):
        raise_error("GATE", location, "readiness_report sem campo booleano allowed", impact="1", examples=[str(report_path)])
    return payload

def _enforce_non_ensemble_predictions(
    *,
    predictions_long_path: Path,
    stage: str,
    location: str,
) -> dict:
    if not predictions_long_path.exists():
        raise_error(stage, location, "predictions_long nao encontrado", impact="1", examples=[str(predictions_long_path)])
    lf = scan_table(
        config=TableReadConfig(
            path=predictions_long_path,
            stage=stage,
            location=location,
            columns=("branch", "target_id"),
        )
    ).select(
        pl.col("branch").cast(pl.Utf8).str.to_lowercase().alias("branch"),
        pl.col("target_id").cast(pl.Utf8).alias("target_id"),
    )
    total_rows = int(
        collect_streaming(
            lf=lf.select(pl.len().alias("n")),
            stage=stage,
            location=location,
        ).get_column("n")[0]
    )
    if total_rows <= 0:
        raise_error(stage, location, "predictions_long vazio", impact="0", examples=[str(predictions_long_path)])
    ensemble_targets_lf = lf.filter(pl.col("branch") == "ensemble").select("target_id").unique()
    ensemble_target_count = int(
        collect_streaming(
            lf=ensemble_targets_lf.select(pl.len().alias("n")),
            stage=stage,
            location=location,
        ).get_column("n")[0]
    )
    if ensemble_target_count > 0:
        examples = (
            collect_streaming(
                lf=ensemble_targets_lf.select("target_id").head(8),
                stage=stage,
                location=location,
            )
            .get_column("target_id")
            .to_list()
        )
        raise_error(
            stage,
            location,
            "submissao originada de blend de coordenadas bloqueada para submit competitivo",
            impact=str(ensemble_target_count),
            examples=[str(x) for x in examples],
        )
    return {
        "predictions_long": str(predictions_long_path),
        "rows": int(total_rows),
        "policy": "non_ensemble_only",
        "allowed": True,
    }

def _looks_like_target_patch(*, text: str) -> bool:
    t = str(text or "").lower()
    if not t:
        return False
    tokens = (
        "target_patch",
        "patch_por_alvo",
        "oracle_local",
        "per_target_patch",
    )
    return any(tok in t for tok in tokens)

def _enforce_submit_hardening(
    *,
    location: str,
    require_min_cv_count: int,
    robust_report_path: Path | None,
    robust_payload: dict | None,
    readiness_report_path: Path | None = None,
    readiness_payload: dict | None = None,
    submission_path: Path,
    message: str,
) -> None:
    if robust_payload is None:
        raise_error(
            "GATE",
            location,
            "submissao bloqueada: robust_report obrigatorio para submit competitivo",
            impact="1",
            examples=["--robust-report runs/<...>/robust_eval.json"],
        )
    if readiness_payload is None:
        raise_error(
            "GATE",
            location,
            "submissao bloqueada: readiness_report obrigatorio para submit competitivo",
            impact="1",
            examples=["--readiness-report runs/<...>/submit_readiness.json"],
        )
    readiness_allowed = bool(readiness_payload.get("allowed", False))
    if not readiness_allowed:
        raise_error(
            "GATE",
            location,
            "submissao bloqueada por readiness_report",
            impact="1",
            examples=[str(readiness_report_path) if readiness_report_path is not None else "-"],
        )

    robust_allowed = bool(robust_payload.get("allowed", False))
    if not robust_allowed:
        raise_error(
            "GATE",
            location,
            "submissao bloqueada por robust_report",
            impact="1",
            examples=[str(robust_report_path) if robust_report_path is not None else "-"],
        )

    robust_comp = robust_payload.get("compatibility_checks")
    if not isinstance(robust_comp, dict):
        raise_error(
            "GATE",
            location,
            "submissao bloqueada: robust_report sem compatibility_checks",
            impact="1",
            examples=[str(robust_report_path) if robust_report_path is not None else "-"],
        )
    if not bool(robust_comp.get("allowed", False)):
        raise_error(
            "GATE",
            location,
            "submissao bloqueada: robust_report com comparabilidade invalida",
            impact="1",
            examples=[str(robust_report_path) if robust_report_path is not None else "-"],
        )
    robust_regime_summary = robust_payload.get("regime_summary")
    robust_regime = str((robust_regime_summary or {}).get("competitive_regime_id") or "")
    if robust_regime != "kaggle_official_5model":
        raise_error(
            "GATE",
            location,
            "submissao bloqueada: robust_report fora do regime competitivo oficial",
            impact="1",
            examples=[f"regime={robust_regime}"],
        )

    readiness_comp = readiness_payload.get("compatibility_checks")
    if not isinstance(readiness_comp, dict):
        raise_error(
            "GATE",
            location,
            "submissao bloqueada: readiness_report sem compatibility_checks",
            impact="1",
            examples=[str(readiness_report_path) if readiness_report_path is not None else "-"],
        )
    if not bool(readiness_comp.get("allowed", False)):
        raise_error(
            "GATE",
            location,
            "submissao bloqueada: readiness_report com comparabilidade invalida",
            impact="1",
            examples=[str(readiness_report_path) if readiness_report_path is not None else "-"],
        )
    readiness_regime_summary = readiness_payload.get("regime_summary")
    readiness_regime = str((readiness_regime_summary or {}).get("competitive_regime_id") or "")
    if readiness_regime != "kaggle_official_5model":
        raise_error(
            "GATE",
            location,
            "submissao bloqueada: readiness_report fora do regime competitivo oficial",
            impact="1",
            examples=[f"regime={readiness_regime}"],
        )

    summary = robust_payload.get("summary")
    if not isinstance(summary, dict):
        raise_error("GATE", location, "robust_report sem bloco summary", impact="1", examples=[str(robust_report_path)])
    try:
        min_cv = int(require_min_cv_count)
    except (TypeError, ValueError):
        raise_error("GATE", location, "require_min_cv_count invalido", impact="1", examples=[str(require_min_cv_count)])
    if min_cv < 0:
        raise_error("GATE", location, "require_min_cv_count deve ser >= 0", impact="1", examples=[str(min_cv)])
    cv_count = int(summary.get("cv_count") or 0)
    if cv_count < min_cv:
        raise_error(
            "GATE",
            location,
            "submissao bloqueada: cv_count insuficiente",
            impact=f"{cv_count}",
            examples=[f"min_cv_count={min_cv}"],
        )

    risk_flags = [str(x) for x in (summary.get("risk_flags") or [])]
    public_score_name = str(summary.get("public_score_name") or "")
    if cv_count <= 0 and public_score_name == "public_validation":
        raise_error(
            "GATE",
            location,
            "submissao bloqueada: candidato dependente de public_validation sem CV",
            impact="1",
            examples=["public_score_name=public_validation", f"cv_count={cv_count}"],
        )
    if "public_validation_without_cv" in risk_flags:
        raise_error(
            "GATE",
            location,
            "submissao bloqueada: risk_flag public_validation_without_cv",
            impact="1",
            examples=risk_flags[:8],
        )

    hints: list[str] = []
    if _looks_like_target_patch(text=submission_path.name):
        hints.append(f"submission={submission_path.name}")
    if _looks_like_target_patch(text=message):
        hints.append("message_hint=target_patch")
    if robust_report_path is not None and _looks_like_target_patch(text=str(robust_report_path)):
        hints.append("robust_report_hint=target_patch")
    if hints:
        raise_error(
            "GATE",
            location,
            "submissao bloqueada: padrao target_patch proibido por gate",
            impact=str(len(hints)),
            examples=hints[:8],
        )

    alignment = robust_payload.get("alignment_decision")
    if isinstance(alignment, dict):
        is_extrapolation = bool(alignment.get("is_extrapolation", False))
        if is_extrapolation:
            raise_error(
                "GATE",
                location,
                "submissao bloqueada: calibracao em extrapolacao fora do range historico",
                impact="1",
                examples=[
                    f"local_score={alignment.get('local_score')}",
                    f"range=[{alignment.get('local_score_min_seen')},{alignment.get('local_score_max_seen')}]",
                ],
            )

def _read_score_json(*, score_json_path: Path, location: str) -> float:
    return read_score_json(score_json_path=score_json_path, stage="CLI", location=location, require_metadata=True)

def _target_ids_for_fold(*, targets_path: Path, fold_id: int, stage: str, location: str) -> list[str]:
    lf = scan_table(
        config=TableReadConfig(
            path=targets_path,
            stage=stage,
            location=location,
            columns=("target_id", "fold_id"),
        )
    )
    out = collect_streaming(
        lf=lf.filter(pl.col("fold_id") == int(fold_id)).select(pl.col("target_id").cast(pl.Utf8)),
        stage=stage,
        location=location,
    )
    target_ids = out.get_column("target_id").to_list()
    if not target_ids:
        raise_error(stage, location, "fold sem targets", impact="0", examples=[str(fold_id)])
    return target_ids
