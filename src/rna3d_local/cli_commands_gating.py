from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cli_commands_common import (
    _enforce_non_ensemble_predictions,
    _enforce_submit_hardening,
    _find_repo_root,
    _parse_named_score_entries,
    _read_readiness_report,
    _read_robust_report,
    _read_score_json,
    _rel_or_abs,
    _utc_now_compact,
)
from .errors import raise_error
from .gating import assert_submission_allowed
from .kaggle_calibration import (
    build_alignment_decision,
    build_kaggle_local_calibration,
    estimate_public_from_local,
    write_calibration_report,
)
from .kaggle_cli import run_kaggle
from .kaggle_submissions import list_kaggle_submissions
from .robust_score import evaluate_robust_gate, write_robust_report
from .submission_readiness import evaluate_submit_readiness, write_submit_readiness_report
from .training_gate import evaluate_training_gate_from_model_json, write_training_gate_report

def _cmd_submit_kaggle(args: argparse.Namespace) -> int:
    location = "src/rna3d_local/cli.py:_cmd_submit_kaggle"
    repo = _find_repo_root(Path.cwd())
    report = (repo / args.gating_report).resolve()
    if args.gating_report == "runs/auto":
        report = repo / "runs" / f"{_utc_now_compact()}_gating_report.json"
    sample_path = (repo / args.sample).resolve()
    submission_path = (repo / args.submission).resolve()
    predictions_long_path = None if args.predictions_long is None else (repo / args.predictions_long).resolve()
    score_json_path = None if args.score_json is None else (repo / args.score_json).resolve()
    calibration_overrides_path = None if args.calibration_overrides is None else (repo / args.calibration_overrides).resolve()
    robust_report = None if args.robust_report is None else (repo / args.robust_report).resolve()
    robust_payload = None if robust_report is None else _read_robust_report(report_path=robust_report, location=location)
    readiness_report = None if args.readiness_report is None else (repo / args.readiness_report).resolve()
    readiness_payload = None if readiness_report is None else _read_readiness_report(report_path=readiness_report, location=location)
    strategy_gate = None
    if predictions_long_path is not None:
        strategy_gate = _enforce_non_ensemble_predictions(
            predictions_long_path=predictions_long_path,
            stage="GATE",
            location=location,
        )

    assert_submission_allowed(
        sample_path=sample_path,
        submission_path=submission_path,
        report_path=report,
        is_smoke=bool(args.is_smoke),
        is_partial=bool(args.is_partial),
        score_json_path=score_json_path,
        baseline_score=None if args.baseline_score is None else float(args.baseline_score),
        min_improvement=float(args.min_improvement),
        allow_regression=False,
    )

    _enforce_submit_hardening(
        location=location,
        require_min_cv_count=int(args.require_min_cv_count),
        robust_report_path=robust_report,
        robust_payload=robust_payload,
        readiness_report_path=readiness_report,
        readiness_payload=readiness_payload,
        submission_path=submission_path,
        message=str(args.message),
    )

    calibration_report_out: Path | None = None
    alignment_decision: dict | None = None
    if args.baseline_public_score is not None:
        if score_json_path is None:
            raise_error(
                "GATE",
                location,
                "baseline_public_score requer --score-json para gate calibrado",
                impact="1",
                examples=["--score-json runs/<...>/score.json"],
            )
        calibration_report_out = (repo / args.calibration_report).resolve()
        if args.calibration_report == "runs/auto":
            calibration_report_out = repo / "runs" / f"{_utc_now_compact()}_kaggle_calibration_gate.json"

        calibration = build_kaggle_local_calibration(
            competition=str(args.competition),
            page_size=int(args.calibration_page_size),
            local_overrides_path=calibration_overrides_path,
        )
        current_local_score = _read_score_json(score_json_path=score_json_path, location=location)
        alignment_decision = build_alignment_decision(
            local_score=float(current_local_score),
            baseline_public_score=float(args.baseline_public_score),
            calibration=calibration,
            method=str(args.calibration_method),
            min_public_improvement=float(args.min_public_improvement),
            min_pairs=int(args.calibration_min_pairs),
            allow_extrapolation=False,
        )
        calibration["alignment_decision"] = alignment_decision
        write_calibration_report(report=calibration, out_path=calibration_report_out)
        if not bool(alignment_decision.get("allowed", False)):
            raise_error(
                "GATE",
                location,
                "submissao bloqueada por gate calibrado local->public",
                impact="1",
                examples=[
                    f"expected={alignment_decision.get('expected_public_score')}",
                    f"threshold={alignment_decision.get('required_threshold')}",
                    f"method={alignment_decision.get('method')}",
                ],
            )

    notebook_ref = str(args.notebook_ref or "").strip()
    if not notebook_ref:
        raise_error(
            "SUBMIT",
            location,
            "competicao notebook-only: --notebook-ref obrigatorio",
            impact="1",
            examples=["owner/notebook-slug"],
        )
    if args.notebook_version is None:
        raise_error(
            "SUBMIT",
            location,
            "competicao notebook-only: --notebook-version obrigatorio",
            impact="1",
            examples=["--notebook-version 53"],
        )
    try:
        notebook_version = int(args.notebook_version)
    except (TypeError, ValueError):
        raise_error("SUBMIT", location, "notebook_version invalido", impact="1", examples=[str(args.notebook_version)])
    if notebook_version <= 0:
        raise_error("SUBMIT", location, "notebook_version deve ser > 0", impact="1", examples=[str(notebook_version)])
    notebook_file = str(args.notebook_file or submission_path.name).strip()
    if not notebook_file:
        raise_error("SUBMIT", location, "notebook_file invalido", impact="1", examples=[str(args.notebook_file)])

    run_kaggle(
        [
            "competitions",
            "submit",
            "-c",
            args.competition,
            "-k",
            notebook_ref,
            "-f",
            notebook_file,
            "-v",
            str(notebook_version),
            "-m",
            args.message,
        ],
        cwd=None,
        location=location,
    )
    payload = {
        "submitted_local_file_checked": _rel_or_abs(submission_path, repo),
        "gating_report": _rel_or_abs(report, repo),
        "notebook_ref": notebook_ref,
        "notebook_version": notebook_version,
        "notebook_file": notebook_file,
    }
    if calibration_report_out is not None:
        payload["calibration_report"] = _rel_or_abs(calibration_report_out, repo)
    if alignment_decision is not None:
        payload["alignment_decision"] = alignment_decision
    if robust_report is not None:
        payload["robust_report"] = _rel_or_abs(robust_report, repo)
    if readiness_report is not None:
        payload["readiness_report"] = _rel_or_abs(readiness_report, repo)
    if predictions_long_path is not None:
        payload["predictions_long"] = _rel_or_abs(predictions_long_path, repo)
    if strategy_gate is not None:
        payload["strategy_gate"] = strategy_gate
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0

def _cmd_calibrate_kaggle_local(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out = (repo / args.out).resolve()
    calibration_overrides_path = None if args.calibration_overrides is None else (repo / args.calibration_overrides).resolve()
    report = build_kaggle_local_calibration(
        competition=str(args.competition),
        page_size=int(args.page_size),
        local_overrides_path=calibration_overrides_path,
    )
    estimate = None
    decision = None
    if args.local_score is not None:
        estimate = estimate_public_from_local(local_score=float(args.local_score), calibration=report)
        report["estimate_for_local_score"] = estimate
    if args.local_score is not None and args.baseline_public_score is not None:
        decision = build_alignment_decision(
            local_score=float(args.local_score),
            baseline_public_score=float(args.baseline_public_score),
            calibration=report,
            method=str(args.method),
            min_public_improvement=float(args.min_public_improvement),
            min_pairs=int(args.min_pairs),
            allow_extrapolation=False,
        )
        report["alignment_decision"] = decision
    write_calibration_report(report=report, out_path=out)
    payload = {"report": _rel_or_abs(out, repo), "pairs": int(report.get("stats", {}).get("n_pairs") or 0)}
    if estimate is not None:
        payload["estimate"] = estimate
    if decision is not None:
        payload["alignment_decision"] = decision
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0

def _cmd_kaggle_submissions(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out = list_kaggle_submissions(
        competition=str(args.competition),
        page_size=int(args.page_size),
        page_token=str(args.page_token or ""),
    )
    if args.out is not None:
        out_path = (repo / args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out = {**out, "out": _rel_or_abs(out_path, repo)}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0

def _cmd_evaluate_robust(args: argparse.Namespace) -> int:
    location = "src/rna3d_local/cli.py:_cmd_evaluate_robust"
    repo = _find_repo_root(Path.cwd())
    predictions_long_path = None if args.predictions_long is None else (repo / args.predictions_long).resolve()
    strategy_gate = None
    if predictions_long_path is not None:
        strategy_gate = _enforce_non_ensemble_predictions(
            predictions_long_path=predictions_long_path,
            stage="ROBUST",
            location=location,
        )
    named_scores = _parse_named_score_entries(raw_entries=list(args.score), repo=repo, location=location)
    report = evaluate_robust_gate(
        named_scores=named_scores,
        public_score_name=str(args.public_score_name),
        baseline_robust_score=None if args.baseline_robust_score is None else float(args.baseline_robust_score),
        min_robust_improvement=float(args.min_robust_improvement),
        competition=None if args.competition is None else str(args.competition),
        baseline_public_score=None if args.baseline_public_score is None else float(args.baseline_public_score),
        calibration_method=str(args.calibration_method),
        calibration_page_size=int(args.calibration_page_size),
        calibration_min_pairs=int(args.calibration_min_pairs),
        calibration_overrides_path=None if args.calibration_overrides is None else (repo / args.calibration_overrides).resolve(),
        min_public_improvement=float(args.min_public_improvement),
        min_cv_count=int(args.min_cv_count),
        block_public_validation_without_cv=True,
        allow_calibration_extrapolation=False,
    )
    if strategy_gate is not None:
        report["strategy_gate"] = strategy_gate
    out_path = (repo / args.out).resolve()
    if args.out == "runs/auto":
        out_path = repo / "runs" / f"{_utc_now_compact()}_robust_eval.json"
    write_robust_report(report=report, out_path=out_path)
    print(
        json.dumps(
            {
                "report": _rel_or_abs(out_path, repo),
                "allowed": bool(report.get("allowed", False)),
                "robust_score": float(report.get("summary", {}).get("robust_score", 0.0)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0

def _cmd_evaluate_submit_readiness(args: argparse.Namespace) -> int:
    location = "src/rna3d_local/cli.py:_cmd_evaluate_submit_readiness"
    repo = _find_repo_root(Path.cwd())
    predictions_long_path = None if args.predictions_long is None else (repo / args.predictions_long).resolve()
    strategy_gate = None
    if predictions_long_path is not None:
        strategy_gate = _enforce_non_ensemble_predictions(
            predictions_long_path=predictions_long_path,
            stage="READINESS",
            location=location,
        )
    candidate_scores = _parse_named_score_entries(
        raw_entries=list(args.candidate_score),
        repo=repo,
        location=location,
    )
    baseline_scores = None
    if args.baseline_score:
        baseline_scores = _parse_named_score_entries(
            raw_entries=list(args.baseline_score),
            repo=repo,
            location=location,
        )
    report = evaluate_submit_readiness(
        candidate_scores=candidate_scores,
        baseline_scores=baseline_scores,
        public_score_name=str(args.public_score_name),
        require_baseline=True,
        require_public_score=True,
        min_cv_count=int(args.min_cv_count),
        min_cv_improvement_count=int(args.min_cv_improvement_count),
        min_fold_improvement=float(args.min_fold_improvement),
        max_cv_regression=float(args.max_cv_regression),
        min_robust_improvement=float(args.min_robust_improvement),
        min_public_local_improvement=float(args.min_public_local_improvement),
        max_cv_std=float(args.max_cv_std),
        max_cv_gap=float(args.max_cv_gap),
        competition=None if args.competition is None else str(args.competition),
        baseline_public_score=None if args.baseline_public_score is None else float(args.baseline_public_score),
        calibration_method=str(args.calibration_method),
        calibration_page_size=int(args.calibration_page_size),
        calibration_min_pairs=int(args.calibration_min_pairs),
        calibration_overrides_path=None if args.calibration_overrides is None else (repo / args.calibration_overrides).resolve(),
        min_public_improvement=float(args.min_public_improvement),
        allow_calibration_extrapolation=False,
        min_calibration_pearson=float(args.min_calibration_pearson),
        min_calibration_spearman=float(args.min_calibration_spearman),
        block_public_validation_without_cv=True,
    )
    if strategy_gate is not None:
        report["strategy_gate"] = strategy_gate
    out_path = (repo / args.out).resolve()
    if args.out == "runs/auto":
        out_path = repo / "runs" / f"{_utc_now_compact()}_submit_readiness.json"
    write_submit_readiness_report(report=report, out_path=out_path)
    print(
        json.dumps(
            {
                "report": _rel_or_abs(out_path, repo),
                "allowed": bool(report.get("allowed", False)),
                "reasons_count": len(report.get("reasons", [])),
                "candidate_robust_score": float(report.get("candidate_summary", {}).get("robust_score", 0.0)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0

def _cmd_evaluate_train_gate(args: argparse.Namespace) -> int:
    location = "src/rna3d_local/cli.py:_cmd_evaluate_train_gate"
    repo = _find_repo_root(Path.cwd())
    model_path = (repo / args.model).resolve()
    out_path = (repo / args.out).resolve()
    if args.out == "runs/auto":
        out_path = repo / "runs" / f"{_utc_now_compact()}_train_gate_eval.json"
    report = evaluate_training_gate_from_model_json(
        model_json_path=model_path,
        min_val_samples=int(args.min_val_samples),
        max_mae_gap_ratio=float(args.max_mae_gap_ratio),
        max_rmse_gap_ratio=float(args.max_rmse_gap_ratio),
        max_r2_drop=float(args.max_r2_drop),
        max_spearman_drop=float(args.max_spearman_drop),
        max_pearson_drop=float(args.max_pearson_drop),
    )
    write_training_gate_report(report=report, out_path=out_path)
    if (not bool(report.get("allowed", False))) and (not bool(args.allow_overfit_model)):
        raise_error(
            "TRAIN_GATE",
            location,
            "modelo bloqueado por gate anti-overfitting",
            impact=str(len(report.get("reasons", []))),
            examples=[str(x) for x in (report.get("reasons") or [])][:8],
        )
    print(
        json.dumps(
            {
                "allowed": bool(report.get("allowed", False)),
                "report": _rel_or_abs(out_path, repo),
                "reasons": report.get("reasons", []),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0

