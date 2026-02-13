from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .bigdata import DEFAULT_MAX_ROWS_IN_MEMORY, DEFAULT_MEMORY_BUDGET_MB, TableReadConfig, collect_streaming, scan_table
from .candidate_pool import build_candidate_pool_from_predictions, parse_prediction_entries
from .contracts import validate_submission_against_sample
from .datasets import (
    build_public_validation_dataset,
    build_train_cv_targets,
    build_train_cv_fold_dataset,
    export_train_solution_for_targets,
    make_sample_submission_for_targets,
    prepare_train_labels_clean,
    prepare_train_labels_parquet,
)
from .drfold2 import predict_drfold2
from .download import download_competition_files
from .ensemble import blend_predictions
from .errors import PipelineError, raise_error
from .export import export_submission_from_long
from .gating import assert_submission_allowed
from .kaggle_calibration import (
    build_alignment_decision,
    build_kaggle_local_calibration,
    estimate_public_from_local,
    write_calibration_report,
)
from .kaggle_cli import run_kaggle
from .qa_gnn_ranker import QA_GNN_DEFAULT_FEATURE_NAMES, score_candidates_with_qa_gnn, train_qa_gnn_ranker
from .qa_ranker import QA_FEATURE_NAMES, train_qa_ranker
from .qa_rnrank import (
    QA_RNRANK_DEFAULT_FEATURE_NAMES,
    score_candidates_with_qa_rnrank,
    select_top5_global_with_qa_rnrank,
    train_qa_rnrank,
)
from .research import generate_report, run_experiment, sync_literature, verify_run
from .retrieval import retrieve_template_candidates
from .robust_score import evaluate_robust_gate, read_score_json, write_robust_report
from .rnapro import RnaProConfig, infer_rnapro, train_rnapro
from .scoring import score_submission, write_score_artifacts
from .submission_readiness import evaluate_submit_readiness, write_submit_readiness_report
from .tbm_predictor import predict_tbm
from .template_pt import convert_templates_to_pt_files
from .template_db import build_template_db
from .training_gate import evaluate_training_gate_from_model_json, write_training_gate_report
from .vendor import DEFAULT_METRIC_KERNEL, vendor_all


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


def _parse_named_score_entries(*, raw_entries: list[str], repo: Path, location: str) -> dict[str, float]:
    if not raw_entries:
        raise_error("ROBUST", location, "nenhum --score informado", impact="0", examples=[])
    named: dict[str, float] = {}
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
        named[score_name] = read_score_json(score_json_path=score_path)
    return named


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
    allow_regression: bool,
    require_robust_report: bool,
    require_min_cv_count: int,
    block_public_validation_without_cv: bool,
    block_target_patch: bool,
    allow_calibration_extrapolation: bool,
    require_readiness_report: bool = False,
    robust_report_path: Path | None,
    robust_payload: dict | None,
    readiness_report_path: Path | None = None,
    readiness_payload: dict | None = None,
    submission_path: Path,
    message: str,
) -> None:
    if bool(require_robust_report) and robust_payload is None and (not bool(allow_regression)):
        raise_error(
            "GATE",
            location,
            "submissao bloqueada: robust_report obrigatorio para submit competitivo",
            impact="1",
            examples=["--robust-report runs/<...>/robust_eval.json"],
        )
    if robust_payload is None:
        if bool(require_readiness_report) and readiness_payload is None and (not bool(allow_regression)):
            raise_error(
                "GATE",
                location,
                "submissao bloqueada: readiness_report obrigatorio para submit competitivo",
                impact="1",
                examples=["--readiness-report runs/<...>/submit_readiness.json"],
            )
        if readiness_payload is None:
            return
        # `robust_report` opcional nesse caminho: readiness aprovado e sem checks extras de robust.
        return
    if bool(require_readiness_report) and readiness_payload is None and (not bool(allow_regression)):
        raise_error(
            "GATE",
            location,
            "submissao bloqueada: readiness_report obrigatorio para submit competitivo",
            impact="1",
            examples=["--readiness-report runs/<...>/submit_readiness.json"],
        )
    if readiness_payload is not None:
        readiness_allowed = bool(readiness_payload.get("allowed", False))
        if (not readiness_allowed) and (not bool(allow_regression)):
            raise_error(
                "GATE",
                location,
                "submissao bloqueada por readiness_report",
                impact="1",
                examples=[str(readiness_report_path) if readiness_report_path is not None else "-"],
            )

    robust_allowed = bool(robust_payload.get("allowed", False))
    if (not robust_allowed) and (not bool(allow_regression)):
        raise_error(
            "GATE",
            location,
            "submissao bloqueada por robust_report",
            impact="1",
            examples=[str(robust_report_path) if robust_report_path is not None else "-"],
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
    if cv_count < min_cv and (not bool(allow_regression)):
        raise_error(
            "GATE",
            location,
            "submissao bloqueada: cv_count insuficiente",
            impact=f"{cv_count}",
            examples=[f"min_cv_count={min_cv}"],
        )

    risk_flags = [str(x) for x in (summary.get("risk_flags") or [])]
    public_score_name = str(summary.get("public_score_name") or "")
    if bool(block_public_validation_without_cv) and cv_count <= 0 and public_score_name == "public_validation" and (not bool(allow_regression)):
        raise_error(
            "GATE",
            location,
            "submissao bloqueada: candidato dependente de public_validation sem CV",
            impact="1",
            examples=["public_score_name=public_validation", f"cv_count={cv_count}"],
        )
    if bool(block_public_validation_without_cv) and ("public_validation_without_cv" in risk_flags) and (not bool(allow_regression)):
        raise_error(
            "GATE",
            location,
            "submissao bloqueada: risk_flag public_validation_without_cv",
            impact="1",
            examples=risk_flags[:8],
        )

    if bool(block_target_patch) and (not bool(allow_regression)):
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
        if is_extrapolation and (not bool(allow_calibration_extrapolation)) and (not bool(allow_regression)):
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
    return read_score_json(score_json_path=score_json_path, stage="CLI", location=location)


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


def _cmd_download(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out_dir = (repo / args.out).resolve()
    res = download_competition_files(competition=args.competition, out_dir=out_dir)
    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "competition": args.competition,
        "out_dir": _rel_or_abs(out_dir, repo),
        "files": {k: _rel_or_abs(v, repo) for k, v in res.files.items()},
        "sha256": res.sha256,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _cmd_vendor(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    res = vendor_all(repo_root=repo, metric_kernel=args.metric_kernel)
    print(json.dumps(asdict(res), indent=2, default=str, sort_keys=True))
    return 0


def _cmd_build_dataset(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    input_dir = (repo / args.input).resolve()
    out_dir = (repo / args.out).resolve()
    if args.type == "public_validation":
        manifest = build_public_validation_dataset(repo_root=repo, input_dir=input_dir, out_dir=out_dir)
        print(str(manifest))
        return 0
    if args.type == "train_cv_targets":
        manifest = build_train_cv_targets(
            repo_root=repo,
            input_dir=input_dir,
            out_dir=out_dir,
            n_folds=args.n_folds,
            seed=args.seed,
            k=args.k,
            n_hashes=args.n_hashes,
            bands=args.bands,
            memory_budget_mb=int(args.memory_budget_mb),
            max_rows_in_memory=int(args.max_rows_in_memory),
        )
        print(str(manifest))
        return 0
    raise_error("CLI", "src/rna3d_local/cli.py:_cmd_build_dataset", "tipo de dataset desconhecido", impact="1", examples=[args.type])
    raise AssertionError("unreachable")


def _cmd_prepare_labels_parquet(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    manifest = prepare_train_labels_parquet(
        repo_root=repo,
        train_labels_csv=(repo / args.train_labels_csv).resolve(),
        out_dir=(repo / args.out_dir).resolve(),
        rows_per_file=int(args.rows_per_file),
        compression=str(args.compression),
        memory_budget_mb=int(args.memory_budget_mb),
    )
    print(str(manifest))
    return 0


def _cmd_prepare_labels_clean(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    manifest = prepare_train_labels_clean(
        repo_root=repo,
        train_labels_parquet_dir=(repo / args.train_labels_parquet_dir).resolve(),
        out_dir=(repo / args.out_dir).resolve(),
        train_sequences_csv=(repo / args.train_sequences).resolve(),
        rows_per_file=int(args.rows_per_file),
        compression=str(args.compression),
        require_complete_targets=bool(args.require_complete_targets),
        memory_budget_mb=int(args.memory_budget_mb),
    )
    print(str(manifest))
    return 0


def _cmd_build_train_fold(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    input_dir = (repo / args.input).resolve()
    targets_parquet = (repo / args.targets).resolve()
    out_dir = (repo / args.out).resolve()
    manifest = build_train_cv_fold_dataset(
        repo_root=repo,
        input_dir=input_dir,
        targets_parquet=targets_parquet,
        fold_id=int(args.fold),
        out_dir=out_dir,
        train_labels_parquet_dir=(repo / args.train_labels_parquet_dir).resolve(),
        memory_budget_mb=int(args.memory_budget_mb),
    )
    print(str(manifest))
    return 0


def _cmd_export_train_solution(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    targets_path = (repo / args.targets).resolve()
    if not targets_path.exists():
        raise_error("DATA", "src/rna3d_local/cli.py:_cmd_export_train_solution", "targets.parquet nao encontrado", impact="1", examples=[str(targets_path)])
    target_ids = _target_ids_for_fold(
        targets_path=targets_path,
        fold_id=int(args.fold),
        stage="DATA",
        location="src/rna3d_local/cli.py:_cmd_export_train_solution",
    )
    out_path = (repo / args.out).resolve()
    out = export_train_solution_for_targets(
        out_path=out_path,
        target_ids=target_ids,
        train_labels_parquet_dir=(repo / args.train_labels_parquet_dir).resolve(),
        memory_budget_mb=int(args.memory_budget_mb),
    )
    print(str(out))
    return 0


def _cmd_make_sample(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    targets_path = (repo / args.targets).resolve()
    sequences_csv = (repo / args.sequences).resolve()
    if not targets_path.exists():
        raise_error("DATA", "src/rna3d_local/cli.py:_cmd_make_sample", "targets.parquet nao encontrado", impact="1", examples=[str(targets_path)])
    target_ids = _target_ids_for_fold(
        targets_path=targets_path,
        fold_id=int(args.fold),
        stage="DATA",
        location="src/rna3d_local/cli.py:_cmd_make_sample",
    )
    out_path = (repo / args.out).resolve()
    out = make_sample_submission_for_targets(
        sequences_csv=sequences_csv,
        out_path=out_path,
        target_ids=target_ids,
        memory_budget_mb=int(args.memory_budget_mb),
    )
    print(str(out))
    return 0


def _cmd_check_submission(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    sample = (repo / args.sample).resolve()
    submission = (repo / args.submission).resolve()
    validate_submission_against_sample(sample_path=sample, submission_path=submission)
    print("OK")
    return 0


def _load_dataset_manifest(repo: Path, dataset_dir: Path) -> dict:
    location = "src/rna3d_local/cli.py:_load_dataset_manifest"
    mf = dataset_dir / "manifest.json"
    if not mf.exists():
        raise_error("SCORE", location, "manifest.json nao encontrado no dataset_dir", impact="1", examples=[str(mf)])
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise_error("SCORE", location, "falha ao ler manifest.json", impact="1", examples=[f"{type(e).__name__}:{e}"])
    raise AssertionError("unreachable")


def _cmd_score(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    if args.dataset == "public_validation":
        dataset_dir = repo / "data" / "derived" / "public_validation"
    else:
        dataset_dir = (repo / args.dataset_dir).resolve()

    mf = _load_dataset_manifest(repo, dataset_dir)
    sample = repo / mf["sample_submission"]
    solution = repo / mf["solution"]
    metric_py = repo / mf["metric_py"]
    usalign_bin = repo / mf["usalign_bin"]
    submission = (repo / args.submission).resolve()

    result = score_submission(
        sample_submission=sample,
        solution=solution,
        submission=submission,
        metric_py=metric_py,
        usalign_bin=usalign_bin,
        per_target=bool(args.per_target),
        keep_tmp=bool(args.keep_tmp),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
        chunk_size=int(args.chunk_size),
    )

    out_dir = (repo / args.out_dir).resolve()
    if args.out_dir == "runs/auto":
        out_dir = repo / "runs" / f"{_utc_now_compact()}_score"
    meta = {"dataset_dir": _rel_or_abs(dataset_dir, repo), "submission": _rel_or_abs(submission, repo)}
    write_score_artifacts(out_dir=out_dir, result=result, meta=meta)
    print(json.dumps({"score": result.score, "out_dir": _rel_or_abs(out_dir, repo)}, indent=2, sort_keys=True))
    return 0


def _cmd_build_template_db(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    res = build_template_db(
        repo_root=repo,
        train_sequences_path=(repo / args.train_sequences).resolve(),
        train_labels_parquet_dir=(repo / args.train_labels_parquet_dir).resolve(),
        external_templates_path=(repo / args.external_templates).resolve(),
        out_dir=(repo / args.out_dir).resolve(),
        max_train_templates=args.max_train_templates,
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(
        json.dumps(
            {
                "manifest": _rel_or_abs(res.manifest_path, repo),
                "templates": _rel_or_abs(res.templates_path, repo),
                "index": _rel_or_abs(res.index_path, repo),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_retrieve_templates(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    res = retrieve_template_candidates(
        repo_root=repo,
        template_index_path=(repo / args.template_index).resolve(),
        target_sequences_path=(repo / args.targets).resolve(),
        out_path=(repo / args.out).resolve(),
        top_k=int(args.top_k),
        kmer_size=int(args.kmer_size),
        length_weight=float(args.length_weight),
        refine_pool_size=int(args.refine_pool_size),
        refine_alignment_weight=float(args.refine_alignment_weight),
        refine_open_gap_score=float(args.refine_open_gap_score),
        refine_extend_gap_score=float(args.refine_extend_gap_score),
        chunk_size=int(args.chunk_size),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(
        json.dumps(
            {
                "candidates": _rel_or_abs(res.candidates_path, repo),
                "manifest": _rel_or_abs(res.manifest_path, repo),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_predict_tbm(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    location = "src/rna3d_local/cli.py:_cmd_predict_tbm"
    gap_open_scores = _parse_float_list_arg(raw=str(args.gap_open_scores), arg_name="gap_open_scores", location=location)
    gap_extend_scores = _parse_float_list_arg(raw=str(args.gap_extend_scores), arg_name="gap_extend_scores", location=location)
    res = predict_tbm(
        repo_root=repo,
        retrieval_candidates_path=(repo / args.retrieval).resolve(),
        templates_path=(repo / args.templates).resolve(),
        target_sequences_path=(repo / args.targets).resolve(),
        out_path=(repo / args.out).resolve(),
        n_models=int(args.n_models),
        min_coverage=float(args.min_coverage),
        rerank_pool_size=int(args.rerank_pool_size),
        gap_open_scores=gap_open_scores,
        gap_extend_scores=gap_extend_scores,
        max_variants_per_template=int(args.max_variants_per_template),
        perturbation_scale=float(args.perturbation_scale),
        mapping_mode=str(args.mapping_mode),
        projection_mode=str(args.projection_mode),
        qa_model_path=None if args.qa_model is None else (repo / args.qa_model).resolve(),
        qa_device=str(args.qa_device),
        qa_top_pool=int(args.qa_top_pool),
        diversity_lambda=float(args.diversity_lambda),
        chunk_size=int(args.chunk_size),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(
        json.dumps(
            {
                "predictions": _rel_or_abs(res.predictions_path, repo),
                "manifest": _rel_or_abs(res.manifest_path, repo),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_predict_drfold2(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out_path = (repo / args.out).resolve()
    work_dir = (repo / args.work_dir).resolve()
    if args.work_dir == "runs/auto_drfold2":
        work_dir = repo / "runs" / f"{_utc_now_compact()}_drfold2_work"
    predictions_path, manifest_path = predict_drfold2(
        repo_root=repo,
        drfold2_root=(repo / args.drfold_root).resolve(),
        target_sequences_path=(repo / args.targets).resolve(),
        out_path=out_path,
        work_dir=work_dir,
        n_models=int(args.n_models),
        python_bin=str(args.python_bin),
        target_limit=None if args.target_limit is None else int(args.target_limit),
        chunk_size=int(args.chunk_size),
        reuse_existing_targets=bool(args.reuse_existing_targets),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(
        json.dumps(
            {
                "predictions": _rel_or_abs(predictions_path, repo),
                "manifest": _rel_or_abs(manifest_path, repo),
                "work_dir": _rel_or_abs(work_dir, repo),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_train_rnapro(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out_dir = (repo / args.out_dir).resolve()
    if args.out_dir == "runs/auto_rnapro":
        out_dir = repo / "runs" / f"{_utc_now_compact()}_rnapro_train"
    cfg = RnaProConfig(
        feature_dim=int(args.feature_dim),
        kmer_size=int(args.kmer_size),
        n_models=int(args.n_models),
        seed=int(args.seed),
        min_coverage=float(args.min_coverage),
    )
    model_path = train_rnapro(
        repo_root=repo,
        train_sequences_path=(repo / args.train_sequences).resolve(),
        train_labels_parquet_dir=(repo / args.train_labels_parquet_dir).resolve(),
        out_dir=out_dir,
        config=cfg,
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(json.dumps({"model": _rel_or_abs(model_path, repo)}, indent=2, sort_keys=True))
    return 0


def _cmd_predict_rnapro(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out = infer_rnapro(
        repo_root=repo,
        model_dir=(repo / args.model_dir).resolve(),
        target_sequences_path=(repo / args.targets).resolve(),
        out_path=(repo / args.out).resolve(),
        n_models=None if args.n_models is None else int(args.n_models),
        min_coverage=None if args.min_coverage is None else float(args.min_coverage),
        rerank_pool_multiplier=int(args.rerank_pool_multiplier),
        chunk_size=int(args.chunk_size),
        use_template=str(args.use_template),
        template_features_dir=None if args.template_features_dir is None else (repo / args.template_features_dir).resolve(),
        template_source=str(args.template_source),
        mapping_mode=str(args.mapping_mode),
        projection_mode=str(args.projection_mode),
        qa_model_path=None if args.qa_model is None else (repo / args.qa_model).resolve(),
        qa_device=str(args.qa_device),
        qa_top_pool=int(args.qa_top_pool),
        diversity_lambda=float(args.diversity_lambda),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(json.dumps({"predictions": _rel_or_abs(out, repo)}, indent=2, sort_keys=True))
    return 0


def _cmd_train_qa_ranker(args: argparse.Namespace) -> int:
    location = "src/rna3d_local/cli.py:_cmd_train_qa_ranker"
    repo = _find_repo_root(Path.cwd())
    feature_names_raw = [v.strip() for v in str(args.feature_names).split(",") if v.strip()]
    if not feature_names_raw:
        feature_names = QA_FEATURE_NAMES
    else:
        feature_names = tuple(feature_names_raw)
    out = train_qa_ranker(
        candidates_path=(repo / args.candidates).resolve(),
        out_model_path=(repo / args.out_model).resolve(),
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        feature_names=feature_names,
        l2_lambda=float(args.l2_lambda),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    model_path = Path(str(out["model_path"])).resolve()
    train_gate_report = model_path.parent / "train_gate_report.json"
    train_gate = evaluate_training_gate_from_model_json(
        model_json_path=model_path,
        min_val_samples=int(args.min_val_samples),
        max_mae_gap_ratio=float(args.max_mae_gap_ratio),
        max_rmse_gap_ratio=float(args.max_rmse_gap_ratio),
        max_r2_drop=float(args.max_r2_drop),
        max_spearman_drop=float(args.max_spearman_drop),
        max_pearson_drop=float(args.max_pearson_drop),
    )
    write_training_gate_report(report=train_gate, out_path=train_gate_report)
    out["train_gate_report"] = _rel_or_abs(train_gate_report, repo)
    out["train_gate_allowed"] = bool(train_gate.get("allowed", False))
    if (not bool(train_gate.get("allowed", False))) and (not bool(args.allow_overfit_model)):
        raise_error(
            "TRAIN_GATE",
            location,
            "modelo bloqueado por gate anti-overfitting",
            impact=str(len(train_gate.get("reasons", []))),
            examples=[str(x) for x in (train_gate.get("reasons") or [])][:8],
        )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _cmd_train_qa_gnn_ranker(args: argparse.Namespace) -> int:
    location = "src/rna3d_local/cli.py:_cmd_train_qa_gnn_ranker"
    repo = _find_repo_root(Path.cwd())
    feature_names_raw = [v.strip() for v in str(args.feature_names).split(",") if v.strip()]
    if not feature_names_raw:
        feature_names = QA_GNN_DEFAULT_FEATURE_NAMES
    else:
        feature_names = tuple(feature_names_raw)
    out_model = (repo / args.out_model).resolve()
    out_weights = (repo / args.out_weights).resolve() if args.out_weights else out_model.with_suffix(".pt")
    out = train_qa_gnn_ranker(
        candidates_path=(repo / args.candidates).resolve(),
        out_model_path=out_model,
        out_weights_path=out_weights,
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        feature_names=feature_names,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        knn_k=int(args.knn_k),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        device=str(args.device),
    )
    model_path = Path(str(out["model_path"])).resolve()
    train_gate_report = model_path.parent / "train_gate_report.json"
    train_gate = evaluate_training_gate_from_model_json(
        model_json_path=model_path,
        min_val_samples=int(args.min_val_samples),
        max_mae_gap_ratio=float(args.max_mae_gap_ratio),
        max_rmse_gap_ratio=float(args.max_rmse_gap_ratio),
        max_r2_drop=float(args.max_r2_drop),
        max_spearman_drop=float(args.max_spearman_drop),
        max_pearson_drop=float(args.max_pearson_drop),
    )
    write_training_gate_report(report=train_gate, out_path=train_gate_report)
    out["train_gate_report"] = _rel_or_abs(train_gate_report, repo)
    out["train_gate_allowed"] = bool(train_gate.get("allowed", False))
    if (not bool(train_gate.get("allowed", False))) and (not bool(args.allow_overfit_model)):
        raise_error(
            "TRAIN_GATE",
            location,
            "modelo bloqueado por gate anti-overfitting",
            impact=str(len(train_gate.get("reasons", []))),
            examples=[str(x) for x in (train_gate.get("reasons") or [])][:8],
        )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _cmd_score_qa_gnn_ranker(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out = score_candidates_with_qa_gnn(
        candidates_path=(repo / args.candidates).resolve(),
        model_path=(repo / args.model).resolve(),
        out_scores_path=(repo / args.out).resolve(),
        weights_path=None if args.weights is None else (repo / args.weights).resolve(),
        device=str(args.device),
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _cmd_build_candidate_pool(args: argparse.Namespace) -> int:
    location = "src/rna3d_local/cli.py:_cmd_build_candidate_pool"
    repo = _find_repo_root(Path.cwd())
    entries = parse_prediction_entries(raw_entries=list(args.predictions), repo_root=repo, location=location)
    out_path, manifest_path = build_candidate_pool_from_predictions(
        repo_root=repo,
        prediction_entries=entries,
        out_path=(repo / args.out).resolve(),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(
        json.dumps(
            {
                "candidate_pool": _rel_or_abs(out_path, repo),
                "manifest": _rel_or_abs(manifest_path, repo),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_train_qa_rnrank(args: argparse.Namespace) -> int:
    location = "src/rna3d_local/cli.py:_cmd_train_qa_rnrank"
    repo = _find_repo_root(Path.cwd())
    feature_names_raw = [v.strip() for v in str(args.feature_names).split(",") if v.strip()]
    if not feature_names_raw:
        feature_names = QA_RNRANK_DEFAULT_FEATURE_NAMES
    else:
        feature_names = tuple(feature_names_raw)
    out_model = (repo / args.out_model).resolve()
    out_weights = (repo / args.out_weights).resolve() if args.out_weights else out_model.with_suffix(".pt")
    out = train_qa_rnrank(
        candidates_path=(repo / args.candidates).resolve(),
        out_model_path=out_model,
        out_weights_path=out_weights,
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        feature_names=feature_names,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_fraction=float(args.val_fraction),
        rank_weight=float(args.rank_weight),
        regression_weight=float(args.regression_weight),
        combined_reg_weight=float(args.combined_reg_weight),
        combined_rank_weight=float(args.combined_rank_weight),
        seed=int(args.seed),
        device=str(args.device),
    )
    model_path = Path(str(out["model_path"])).resolve()
    train_gate_report = model_path.parent / "train_gate_report.json"
    train_gate = evaluate_training_gate_from_model_json(
        model_json_path=model_path,
        min_val_samples=int(args.min_val_samples),
        max_mae_gap_ratio=float(args.max_mae_gap_ratio),
        max_rmse_gap_ratio=float(args.max_rmse_gap_ratio),
        max_r2_drop=float(args.max_r2_drop),
        max_spearman_drop=float(args.max_spearman_drop),
        max_pearson_drop=float(args.max_pearson_drop),
    )
    write_training_gate_report(report=train_gate, out_path=train_gate_report)
    out["train_gate_report"] = _rel_or_abs(train_gate_report, repo)
    out["train_gate_allowed"] = bool(train_gate.get("allowed", False))
    if (not bool(train_gate.get("allowed", False))) and (not bool(args.allow_overfit_model)):
        raise_error(
            "TRAIN_GATE",
            location,
            "modelo bloqueado por gate anti-overfitting",
            impact=str(len(train_gate.get("reasons", []))),
            examples=[str(x) for x in (train_gate.get("reasons") or [])][:8],
        )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _cmd_score_qa_rnrank(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out = score_candidates_with_qa_rnrank(
        candidates_path=(repo / args.candidates).resolve(),
        model_path=(repo / args.model).resolve(),
        out_scores_path=(repo / args.out).resolve(),
        weights_path=None if args.weights is None else (repo / args.weights).resolve(),
        device=str(args.device),
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _cmd_select_top5_global(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out_path, manifest_path = select_top5_global_with_qa_rnrank(
        candidates_path=(repo / args.candidates).resolve(),
        model_path=(repo / args.model).resolve(),
        out_predictions_path=(repo / args.out).resolve(),
        n_models=int(args.n_models),
        qa_top_pool=int(args.qa_top_pool),
        diversity_lambda=float(args.diversity_lambda),
        weights_path=None if args.weights is None else (repo / args.weights).resolve(),
        device=str(args.device),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(
        json.dumps(
            {
                "predictions": _rel_or_abs(out_path, repo),
                "manifest": _rel_or_abs(manifest_path, repo),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_convert_templates_to_pt(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    manifest_path = convert_templates_to_pt_files(
        repo_root=repo,
        templates_submission_path=(repo / args.templates_submission).resolve(),
        target_sequences_path=(repo / args.targets).resolve(),
        out_dir=(repo / args.out_dir).resolve(),
        n_models=None if args.n_models is None else int(args.n_models),
        template_source=str(args.template_source),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(json.dumps({"manifest": _rel_or_abs(manifest_path, repo)}, indent=2, sort_keys=True))
    return 0


def _cmd_ensemble_predict(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out = blend_predictions(
        tbm_predictions_path=(repo / args.tbm).resolve(),
        rnapro_predictions_path=(repo / args.rnapro).resolve(),
        out_path=(repo / args.out).resolve(),
        tbm_weight=float(args.tbm_weight),
        rnapro_weight=float(args.rnapro_weight),
        dynamic_by_coverage=bool(args.dynamic_by_coverage),
        coverage_power=float(args.coverage_power),
        coverage_floor=float(args.coverage_floor),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(json.dumps({"predictions": _rel_or_abs(out, repo)}, indent=2, sort_keys=True))
    return 0


def _cmd_export_submission(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out = export_submission_from_long(
        sample_submission_path=(repo / args.sample).resolve(),
        predictions_long_path=(repo / args.predictions).resolve(),
        out_submission_path=(repo / args.out).resolve(),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(json.dumps({"submission": _rel_or_abs(out, repo)}, indent=2, sort_keys=True))
    return 0


def _cmd_submit_kaggle(args: argparse.Namespace) -> int:
    location = "src/rna3d_local/cli.py:_cmd_submit_kaggle"
    repo = _find_repo_root(Path.cwd())
    report = (repo / args.gating_report).resolve()
    if args.gating_report == "runs/auto":
        report = repo / "runs" / f"{_utc_now_compact()}_gating_report.json"
    sample_path = (repo / args.sample).resolve()
    submission_path = (repo / args.submission).resolve()
    score_json_path = None if args.score_json is None else (repo / args.score_json).resolve()
    robust_report = None if args.robust_report is None else (repo / args.robust_report).resolve()
    robust_payload = None if robust_report is None else _read_robust_report(report_path=robust_report, location=location)
    readiness_report = None if args.readiness_report is None else (repo / args.readiness_report).resolve()
    readiness_payload = None if readiness_report is None else _read_readiness_report(report_path=readiness_report, location=location)

    assert_submission_allowed(
        sample_path=sample_path,
        submission_path=submission_path,
        report_path=report,
        is_smoke=bool(args.is_smoke),
        is_partial=bool(args.is_partial),
        score_json_path=score_json_path,
        baseline_score=None if args.baseline_score is None else float(args.baseline_score),
        min_improvement=float(args.min_improvement),
        allow_regression=bool(args.allow_regression),
    )

    _enforce_submit_hardening(
        location=location,
        allow_regression=bool(args.allow_regression),
        require_robust_report=bool(args.require_robust_report),
        require_readiness_report=bool(args.require_readiness_report),
        require_min_cv_count=int(args.require_min_cv_count),
        block_public_validation_without_cv=bool(args.block_public_validation_without_cv),
        block_target_patch=bool(args.block_target_patch),
        allow_calibration_extrapolation=bool(args.allow_calibration_extrapolation),
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
        )
        current_local_score = _read_score_json(score_json_path=score_json_path, location=location)
        alignment_decision = build_alignment_decision(
            local_score=float(current_local_score),
            baseline_public_score=float(args.baseline_public_score),
            calibration=calibration,
            method=str(args.calibration_method),
            min_public_improvement=float(args.min_public_improvement),
            min_pairs=int(args.calibration_min_pairs),
            allow_extrapolation=bool(args.allow_calibration_extrapolation),
        )
        calibration["alignment_decision"] = alignment_decision
        write_calibration_report(report=calibration, out_path=calibration_report_out)
        if not bool(alignment_decision.get("allowed", False)) and not bool(args.allow_regression):
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
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_calibrate_kaggle_local(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out = (repo / args.out).resolve()
    report = build_kaggle_local_calibration(
        competition=str(args.competition),
        page_size=int(args.page_size),
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
            allow_extrapolation=bool(args.allow_calibration_extrapolation),
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


def _cmd_evaluate_robust(args: argparse.Namespace) -> int:
    location = "src/rna3d_local/cli.py:_cmd_evaluate_robust"
    repo = _find_repo_root(Path.cwd())
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
        min_public_improvement=float(args.min_public_improvement),
        min_cv_count=int(args.min_cv_count),
        block_public_validation_without_cv=bool(args.block_public_validation_without_cv),
        allow_calibration_extrapolation=bool(args.allow_calibration_extrapolation),
    )
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
        require_baseline=bool(args.require_baseline),
        require_public_score=bool(args.require_public_score),
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
        min_public_improvement=float(args.min_public_improvement),
        allow_calibration_extrapolation=bool(args.allow_calibration_extrapolation),
        min_calibration_pearson=float(args.min_calibration_pearson),
        min_calibration_spearman=float(args.min_calibration_spearman),
        block_public_validation_without_cv=bool(args.block_public_validation_without_cv),
    )
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


def _cmd_research_sync_literature(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out_dir = (repo / args.out_dir).resolve()
    if args.out_dir == "runs/research/literature/auto":
        topic_slug = str(args.topic_slug or "topic").strip().lower()
        topic_slug = "".join([ch if ch.isalnum() else "-" for ch in topic_slug]).strip("-")
        if topic_slug in {"", "topic"}:
            raw = str(args.topic).strip().lower()
            topic_slug = "".join([ch if ch.isalnum() else "-" for ch in raw]).strip("-")
        while "--" in topic_slug:
            topic_slug = topic_slug.replace("--", "-")
        if not topic_slug:
            topic_slug = "topic"
        out_dir = repo / "runs" / "research" / "literature" / f"{_utc_now_compact()}_{topic_slug}"
    res = sync_literature(
        topic=str(args.topic),
        out_dir=out_dir,
        limit_per_source=int(args.limit_per_source),
        timeout_s=int(args.timeout_s),
        download_pdfs=bool(args.download_pdfs),
        strict_pdf_download=bool(args.strict_pdf_download),
        max_pdf_mb=int(args.max_pdf_mb),
        strict_sources=bool(args.strict_sources),
    )
    print(
        json.dumps(
            {
                "out_dir": _rel_or_abs(res.out_dir, repo),
                "papers": _rel_or_abs(res.papers_path, repo),
                "manifest": _rel_or_abs(res.manifest_path, repo),
                "related_work": _rel_or_abs(res.related_work_path, repo),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_research_run(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out_base_dir = (repo / args.out_base_dir).resolve()
    res = run_experiment(
        repo_root=repo,
        config_path=(repo / args.config).resolve(),
        run_id=str(args.run_id),
        out_base_dir=out_base_dir,
        allow_existing_run_dir=bool(args.allow_existing_run_dir),
    )
    print(
        json.dumps(
            {
                "run_dir": _rel_or_abs(res.run_dir, repo),
                "manifest": _rel_or_abs(res.manifest_path, repo),
                "results": _rel_or_abs(res.results_path, repo),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_research_verify(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    allowed_statuses = tuple([s.strip() for s in str(args.allowed_statuses).split(",") if s.strip()])
    res = verify_run(
        repo_root=repo,
        run_dir=(repo / args.run_dir).resolve(),
        allowed_statuses=allowed_statuses,
    )
    print(
        json.dumps(
            {
                "accepted": bool(res.accepted),
                "verify_path": _rel_or_abs(res.verify_path, repo),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_research_report(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    run_dir = (repo / args.run_dir).resolve()
    out_path = (repo / args.out).resolve()
    if args.out == "runs/research/reports/auto.md":
        out_path = repo / "runs" / "research" / "reports" / f"{run_dir.name}.md"
    report_path = generate_report(run_dir=run_dir, out_path=out_path)
    print(json.dumps({"report": _rel_or_abs(report_path, repo)}, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rna3d_local", add_help=True)
    sp = p.add_subparsers(dest="cmd", required=True)

    d = sp.add_parser("download", help="Download competition files via Kaggle API")
    d.add_argument("--competition", default="stanford-rna-3d-folding-2")
    d.add_argument("--out", default="input/stanford-rna-3d-folding-2")
    d.set_defaults(fn=_cmd_download)

    v = sp.add_parser("vendor", help="Vendor Kaggle metric + USalign")
    v.add_argument("--metric-kernel", default=DEFAULT_METRIC_KERNEL)
    v.set_defaults(fn=_cmd_vendor)

    b = sp.add_parser("build-dataset", help="Build derived local datasets")
    b.add_argument("--type", choices=["public_validation", "train_cv_targets"], required=True)
    b.add_argument("--input", default="input/stanford-rna-3d-folding-2")
    b.add_argument("--out", required=True)
    b.add_argument("--n-folds", type=int, default=5)
    b.add_argument("--seed", type=int, default=123)
    b.add_argument("--k", type=int, default=5)
    b.add_argument("--n-hashes", type=int, default=32)
    b.add_argument("--bands", type=int, default=8)
    b.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    b.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    b.set_defaults(fn=_cmd_build_dataset)

    lp = sp.add_parser("prepare-labels-parquet", help="Convert train_labels.csv to canonical partitioned parquet")
    lp.add_argument("--train-labels-csv", default="input/stanford-rna-3d-folding-2/train_labels.csv")
    lp.add_argument("--out-dir", default="data/derived/train_labels_parquet")
    lp.add_argument("--rows-per-file", type=int, default=2_000_000)
    lp.add_argument("--compression", choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"], default="zstd")
    lp.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    lp.set_defaults(fn=_cmd_prepare_labels_parquet)

    lpc = sp.add_parser(
        "prepare-train-labels-clean",
        help="Create cleaned labels parquet by explicitly dropping rows with null xyz",
    )
    lpc.add_argument("--train-labels-parquet-dir", required=True, help="Input labels parquet dir (part-*.parquet)")
    lpc.add_argument("--out-dir", default="data/derived/train_labels_parquet_nonnull_xyz")
    lpc.add_argument("--train-sequences", default="input/stanford-rna-3d-folding-2/train_sequences.csv")
    lpc.add_argument("--rows-per-file", type=int, default=2_000_000)
    lpc.add_argument("--compression", choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"], default="zstd")
    grp = lpc.add_mutually_exclusive_group()
    grp.add_argument(
        "--require-complete-targets",
        dest="require_complete_targets",
        action="store_true",
        default=True,
        help="Fail if any target from train_sequences is missing after cleaning (default)",
    )
    grp.add_argument(
        "--allow-incomplete-targets",
        dest="require_complete_targets",
        action="store_false",
        help="Allow missing targets after cleaning",
    )
    lpc.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    lpc.set_defaults(fn=_cmd_prepare_labels_clean)

    bf = sp.add_parser("build-train-fold", help="Build a scoring dataset for one CV fold (sample + solution + manifest + target_sequences)")
    bf.add_argument("--input", default="input/stanford-rna-3d-folding-2")
    bf.add_argument("--targets", required=True, help="Path to targets.parquet from train_cv_targets dataset")
    bf.add_argument("--fold", type=int, required=True)
    bf.add_argument("--out", required=True)
    bf.add_argument("--train-labels-parquet-dir", required=True, help="Canonical labels parquet dir (part-*.parquet)")
    bf.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    bf.set_defaults(fn=_cmd_build_train_fold)

    e = sp.add_parser("export-train-solution", help="Export train_labels subset as wide solution (parquet)")
    e.add_argument("--targets", required=True, help="Path to targets.parquet from train_cv_targets dataset")
    e.add_argument("--fold", type=int, required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--train-labels-parquet-dir", required=True, help="Canonical labels parquet dir (part-*.parquet)")
    e.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    e.set_defaults(fn=_cmd_export_train_solution)

    m = sp.add_parser("make-sample", help="Create sample_submission template for a CV fold (CSV)")
    m.add_argument("--targets", required=True, help="Path to targets.parquet from train_cv_targets dataset")
    m.add_argument("--fold", type=int, required=True)
    m.add_argument("--sequences", required=True, help="train_sequences.csv (or other sequences CSV)")
    m.add_argument("--out", required=True)
    m.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    m.set_defaults(fn=_cmd_make_sample)

    c = sp.add_parser("check-submission", help="Strict contract validation vs sample_submission")
    c.add_argument("--sample", required=True)
    c.add_argument("--submission", required=True)
    c.set_defaults(fn=_cmd_check_submission)

    s = sp.add_parser("score", help="Score a submission locally with vendored Kaggle metric")
    s.add_argument("--dataset", choices=["public_validation"], default=None)
    s.add_argument("--dataset-dir", default="data/derived/public_validation")
    s.add_argument("--submission", required=True)
    s.add_argument("--out-dir", default="runs/auto")
    s.add_argument("--per-target", action="store_true")
    s.add_argument("--keep-tmp", action="store_true")
    s.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    s.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    s.add_argument("--chunk-size", type=int, default=100_000)
    s.set_defaults(fn=_cmd_score)

    tdb = sp.add_parser("build-template-db", help="Build template database (local train + external templates)")
    tdb.add_argument("--train-sequences", default="input/stanford-rna-3d-folding-2/train_sequences.csv")
    tdb.add_argument("--train-labels-parquet-dir", required=True, help="Canonical labels parquet dir (part-*.parquet)")
    tdb.add_argument("--external-templates", required=True, help="CSV/Parquet with template_id,sequence,release_date,resid,resname,x,y,z")
    tdb.add_argument("--out-dir", default="data/derived/template_db")
    tdb.add_argument("--max-train-templates", type=int, default=None)
    tdb.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    tdb.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    tdb.set_defaults(fn=_cmd_build_template_db)

    rt = sp.add_parser("retrieve-templates", help="Retrieve temporal-valid template candidates per target")
    rt.add_argument("--template-index", default="data/derived/template_db/template_index.parquet")
    rt.add_argument("--targets", default="input/stanford-rna-3d-folding-2/test_sequences.csv")
    rt.add_argument("--out", default="data/derived/template_db/retrieval_candidates.parquet")
    rt.add_argument("--top-k", type=int, default=20)
    rt.add_argument("--kmer-size", type=int, default=3)
    rt.add_argument("--length-weight", type=float, default=0.15)
    rt.add_argument("--refine-pool-size", type=int, default=64)
    rt.add_argument("--refine-alignment-weight", type=float, default=0.25)
    rt.add_argument("--refine-open-gap-score", type=float, default=-5.0)
    rt.add_argument("--refine-extend-gap-score", type=float, default=-1.0)
    rt.add_argument("--chunk-size", type=int, default=200_000)
    rt.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    rt.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    rt.set_defaults(fn=_cmd_retrieve_templates)

    pt = sp.add_parser("predict-tbm", help="Generate TBM predictions in long format")
    pt.add_argument("--retrieval", default="data/derived/template_db/retrieval_candidates.parquet")
    pt.add_argument("--templates", default="data/derived/template_db/templates.parquet")
    pt.add_argument("--targets", default="input/stanford-rna-3d-folding-2/test_sequences.csv")
    pt.add_argument("--out", required=True)
    pt.add_argument("--n-models", type=int, default=5)
    pt.add_argument("--min-coverage", type=float, default=0.35)
    pt.add_argument("--rerank-pool-size", type=int, default=64)
    pt.add_argument("--gap-open-scores", type=str, default="-5.0")
    pt.add_argument("--gap-extend-scores", type=str, default="-1.0")
    pt.add_argument("--max-variants-per-template", type=int, default=1)
    pt.add_argument("--perturbation-scale", type=float, default=0.0)
    pt.add_argument("--mapping-mode", choices=["strict_match", "hybrid", "chemical_class"], default="hybrid")
    pt.add_argument("--projection-mode", choices=["target_linear", "template_warped"], default="template_warped")
    pt.add_argument("--qa-model", default=None, help="Optional qa_model.json path")
    pt.add_argument("--qa-device", choices=["auto", "cpu", "cuda"], default="cuda")
    pt.add_argument("--qa-top-pool", type=int, default=40)
    pt.add_argument("--diversity-lambda", type=float, default=0.15)
    pt.add_argument("--chunk-size", type=int, default=200_000)
    pt.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    pt.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    pt.set_defaults(fn=_cmd_predict_tbm)

    pd2 = sp.add_parser("predict-drfold2", help="Run local DRfold2 inference and export long predictions")
    pd2.add_argument("--drfold-root", required=True, help="Path to local DRfold2 repository (with model_hub and Arena)")
    pd2.add_argument("--targets", default="input/stanford-rna-3d-folding-2/test_sequences.csv")
    pd2.add_argument("--out", required=True)
    pd2.add_argument("--work-dir", default="runs/auto_drfold2")
    pd2.add_argument("--n-models", type=int, default=5)
    pd2.add_argument("--python-bin", default="python")
    pd2.add_argument("--target-limit", type=int, default=None, help="Optional limit for smoke validation")
    pd2.add_argument("--chunk-size", type=int, default=200_000)
    pd2.add_argument("--reuse-existing-targets", action="store_true", help="Reuse existing per-target DRfold2 outputs in work-dir")
    pd2.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    pd2.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    pd2.set_defaults(fn=_cmd_predict_drfold2)

    tr = sp.add_parser("train-rnapro", help="Train RNAPro proxy model locally")
    tr.add_argument("--train-sequences", default="input/stanford-rna-3d-folding-2/train_sequences.csv")
    tr.add_argument("--train-labels-parquet-dir", required=True, help="Canonical labels parquet dir (part-*.parquet)")
    tr.add_argument("--out-dir", default="runs/auto_rnapro")
    tr.add_argument("--feature-dim", type=int, default=256)
    tr.add_argument("--kmer-size", type=int, default=4)
    tr.add_argument("--n-models", type=int, default=5)
    tr.add_argument("--seed", type=int, default=123)
    tr.add_argument("--min-coverage", type=float, default=0.30)
    tr.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    tr.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    tr.set_defaults(fn=_cmd_train_rnapro)

    ct = sp.add_parser(
        "convert-templates-to-pt",
        help="Convert wide template submission (CSV/Parquet) into per-target template_features.pt artifacts",
    )
    ct.add_argument("--templates-submission", required=True, help="Wide template submission (ID,resname,resid,x_i,y_i,z_i)")
    ct.add_argument("--targets", default="input/stanford-rna-3d-folding-2/test_sequences.csv")
    ct.add_argument("--out-dir", required=True)
    ct.add_argument("--n-models", type=int, default=None, help="If set, enforce exact number of models in submission header")
    ct.add_argument("--template-source", choices=["tbm", "mmseqs2", "external"], default="tbm")
    ct.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    ct.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    ct.set_defaults(fn=_cmd_convert_templates_to_pt)

    ir = sp.add_parser("predict-rnapro", help="Run RNAPro inference in long format")
    ir.add_argument("--model-dir", required=True)
    ir.add_argument("--targets", default="input/stanford-rna-3d-folding-2/test_sequences.csv")
    ir.add_argument("--out", required=True)
    ir.add_argument("--n-models", type=int, default=None)
    ir.add_argument("--min-coverage", type=float, default=None)
    ir.add_argument("--rerank-pool-multiplier", type=int, default=8)
    ir.add_argument("--chunk-size", type=int, default=200_000)
    ir.add_argument("--use-template", choices=["none", "ca_precomputed"], default="none")
    ir.add_argument("--template-features-dir", default=None, help="Required when --use-template ca_precomputed")
    ir.add_argument("--template-source", choices=["tbm", "mmseqs2", "external"], default="tbm")
    ir.add_argument("--mapping-mode", choices=["strict_match", "hybrid", "chemical_class"], default="hybrid")
    ir.add_argument("--projection-mode", choices=["target_linear", "template_warped"], default="template_warped")
    ir.add_argument("--qa-model", default=None, help="Optional qa_model.json path")
    ir.add_argument("--qa-device", choices=["auto", "cpu", "cuda"], default="cuda")
    ir.add_argument("--qa-top-pool", type=int, default=40)
    ir.add_argument("--diversity-lambda", type=float, default=0.15)
    ir.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    ir.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    ir.set_defaults(fn=_cmd_predict_rnapro)

    qtr = sp.add_parser("train-qa-ranker", help="Train lightweight QA ranker from candidate feature table")
    qtr.add_argument("--candidates", required=True, help="Parquet/CSV with feature columns and labels")
    qtr.add_argument("--out-model", required=True, help="Output qa_model.json")
    qtr.add_argument("--label-col", default="label")
    qtr.add_argument("--group-col", default="target_id")
    qtr.add_argument(
        "--feature-names",
        default=",".join(QA_FEATURE_NAMES),
        help="Comma-separated feature names; empty keeps built-in defaults",
    )
    qtr.add_argument("--l2-lambda", type=float, default=1.0)
    qtr.add_argument("--val-fraction", type=float, default=0.2)
    qtr.add_argument("--seed", type=int, default=123)
    qtr.add_argument("--min-val-samples", type=int, default=32)
    qtr.add_argument("--max-mae-gap-ratio", type=float, default=0.40)
    qtr.add_argument("--max-rmse-gap-ratio", type=float, default=0.40)
    qtr.add_argument("--max-r2-drop", type=float, default=0.30)
    qtr.add_argument("--max-spearman-drop", type=float, default=0.30)
    qtr.add_argument("--max-pearson-drop", type=float, default=0.30)
    qtr.add_argument("--allow-overfit-model", action="store_true", default=False)
    qtr.set_defaults(fn=_cmd_train_qa_ranker)

    qgnn = sp.add_parser("train-qa-gnn-ranker", help="Train graph QA ranker from candidate feature table")
    qgnn.add_argument("--candidates", required=True, help="Parquet/CSV with feature columns and labels")
    qgnn.add_argument("--out-model", required=True, help="Output qa_gnn_model.json")
    qgnn.add_argument("--out-weights", default=None, help="Optional output .pt path (default: out-model with .pt)")
    qgnn.add_argument("--label-col", default="label")
    qgnn.add_argument("--group-col", default="target_id")
    qgnn.add_argument(
        "--feature-names",
        default=",".join(QA_GNN_DEFAULT_FEATURE_NAMES),
        help="Comma-separated feature names; empty keeps built-in defaults",
    )
    qgnn.add_argument("--hidden-dim", type=int, default=64)
    qgnn.add_argument("--num-layers", type=int, default=2)
    qgnn.add_argument("--dropout", type=float, default=0.1)
    qgnn.add_argument("--knn-k", type=int, default=8)
    qgnn.add_argument("--epochs", type=int, default=120)
    qgnn.add_argument("--lr", type=float, default=1e-3)
    qgnn.add_argument("--weight-decay", type=float, default=1e-4)
    qgnn.add_argument("--val-fraction", type=float, default=0.2)
    qgnn.add_argument("--seed", type=int, default=123)
    qgnn.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    qgnn.add_argument("--min-val-samples", type=int, default=32)
    qgnn.add_argument("--max-mae-gap-ratio", type=float, default=0.40)
    qgnn.add_argument("--max-rmse-gap-ratio", type=float, default=0.40)
    qgnn.add_argument("--max-r2-drop", type=float, default=0.30)
    qgnn.add_argument("--max-spearman-drop", type=float, default=0.30)
    qgnn.add_argument("--max-pearson-drop", type=float, default=0.30)
    qgnn.add_argument("--allow-overfit-model", action="store_true", default=False)
    qgnn.set_defaults(fn=_cmd_train_qa_gnn_ranker)

    qgnn_sc = sp.add_parser("score-qa-gnn-ranker", help="Score candidate table with trained graph QA ranker")
    qgnn_sc.add_argument("--candidates", required=True, help="Parquet/CSV with candidate features")
    qgnn_sc.add_argument("--model", required=True, help="qa_gnn_model.json path")
    qgnn_sc.add_argument("--weights", default=None, help="Optional .pt path (default from model json)")
    qgnn_sc.add_argument("--out", required=True, help="Output scored table (parquet/csv)")
    qgnn_sc.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    qgnn_sc.set_defaults(fn=_cmd_score_qa_gnn_ranker)

    bcp = sp.add_parser("build-candidate-pool", help="Build global candidate pool from long prediction tables")
    bcp.add_argument(
        "--predictions",
        action="append",
        required=True,
        help="Prediction input path or source=path (repeatable). Example: --predictions tbm=runs/tbm.parquet --predictions rnapro=runs/rnapro.parquet",
    )
    bcp.add_argument("--out", required=True, help="Output candidate pool parquet")
    bcp.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    bcp.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    bcp.set_defaults(fn=_cmd_build_candidate_pool)

    qrn = sp.add_parser("train-qa-rnrank", help="Train RNArank-style QA reranker (hybrid regression + ranking)")
    qrn.add_argument("--candidates", required=True, help="Candidate pool table (parquet/csv) with labels")
    qrn.add_argument("--out-model", required=True, help="Output qa_rnrank_model.json")
    qrn.add_argument("--out-weights", default=None, help="Optional output .pt path (default: out-model with .pt)")
    qrn.add_argument("--label-col", default="label")
    qrn.add_argument("--group-col", default="target_id")
    qrn.add_argument(
        "--feature-names",
        default=",".join(QA_RNRANK_DEFAULT_FEATURE_NAMES),
        help="Comma-separated feature names; empty keeps built-in defaults",
    )
    qrn.add_argument("--hidden-dim", type=int, default=128)
    qrn.add_argument("--dropout", type=float, default=0.10)
    qrn.add_argument("--epochs", type=int, default=160)
    qrn.add_argument("--lr", type=float, default=1e-3)
    qrn.add_argument("--weight-decay", type=float, default=1e-4)
    qrn.add_argument("--val-fraction", type=float, default=0.2)
    qrn.add_argument("--rank-weight", type=float, default=0.4)
    qrn.add_argument("--regression-weight", type=float, default=0.6)
    qrn.add_argument("--combined-reg-weight", type=float, default=0.6)
    qrn.add_argument("--combined-rank-weight", type=float, default=0.4)
    qrn.add_argument("--seed", type=int, default=123)
    qrn.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    qrn.add_argument("--min-val-samples", type=int, default=32)
    qrn.add_argument("--max-mae-gap-ratio", type=float, default=0.40)
    qrn.add_argument("--max-rmse-gap-ratio", type=float, default=0.40)
    qrn.add_argument("--max-r2-drop", type=float, default=0.30)
    qrn.add_argument("--max-spearman-drop", type=float, default=0.30)
    qrn.add_argument("--max-pearson-drop", type=float, default=0.30)
    qrn.add_argument("--allow-overfit-model", action="store_true", default=False)
    qrn.set_defaults(fn=_cmd_train_qa_rnrank)

    qrn_sc = sp.add_parser("score-qa-rnrank", help="Score candidate table with trained RNArank-style QA reranker")
    qrn_sc.add_argument("--candidates", required=True, help="Candidate pool table (parquet/csv)")
    qrn_sc.add_argument("--model", required=True, help="qa_rnrank_model.json path")
    qrn_sc.add_argument("--weights", default=None, help="Optional .pt path (default from model json)")
    qrn_sc.add_argument("--out", required=True, help="Output scored table (parquet/csv)")
    qrn_sc.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    qrn_sc.set_defaults(fn=_cmd_score_qa_rnrank)

    st5 = sp.add_parser("select-top5-global", help="Select final 5 candidates globally per target using QA RNArank + diversity")
    st5.add_argument("--candidates", required=True, help="Candidate pool table (parquet/csv)")
    st5.add_argument("--model", required=True, help="qa_rnrank_model.json path")
    st5.add_argument("--weights", default=None, help="Optional .pt path (default from model json)")
    st5.add_argument("--out", required=True, help="Output long predictions parquet")
    st5.add_argument("--n-models", type=int, default=5)
    st5.add_argument("--qa-top-pool", type=int, default=80)
    st5.add_argument("--diversity-lambda", type=float, default=0.15)
    st5.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    st5.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    st5.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    st5.set_defaults(fn=_cmd_select_top5_global)

    ep = sp.add_parser("ensemble-predict", help="Blend TBM and RNAPro predictions")
    ep.add_argument("--tbm", required=True)
    ep.add_argument("--rnapro", required=True)
    ep.add_argument("--out", required=True)
    ep.add_argument("--tbm-weight", type=float, default=0.6)
    ep.add_argument("--rnapro-weight", type=float, default=0.4)
    ep.add_argument("--dynamic-by-coverage", action="store_true")
    ep.add_argument("--coverage-power", type=float, default=1.0)
    ep.add_argument("--coverage-floor", type=float, default=1e-6)
    ep.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    ep.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    ep.set_defaults(fn=_cmd_ensemble_predict)

    ex = sp.add_parser("export-submission", help="Export strict Kaggle submission from long predictions")
    ex.add_argument("--sample", default="input/stanford-rna-3d-folding-2/sample_submission.csv")
    ex.add_argument("--predictions", required=True)
    ex.add_argument("--out", required=True)
    ex.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    ex.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    ex.set_defaults(fn=_cmd_export_submission)

    sk = sp.add_parser("submit-kaggle", help="Notebook-only Kaggle submit after strict local + calibrated gating")
    sk.add_argument("--competition", default="stanford-rna-3d-folding-2")
    sk.add_argument("--sample", default="input/stanford-rna-3d-folding-2/sample_submission.csv")
    sk.add_argument("--submission", required=True, help="Local submission.csv used for strict validation/gating before submit")
    sk.add_argument("--notebook-ref", required=True, help="Kaggle notebook ref owner/notebook-slug")
    sk.add_argument("--notebook-version", type=int, required=True, help="Notebook version number to submit")
    sk.add_argument("--notebook-file", default=None, help="File name generated inside notebook output (default: basename of --submission)")
    sk.add_argument("--message", required=True)
    sk.add_argument("--gating-report", default="runs/auto")
    sk.add_argument("--calibration-report", default="runs/auto")
    sk.add_argument("--robust-report", default=None, help="Optional robust evaluation report path; if provided and allowed=false submit is blocked")
    sk.add_argument(
        "--readiness-report",
        default=None,
        help="Submit readiness report generated by evaluate-submit-readiness; if provided and allowed=false submit is blocked",
    )
    sk.add_argument("--score-json", default=None)
    sk.add_argument("--baseline-score", type=float, default=None)
    sk.add_argument("--min-improvement", type=float, default=0.0, help="Minimum strict improvement required over baseline-score")
    sk.add_argument(
        "--baseline-public-score",
        type=float,
        default=None,
        help="If provided, enables calibrated gate using Kaggle submission history (requires --score-json)",
    )
    sk.add_argument("--min-public-improvement", type=float, default=0.0)
    sk.add_argument("--calibration-method", choices=["median", "p10", "worst_seen", "linear_fit"], default="p10")
    sk.add_argument("--calibration-page-size", type=int, default=100)
    sk.add_argument("--calibration-min-pairs", type=int, default=3)
    sk.add_argument("--allow-calibration-extrapolation", action="store_true", default=False)
    sk.add_argument("--require-robust-report", action="store_true", default=True)
    sk.add_argument("--allow-missing-robust-report", dest="require_robust_report", action="store_false")
    sk.add_argument("--require-readiness-report", action="store_true", default=True)
    sk.add_argument("--allow-missing-readiness-report", dest="require_readiness_report", action="store_false")
    sk.add_argument("--require-min-cv-count", type=int, default=2)
    sk.add_argument("--block-public-validation-without-cv", action="store_true", default=True)
    sk.add_argument("--allow-public-validation-without-cv", dest="block_public_validation_without_cv", action="store_false")
    sk.add_argument("--block-target-patch", action="store_true", default=True)
    sk.add_argument("--allow-target-patch", dest="block_target_patch", action="store_false")
    sk.add_argument("--allow-regression", action="store_true")
    sk.add_argument("--is-smoke", action="store_true")
    sk.add_argument("--is-partial", action="store_true")
    sk.set_defaults(fn=_cmd_submit_kaggle)

    kc = sp.add_parser("calibrate-kaggle-local", help="Build local-vs-Kaggle public calibration report from submission history")
    kc.add_argument("--competition", default="stanford-rna-3d-folding-2")
    kc.add_argument("--out", default="runs/kaggle_calibration/latest.json")
    kc.add_argument("--page-size", type=int, default=100)
    kc.add_argument("--local-score", type=float, default=None, help="Optional candidate local score to estimate expected public range")
    kc.add_argument("--baseline-public-score", type=float, default=None, help="Optional baseline public score for a strict calibrated go/no-go decision")
    kc.add_argument("--method", choices=["median", "p10", "worst_seen", "linear_fit"], default="p10")
    kc.add_argument("--min-public-improvement", type=float, default=0.0)
    kc.add_argument("--min-pairs", type=int, default=3)
    kc.add_argument("--allow-calibration-extrapolation", action="store_true", default=False)
    kc.set_defaults(fn=_cmd_calibrate_kaggle_local)

    rb = sp.add_parser("evaluate-robust", help="Aggregate multiple local score.json files and apply robust + calibrated go/no-go gate")
    rb.add_argument(
        "--score",
        action="append",
        required=True,
        help="Named score entry in the format name=path/to/score.json. Use prefix cv: for CV splits (e.g., cv:fold0=...).",
    )
    rb.add_argument("--out", default="runs/auto")
    rb.add_argument("--public-score-name", default="public_validation")
    rb.add_argument("--baseline-robust-score", type=float, default=None)
    rb.add_argument("--min-robust-improvement", type=float, default=0.0)
    rb.add_argument("--competition", default="stanford-rna-3d-folding-2")
    rb.add_argument("--baseline-public-score", type=float, default=None)
    rb.add_argument("--calibration-method", choices=["median", "p10", "worst_seen", "linear_fit"], default="p10")
    rb.add_argument("--calibration-page-size", type=int, default=100)
    rb.add_argument("--calibration-min-pairs", type=int, default=3)
    rb.add_argument("--min-public-improvement", type=float, default=0.0)
    rb.add_argument("--min-cv-count", type=int, default=2)
    rb.add_argument("--block-public-validation-without-cv", action="store_true", default=True)
    rb.add_argument("--allow-public-validation-without-cv", dest="block_public_validation_without_cv", action="store_false")
    rb.add_argument("--allow-calibration-extrapolation", action="store_true", default=False)
    rb.set_defaults(fn=_cmd_evaluate_robust)

    sr = sp.add_parser(
        "evaluate-submit-readiness",
        help="Evaluate complete pre-submit readiness (CV stability + strict improvements + local->Kaggle calibration)",
    )
    sr.add_argument(
        "--candidate-score",
        action="append",
        required=True,
        help="Candidate named score entry in the format name=path/to/score.json. Use prefix cv: for CV splits.",
    )
    sr.add_argument(
        "--baseline-score",
        action="append",
        default=[],
        help="Baseline named score entry in the format name=path/to/score.json. Must match candidate fold names for fold-level checks.",
    )
    sr.add_argument("--out", default="runs/auto")
    sr.add_argument("--public-score-name", default="public_validation")
    sr.add_argument("--require-baseline", action="store_true", default=True)
    sr.add_argument("--allow-missing-baseline", dest="require_baseline", action="store_false")
    sr.add_argument("--require-public-score", action="store_true", default=True)
    sr.add_argument("--allow-missing-public-score", dest="require_public_score", action="store_false")
    sr.add_argument("--min-cv-count", type=int, default=3)
    sr.add_argument("--min-cv-improvement-count", type=int, default=2)
    sr.add_argument("--min-fold-improvement", type=float, default=0.0)
    sr.add_argument("--max-cv-regression", type=float, default=0.0)
    sr.add_argument("--min-robust-improvement", type=float, default=0.0)
    sr.add_argument("--min-public-local-improvement", type=float, default=0.0)
    sr.add_argument("--max-cv-std", type=float, default=0.03)
    sr.add_argument("--max-cv-gap", type=float, default=0.08)
    sr.add_argument("--competition", default="stanford-rna-3d-folding-2")
    sr.add_argument("--baseline-public-score", type=float, default=None)
    sr.add_argument("--calibration-method", choices=["median", "p10", "worst_seen", "linear_fit"], default="p10")
    sr.add_argument("--calibration-page-size", type=int, default=100)
    sr.add_argument("--calibration-min-pairs", type=int, default=3)
    sr.add_argument("--min-public-improvement", type=float, default=0.0)
    sr.add_argument("--allow-calibration-extrapolation", action="store_true", default=False)
    sr.add_argument("--min-calibration-pearson", type=float, default=0.0)
    sr.add_argument("--min-calibration-spearman", type=float, default=0.0)
    sr.add_argument("--block-public-validation-without-cv", action="store_true", default=True)
    sr.add_argument("--allow-public-validation-without-cv", dest="block_public_validation_without_cv", action="store_false")
    sr.set_defaults(fn=_cmd_evaluate_submit_readiness)

    tg = sp.add_parser("evaluate-train-gate", help="Evaluate anti-overfitting gate from trained QA model JSON (train_metrics vs val_metrics)")
    tg.add_argument("--model", required=True, help="qa_model.json or qa_gnn_model.json")
    tg.add_argument("--out", default="runs/auto")
    tg.add_argument("--min-val-samples", type=int, default=32)
    tg.add_argument("--max-mae-gap-ratio", type=float, default=0.40)
    tg.add_argument("--max-rmse-gap-ratio", type=float, default=0.40)
    tg.add_argument("--max-r2-drop", type=float, default=0.30)
    tg.add_argument("--max-spearman-drop", type=float, default=0.30)
    tg.add_argument("--max-pearson-drop", type=float, default=0.30)
    tg.add_argument("--allow-overfit-model", action="store_true", default=False)
    tg.set_defaults(fn=_cmd_evaluate_train_gate)

    rsl = sp.add_parser("research-sync-literature", help="Search literature and optionally download OA PDFs")
    rsl.add_argument("--topic", required=True)
    rsl.add_argument("--topic-slug", default="topic")
    rsl.add_argument("--out-dir", default="runs/research/literature/auto")
    rsl.add_argument("--limit-per-source", type=int, default=5)
    rsl.add_argument("--timeout-s", type=int, default=30)
    rsl.add_argument("--max-pdf-mb", type=int, default=30)
    rsl.add_argument("--download-pdfs", action="store_true", default=True)
    rsl.add_argument("--no-download-pdfs", dest="download_pdfs", action="store_false")
    rsl.add_argument("--strict-pdf-download", action="store_true", default=True)
    rsl.add_argument("--allow-pdf-download-failures", dest="strict_pdf_download", action="store_false")
    rsl.add_argument("--strict-sources", action="store_true", default=True)
    rsl.add_argument("--allow-source-failures", dest="strict_sources", action="store_false")
    rsl.set_defaults(fn=_cmd_research_sync_literature)

    rr = sp.add_parser("research-run", help="Run experiment harness and persist structured artifacts")
    rr.add_argument("--config", required=True, help="YAML/JSON config for the experiment")
    rr.add_argument("--run-id", required=True)
    rr.add_argument("--out-base-dir", default="runs/research/experiments")
    rr.add_argument("--allow-existing-run-dir", action="store_true")
    rr.set_defaults(fn=_cmd_research_run)

    rv = sp.add_parser("research-verify", help="Run strict gate: solver + checks + reproducibility")
    rv.add_argument("--run-dir", required=True, help="Experiment run directory")
    rv.add_argument("--allowed-statuses", default="optimal,feasible,success,complete")
    rv.set_defaults(fn=_cmd_research_verify)

    rp = sp.add_parser("research-report", help="Generate markdown report from experiment artifacts")
    rp.add_argument("--run-dir", required=True, help="Experiment run directory")
    rp.add_argument("--out", default="runs/research/reports/auto.md")
    rp.set_defaults(fn=_cmd_research_report)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    try:
        return int(args.fn(args))
    except PipelineError as e:
        print(str(e))
        return 2
