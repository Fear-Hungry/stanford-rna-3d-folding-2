from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cli_commands_common import _find_repo_root, _parse_float_list_arg, _rel_or_abs, _utc_now_compact
from .drfold2 import predict_drfold2
from .errors import raise_error
from .export import export_submission_from_long
from .retrieval import retrieve_template_candidates
from .rnapro import RnaProConfig, infer_rnapro, train_rnapro
from .tbm_predictor import predict_tbm
from .template_audit import audit_external_templates
from .template_db import build_template_db
from .template_pt import convert_templates_to_pt_files

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

def _cmd_audit_external_templates(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out = (repo / args.out).resolve()
    if args.out == "runs/auto":
        out = repo / "runs" / f"{_utc_now_compact()}_external_templates_audit.json"
    report = audit_external_templates(
        external_templates_path=(repo / args.external_templates).resolve(),
        out_report_path=out,
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(
        json.dumps(
            {
                "report": _rel_or_abs(out, repo),
                "n_rows": int(report.get("n_rows", 0)),
                "n_templates": int(report.get("n_templates", 0)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0

def _cmd_retrieve_templates(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    cache_dir = None if bool(args.no_cache) else (repo / args.cache_dir).resolve()
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
        compute_backend=str(args.compute_backend),
        gpu_memory_budget_mb=int(args.gpu_memory_budget_mb),
        gpu_precision=str(args.gpu_precision),
        gpu_hash_dim=int(args.gpu_hash_dim),
        cache_dir=cache_dir,
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
        max_mismatch_ratio=None if args.max_mismatch_ratio is None else float(args.max_mismatch_ratio),
        qa_model_path=None if args.qa_model is None else (repo / args.qa_model).resolve(),
        qa_device=str(args.qa_device),
        qa_top_pool=int(args.qa_top_pool),
        diversity_lambda=float(args.diversity_lambda),
        compute_backend=str(args.compute_backend),
        gpu_memory_budget_mb=int(args.gpu_memory_budget_mb),
        gpu_precision=str(args.gpu_precision),
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
    target_ids_file = None if args.target_ids_file is None else (repo / args.target_ids_file).resolve()
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
        target_ids_file=target_ids_file,
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
        compute_backend=str(args.compute_backend),
        gpu_memory_budget_mb=int(args.gpu_memory_budget_mb),
        gpu_precision=str(args.gpu_precision),
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
        compute_backend=str(args.compute_backend),
        gpu_memory_budget_mb=int(args.gpu_memory_budget_mb),
        gpu_precision=str(args.gpu_precision),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(json.dumps({"predictions": _rel_or_abs(out, repo)}, indent=2, sort_keys=True))
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
    raise_error(
        "ENSEMBLE",
        "src/rna3d_local/cli_commands_templates.py:_cmd_ensemble_predict",
        "blend de coordenadas bloqueado por contrato competitivo",
        impact="1",
        examples=["use build-candidate-pool + select-top5-global"],
    )
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
