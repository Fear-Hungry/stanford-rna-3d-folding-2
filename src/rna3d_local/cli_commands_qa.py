from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_pool import add_labels_to_candidate_pool, build_candidate_pool_from_predictions, parse_prediction_entries
from .cli_commands_common import _find_repo_root, _rel_or_abs
from .errors import raise_error
from .qa_gnn_ranker import QA_GNN_DEFAULT_FEATURE_NAMES, score_candidates_with_qa_gnn, train_qa_gnn_ranker
from .qa_ranker import QA_FEATURE_NAMES, train_qa_ranker
from .qa_rnrank import QA_RNRANK_DEFAULT_FEATURE_NAMES, score_candidates_with_qa_rnrank, select_top5_global_with_qa_rnrank, train_qa_rnrank
from .training_gate import evaluate_training_gate_from_model_json, write_training_gate_report

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
        compute_backend=str(args.compute_backend),
        gpu_memory_budget_mb=int(args.gpu_memory_budget_mb),
        gpu_precision=str(args.gpu_precision),
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

def _cmd_add_labels_candidate_pool(args: argparse.Namespace) -> int:
    location = "src/rna3d_local/cli.py:_cmd_add_labels_candidate_pool"
    repo = _find_repo_root(Path.cwd())
    metric_py_path = None if args.metric_py is None else (repo / args.metric_py).resolve()
    usalign_bin_path = None if args.usalign_bin is None else (repo / args.usalign_bin).resolve()
    out_path, manifest_path = add_labels_to_candidate_pool(
        candidate_pool_path=(repo / args.candidates).resolve(),
        solution_path=(repo / args.solution).resolve(),
        out_path=(repo / args.out).resolve(),
        label_col=str(args.label_col),
        label_source_col=str(args.label_source_col),
        label_source_name=None if args.label_source is None else str(args.label_source),
        label_method=str(args.label_method),
        metric_py_path=metric_py_path,
        usalign_bin_path=usalign_bin_path,
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(
        json.dumps(
            {
                "labeled_candidate_pool": _rel_or_abs(out_path, repo),
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

