from __future__ import annotations

import argparse

from .candidate_pool import LABEL_METHOD_CHOICES, LABEL_METHOD_TM_SCORE_USALIGN
from .cli_commands import (
    _cmd_add_labels_candidate_pool,
    _cmd_build_candidate_pool,
    _cmd_score_qa_gnn_ranker,
    _cmd_score_qa_rnrank,
    _cmd_select_top5_global,
    _cmd_train_qa_gnn_ranker,
    _cmd_train_qa_ranker,
    _cmd_train_qa_rnrank,
)
from .cli_parser_args import add_compute_backend_args, add_memory_budget_and_rows_args, add_training_overfit_gate_args
from .qa_gnn_ranker import QA_GNN_DEFAULT_FEATURE_NAMES
from .qa_ranker import QA_FEATURE_NAMES
from .qa_rnrank import QA_RNRANK_DEFAULT_FEATURE_NAMES


def register_qa_parsers(sp: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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
    add_training_overfit_gate_args(qtr)
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
    add_training_overfit_gate_args(qgnn)
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
    add_compute_backend_args(bcp)
    add_memory_budget_and_rows_args(bcp)
    bcp.set_defaults(fn=_cmd_build_candidate_pool)

    alc = sp.add_parser("add-labels-candidate-pool", help="Add labels to candidate pool using train labels")
    alc.add_argument("--candidates", required=True, help="Candidate pool parquet/CSV")
    alc.add_argument("--solution", required=True, help="Solution file in public format (CSV or parquet)")
    alc.add_argument("--out", required=True, help="Output labeled candidate pool parquet")
    alc.add_argument("--label-col", default="label")
    alc.add_argument("--label-source-col", default="label_source")
    alc.add_argument("--label-source", default=None, help="Optional label source name; default follows label-method")
    alc.add_argument("--label-method", choices=LABEL_METHOD_CHOICES, default=LABEL_METHOD_TM_SCORE_USALIGN)
    alc.add_argument("--metric-py", default="vendor/tm_score_permutechains/metric.py")
    alc.add_argument("--usalign-bin", default="vendor/usalign/USalign")
    add_memory_budget_and_rows_args(alc)
    alc.set_defaults(fn=_cmd_add_labels_candidate_pool)

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
    add_training_overfit_gate_args(qrn)
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
    add_memory_budget_and_rows_args(st5)
    st5.set_defaults(fn=_cmd_select_top5_global)
