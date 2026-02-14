from __future__ import annotations

import argparse

from .cli_commands import (
    _cmd_audit_external_templates,
    _cmd_build_template_db,
    _cmd_convert_templates_to_pt,
    _cmd_ensemble_predict,
    _cmd_export_submission,
    _cmd_predict_drfold2,
    _cmd_predict_rnapro,
    _cmd_predict_tbm,
    _cmd_retrieve_templates,
    _cmd_train_rnapro,
)
from .cli_parser_args import add_compute_backend_args, add_memory_budget_and_rows_args


def register_template_parsers(sp: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    tdb = sp.add_parser("build-template-db", help="Build template database (local train + external templates)")
    tdb.add_argument("--train-sequences", default="input/stanford-rna-3d-folding-2/train_sequences.csv")
    tdb.add_argument("--train-labels-parquet-dir", required=True, help="Canonical labels parquet dir (part-*.parquet)")
    tdb.add_argument("--external-templates", required=True, help="CSV/Parquet with template_id,sequence,release_date,resid,resname,x,y,z")
    tdb.add_argument("--out-dir", default="data/derived/template_db")
    tdb.add_argument("--max-train-templates", type=int, default=None)
    add_memory_budget_and_rows_args(tdb)
    tdb.set_defaults(fn=_cmd_build_template_db)

    at = sp.add_parser("audit-external-templates", help="Run strict validation for external template table")
    at.add_argument("--external-templates", required=True, help="CSV/Parquet with template coordinates")
    at.add_argument("--out", default="runs/auto", help="Output audit report JSON")
    add_memory_budget_and_rows_args(at)
    at.set_defaults(fn=_cmd_audit_external_templates)

    rt = sp.add_parser("retrieve-templates", help="Retrieve temporal-valid template candidates per target")
    rt.add_argument("--template-index", default="data/derived/template_db/template_index.parquet")
    rt.add_argument("--targets", default="input/stanford-rna-3d-folding-2/test_sequences.csv")
    rt.add_argument("--out", default="data/derived/template_db/retrieval_candidates.parquet")
    rt.add_argument("--top-k", type=int, default=64)
    rt.add_argument("--kmer-size", type=int, default=3)
    rt.add_argument("--length-weight", type=float, default=0.15)
    rt.add_argument("--refine-pool-size", type=int, default=192)
    rt.add_argument("--refine-alignment-weight", type=float, default=0.35)
    rt.add_argument("--refine-open-gap-score", type=float, default=-5.0)
    rt.add_argument("--refine-extend-gap-score", type=float, default=-1.0)
    add_compute_backend_args(rt, include_hash_dim=True)
    rt.add_argument("--cache-dir", default="runs/cache/retrieval")
    rt.add_argument("--no-cache", action="store_true", default=False)
    rt.add_argument("--chunk-size", type=int, default=200_000)
    add_memory_budget_and_rows_args(rt)
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
    pt.add_argument("--max-mismatch-ratio", type=float, default=None, help="Descarta candidatos com mismatch_ratio > limiar")
    pt.add_argument("--qa-model", default=None, help="Optional qa_model.json path")
    pt.add_argument("--qa-device", choices=["auto", "cpu", "cuda"], default="cuda")
    pt.add_argument("--qa-top-pool", type=int, default=40)
    pt.add_argument("--diversity-lambda", type=float, default=0.15)
    add_compute_backend_args(pt)
    pt.add_argument("--chunk-size", type=int, default=200_000)
    add_memory_budget_and_rows_args(pt)
    pt.set_defaults(fn=_cmd_predict_tbm)

    pd2 = sp.add_parser("predict-drfold2", help="Run local DRfold2 inference and export long predictions")
    pd2.add_argument("--drfold-root", required=True, help="Path to local DRfold2 repository (with model_hub and Arena)")
    pd2.add_argument("--targets", default="input/stanford-rna-3d-folding-2/test_sequences.csv")
    pd2.add_argument("--out", required=True)
    pd2.add_argument("--work-dir", default="runs/auto_drfold2")
    pd2.add_argument("--n-models", type=int, default=5)
    pd2.add_argument("--python-bin", default="python")
    pd2.add_argument("--target-limit", type=int, default=None, help="Optional limit for smoke validation")
    pd2.add_argument("--target-ids-file", default=None, help="Optional file with one target_id per line (or table with target_id column)")
    pd2.add_argument("--chunk-size", type=int, default=200_000)
    pd2.add_argument("--reuse-existing-targets", action="store_true", help="Reuse existing per-target DRfold2 outputs in work-dir")
    add_memory_budget_and_rows_args(pd2)
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
    add_compute_backend_args(tr)
    add_memory_budget_and_rows_args(tr)
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
    add_memory_budget_and_rows_args(ct)
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
    add_compute_backend_args(ir)
    add_memory_budget_and_rows_args(ir)
    ir.set_defaults(fn=_cmd_predict_rnapro)


def register_template_post_qa_parsers(sp: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    ep = sp.add_parser("ensemble-predict", help="Blocked: coordinate blend is disabled in competitive pipeline")
    ep.add_argument("--tbm", required=True)
    ep.add_argument("--rnapro", required=True)
    ep.add_argument("--out", required=True)
    ep.add_argument("--tbm-weight", type=float, default=0.6)
    ep.add_argument("--rnapro-weight", type=float, default=0.4)
    ep.add_argument("--dynamic-by-coverage", action="store_true")
    ep.add_argument("--coverage-power", type=float, default=1.0)
    ep.add_argument("--coverage-floor", type=float, default=1e-6)
    add_memory_budget_and_rows_args(ep)
    ep.set_defaults(fn=_cmd_ensemble_predict)

    ex = sp.add_parser("export-submission", help="Export strict Kaggle submission from long predictions")
    ex.add_argument("--sample", default="input/stanford-rna-3d-folding-2/sample_submission.csv")
    ex.add_argument("--predictions", required=True)
    ex.add_argument("--out", required=True)
    add_memory_budget_and_rows_args(ex)
    ex.set_defaults(fn=_cmd_export_submission)
