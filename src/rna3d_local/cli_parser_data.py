from __future__ import annotations

import argparse

from .cli_commands import (
    _cmd_build_dataset,
    _cmd_build_train_fold,
    _cmd_check_submission,
    _cmd_download,
    _cmd_export_train_solution,
    _cmd_make_sample,
    _cmd_prepare_labels_clean,
    _cmd_prepare_labels_parquet,
    _cmd_score,
    _cmd_vendor,
)
from .cli_parser_args import add_memory_budget_and_rows_args, add_memory_budget_arg
from .vendor import DEFAULT_METRIC_KERNEL


def register_data_parsers(sp: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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
    add_memory_budget_and_rows_args(b)
    b.set_defaults(fn=_cmd_build_dataset)

    lp = sp.add_parser("prepare-labels-parquet", help="Convert train_labels.csv to canonical partitioned parquet")
    lp.add_argument("--train-labels-csv", default="input/stanford-rna-3d-folding-2/train_labels.csv")
    lp.add_argument("--out-dir", default="data/derived/train_labels_parquet")
    lp.add_argument("--rows-per-file", type=int, default=2_000_000)
    lp.add_argument("--compression", choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"], default="zstd")
    add_memory_budget_arg(lp)
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
    add_memory_budget_arg(lpc)
    lpc.set_defaults(fn=_cmd_prepare_labels_clean)

    bf = sp.add_parser("build-train-fold", help="Build a scoring dataset for one CV fold (sample + solution + manifest + target_sequences)")
    bf.add_argument("--input", default="input/stanford-rna-3d-folding-2")
    bf.add_argument("--targets", required=True, help="Path to targets.parquet from train_cv_targets dataset")
    bf.add_argument("--fold", type=int, required=True)
    bf.add_argument("--out", required=True)
    bf.add_argument("--train-labels-parquet-dir", required=True, help="Canonical labels parquet dir (part-*.parquet)")
    add_memory_budget_arg(bf)
    bf.set_defaults(fn=_cmd_build_train_fold)

    e = sp.add_parser("export-train-solution", help="Export train_labels subset as wide solution (parquet)")
    e.add_argument("--targets", required=True, help="Path to targets.parquet from train_cv_targets dataset")
    e.add_argument("--fold", type=int, required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--train-labels-parquet-dir", required=True, help="Canonical labels parquet dir (part-*.parquet)")
    add_memory_budget_arg(e)
    e.set_defaults(fn=_cmd_export_train_solution)

    m = sp.add_parser("make-sample", help="Create sample_submission template for a CV fold (CSV)")
    m.add_argument("--targets", required=True, help="Path to targets.parquet from train_cv_targets dataset")
    m.add_argument("--fold", type=int, required=True)
    m.add_argument("--sequences", required=True, help="train_sequences.csv (or other sequences CSV)")
    m.add_argument("--out", required=True)
    add_memory_budget_arg(m)
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
    add_memory_budget_and_rows_args(s)
    s.add_argument("--chunk-size", type=int, default=100_000)
    s.set_defaults(fn=_cmd_score)
