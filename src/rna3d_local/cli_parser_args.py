from __future__ import annotations

import argparse

from .bigdata import DEFAULT_MAX_ROWS_IN_MEMORY, DEFAULT_MEMORY_BUDGET_MB

_CALIBRATION_OVERRIDES_HELP = (
    "Optional JSON with calibration cleanup rules: "
    '{"by_ref": {"<submission_ref>": <local_score>}, '
    '"exclude_refs": ["<submission_ref>"], '
    '"only_override_refs": true|false}'
)


def add_memory_budget_arg(
    parser: argparse.ArgumentParser,
    *,
    default_memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
) -> None:
    parser.add_argument("--memory-budget-mb", type=int, default=int(default_memory_budget_mb))


def add_memory_budget_and_rows_args(
    parser: argparse.ArgumentParser,
    *,
    default_memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    default_max_rows_in_memory: int = DEFAULT_MAX_ROWS_IN_MEMORY,
) -> None:
    add_memory_budget_arg(parser, default_memory_budget_mb=default_memory_budget_mb)
    parser.add_argument("--max-rows-in-memory", type=int, default=int(default_max_rows_in_memory))


def add_compute_backend_args(
    parser: argparse.ArgumentParser,
    *,
    include_hash_dim: bool = False,
    default_gpu_memory_budget_mb: int = 12_288,
    default_gpu_precision: str = "fp32",
    default_gpu_hash_dim: int = 4096,
) -> None:
    parser.add_argument("--compute-backend", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu-memory-budget-mb", type=int, default=int(default_gpu_memory_budget_mb))
    parser.add_argument("--gpu-precision", choices=["fp32", "fp16"], default=str(default_gpu_precision))
    if include_hash_dim:
        parser.add_argument("--gpu-hash-dim", type=int, default=int(default_gpu_hash_dim))


def add_calibration_overrides_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--calibration-overrides",
        default=None,
        help=_CALIBRATION_OVERRIDES_HELP,
    )


def add_calibration_history_gate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--calibration-method", choices=["median", "p10", "worst_seen", "linear_fit"], default="p10")
    parser.add_argument("--calibration-page-size", type=int, default=100)
    parser.add_argument("--calibration-min-pairs", type=int, default=3)
    add_calibration_overrides_arg(parser)
    parser.add_argument("--min-public-improvement", type=float, default=0.0)


def add_calibration_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=["median", "p10", "worst_seen", "linear_fit"], default="p10")
    parser.add_argument("--min-public-improvement", type=float, default=0.0)
    parser.add_argument("--min-pairs", type=int, default=3)
    add_calibration_overrides_arg(parser)


def add_training_overfit_gate_args(
    parser: argparse.ArgumentParser,
    *,
    include_allow_overfit_model: bool = True,
) -> None:
    parser.add_argument("--min-val-samples", type=int, default=32)
    parser.add_argument("--max-mae-gap-ratio", type=float, default=0.40)
    parser.add_argument("--max-rmse-gap-ratio", type=float, default=0.40)
    parser.add_argument("--max-r2-drop", type=float, default=0.30)
    parser.add_argument("--max-spearman-drop", type=float, default=0.30)
    parser.add_argument("--max-pearson-drop", type=float, default=0.30)
    if include_allow_overfit_model:
        parser.add_argument("--allow-overfit-model", action="store_true", default=False)


__all__ = [
    "add_calibration_history_gate_args",
    "add_calibration_overrides_arg",
    "add_calibration_report_args",
    "add_compute_backend_args",
    "add_memory_budget_and_rows_args",
    "add_memory_budget_arg",
    "add_training_overfit_gate_args",
]
