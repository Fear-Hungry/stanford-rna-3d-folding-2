from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .contracts import require_columns
from .errors import raise_error
from .io_tables import read_table
from .utils import rel_or_abs, utc_now_iso, write_json


@dataclass(frozen=True)
class HomologyFoldEvaluationResult:
    report_path: Path


def _resolve_score_column(df: pl.DataFrame, *, score_column: str | None, stage: str, location: str, label: str) -> str:
    if score_column is not None:
        if score_column not in df.columns:
            raise_error(stage, location, f"{label} sem coluna de score solicitada", impact="1", examples=[score_column])
        return score_column
    for candidate in ("score", "final_score", "tm_score", "metric"):
        if candidate in df.columns:
            return candidate
    raise_error(stage, location, f"{label} sem coluna de score suportada", impact="1", examples=df.columns[:8])


def _parse_orphan_flag(value: object, *, stage: str, location: str, target_id: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise_error(stage, location, "is_orphan invalido", impact="1", examples=[f"{target_id}:{value}"])


def evaluate_homology_folds(
    *,
    repo_root: Path,
    train_folds_path: Path,
    target_metrics_path: Path,
    report_path: Path,
    orphan_labels_path: Path | None,
    retrieval_path: Path | None,
    metric_column: str | None,
    retrieval_score_column: str | None,
    orphan_score_threshold: float,
    orphan_weight: float,
) -> HomologyFoldEvaluationResult:
    stage = "EVALUATE_HOMOLOGY_FOLDS"
    location = "src/rna3d_local/homology_eval.py:evaluate_homology_folds"
    if orphan_weight <= 0.0 or orphan_weight >= 1.0:
        raise_error(stage, location, "orphan_weight deve estar em (0,1)", impact="1", examples=[str(orphan_weight)])
    if orphan_score_threshold < 0.0:
        raise_error(stage, location, "orphan_score_threshold invalido (>=0)", impact="1", examples=[str(orphan_score_threshold)])
    if (orphan_labels_path is None and retrieval_path is None) or (orphan_labels_path is not None and retrieval_path is not None):
        raise_error(
            stage,
            location,
            "informe exatamente uma fonte de orphan (labels ou retrieval)",
            impact="1",
            examples=[f"orphan_labels={orphan_labels_path}", f"retrieval={retrieval_path}"],
        )

    folds = read_table(train_folds_path, stage=stage, location=location)
    require_columns(folds, ["target_id", "fold_id"], stage=stage, location=location, label="train_folds")
    if "domain_label" not in folds.columns:
        raise_error(stage, location, "train_folds sem domain_label", impact="1", examples=["domain_label"])
    fold_dup = (
        folds.select(pl.col("target_id").cast(pl.Utf8).alias("target_id"))
        .group_by("target_id")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
    )
    if fold_dup.height > 0:
        examples = fold_dup.get_column("target_id").head(8).to_list()
        raise_error(stage, location, "train_folds com target_id duplicado", impact=str(int(fold_dup.height)), examples=[str(x) for x in examples])

    metrics = read_table(target_metrics_path, stage=stage, location=location)
    require_columns(metrics, ["target_id"], stage=stage, location=location, label="target_metrics")
    metric_col = _resolve_score_column(metrics, score_column=metric_column, stage=stage, location=location, label="target_metrics")
    metrics_view = metrics.select(pl.col("target_id").cast(pl.Utf8).alias("target_id"), pl.col(metric_col).cast(pl.Float64).alias("score"))
    metrics_dup = metrics_view.group_by("target_id").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if metrics_dup.height > 0:
        examples = metrics_dup.get_column("target_id").head(8).to_list()
        raise_error(stage, location, "target_metrics com target_id duplicado", impact=str(int(metrics_dup.height)), examples=[str(x) for x in examples])

    fold_targets = set(str(item) for item in folds.get_column("target_id").cast(pl.Utf8).to_list())
    metric_targets = set(str(item) for item in metrics_view.get_column("target_id").to_list())
    missing_metrics = sorted(fold_targets - metric_targets)
    if missing_metrics:
        raise_error(stage, location, "target sem metrica no fold evaluation", impact=str(len(missing_metrics)), examples=missing_metrics[:8])

    orphans: dict[str, bool] = {}
    orphan_source = "none"
    selected_retrieval_score_col: str | None = None
    if orphan_labels_path is not None:
        orphan_source = "labels"
        orphan_labels = read_table(orphan_labels_path, stage=stage, location=location)
        require_columns(orphan_labels, ["target_id", "is_orphan"], stage=stage, location=location, label="orphan_labels")
        rows = orphan_labels.select("target_id", "is_orphan").iter_rows(named=True)
        for row in rows:
            target_id = str(row["target_id"])
            if target_id in orphans:
                raise_error(stage, location, "orphan_labels com target_id duplicado", impact="1", examples=[target_id])
            if target_id in fold_targets:
                orphans[target_id] = _parse_orphan_flag(row["is_orphan"], stage=stage, location=location, target_id=target_id)
    if retrieval_path is not None:
        orphan_source = "retrieval"
        retrieval = read_table(retrieval_path, stage=stage, location=location)
        require_columns(retrieval, ["target_id"], stage=stage, location=location, label="retrieval")
        retrieval_score_col = _resolve_score_column(
            retrieval,
            score_column=retrieval_score_column,
            stage=stage,
            location=location,
            label="retrieval",
        )
        selected_retrieval_score_col = retrieval_score_col
        agg = (
            retrieval.select(
                pl.col("target_id").cast(pl.Utf8).alias("target_id"),
                pl.col(retrieval_score_col).cast(pl.Float64).alias("template_score"),
            )
            .group_by("target_id")
            .agg(pl.col("template_score").max().alias("template_score"))
        )
        max_score = {str(row["target_id"]): float(row["template_score"]) for row in agg.iter_rows(named=True)}
        for target_id in fold_targets:
            orphans[target_id] = bool(max_score.get(target_id, float("-inf")) < float(orphan_score_threshold))

    missing_orphan = sorted(fold_targets - set(orphans.keys()))
    if missing_orphan:
        raise_error(stage, location, "target sem classificacao orphan", impact=str(len(missing_orphan)), examples=missing_orphan[:8])

    joined = (
        folds.select(
            pl.col("target_id").cast(pl.Utf8).alias("target_id"),
            pl.col("fold_id").cast(pl.Int64).alias("fold_id"),
            pl.col("domain_label").cast(pl.Utf8).alias("domain_label"),
        )
        .join(metrics_view, on="target_id", how="inner")
        .with_columns(pl.col("target_id").map_elements(lambda target_id: bool(orphans[str(target_id)]), return_dtype=pl.Boolean).alias("is_orphan"))
        .sort(["fold_id", "target_id"])
    )
    orphan_count = int(joined.filter(pl.col("is_orphan")).height)
    non_orphan_count = int(joined.filter(~pl.col("is_orphan")).height)
    if orphan_count == 0:
        raise_error(stage, location, "nenhum orphan identificado para avaliacao", impact="1", examples=[orphan_source])
    if non_orphan_count == 0:
        raise_error(stage, location, "todos os alvos classificados como orphan", impact="1", examples=[orphan_source])

    overall_mean = float(joined.get_column("score").mean())
    orphan_mean = float(joined.filter(pl.col("is_orphan")).get_column("score").mean())
    non_orphan_mean = float(joined.filter(~pl.col("is_orphan")).get_column("score").mean())
    priority_score = (float(orphan_weight) * orphan_mean) + ((1.0 - float(orphan_weight)) * non_orphan_mean)

    fold_metrics: list[dict[str, object]] = []
    for fold_id in sorted(joined.get_column("fold_id").unique().to_list()):
        fold_df = joined.filter(pl.col("fold_id") == int(fold_id))
        fold_orphans = fold_df.filter(pl.col("is_orphan"))
        fold_non_orphans = fold_df.filter(~pl.col("is_orphan"))
        fold_metrics.append(
            {
                "fold_id": int(fold_id),
                "n_targets": int(fold_df.height),
                "mean_score": float(fold_df.get_column("score").mean()),
                "orphan_count": int(fold_orphans.height),
                "orphan_mean_score": None if fold_orphans.height == 0 else float(fold_orphans.get_column("score").mean()),
                "non_orphan_count": int(fold_non_orphans.height),
                "non_orphan_mean_score": None if fold_non_orphans.height == 0 else float(fold_non_orphans.get_column("score").mean()),
            }
        )

    domain_metrics: list[dict[str, object]] = []
    for domain_label in sorted(joined.get_column("domain_label").unique().to_list()):
        domain_df = joined.filter(pl.col("domain_label") == str(domain_label))
        domain_orphans = domain_df.filter(pl.col("is_orphan"))
        domain_metrics.append(
            {
                "domain_label": str(domain_label),
                "n_targets": int(domain_df.height),
                "orphan_count": int(domain_orphans.height),
                "orphan_ratio": float(domain_orphans.height / domain_df.height),
                "mean_score": float(domain_df.get_column("score").mean()),
                "orphan_mean_score": None if domain_orphans.height == 0 else float(domain_orphans.get_column("score").mean()),
            }
        )

    write_json(
        report_path,
        {
            "created_utc": utc_now_iso(),
            "paths": {
                "train_folds": rel_or_abs(train_folds_path, repo_root),
                "target_metrics": rel_or_abs(target_metrics_path, repo_root),
                "orphan_labels": None if orphan_labels_path is None else rel_or_abs(orphan_labels_path, repo_root),
                "retrieval": None if retrieval_path is None else rel_or_abs(retrieval_path, repo_root),
            },
            "params": {
                "metric_column": metric_col,
                "retrieval_score_column": selected_retrieval_score_col,
                "orphan_source": orphan_source,
                "orphan_score_threshold": float(orphan_score_threshold),
                "orphan_weight": float(orphan_weight),
            },
            "overall": {
                "n_targets": int(joined.height),
                "orphan_count": orphan_count,
                "non_orphan_count": non_orphan_count,
                "mean_score": overall_mean,
                "orphan_mean_score": orphan_mean,
                "non_orphan_mean_score": non_orphan_mean,
                "priority_score": priority_score,
            },
            "fold_metrics": fold_metrics,
            "domain_metrics": domain_metrics,
        },
    )
    return HomologyFoldEvaluationResult(report_path=report_path)
