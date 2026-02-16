from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.homology_eval import evaluate_homology_folds


def _write_folds(path: Path) -> None:
    pl.DataFrame(
        [
            {"target_id": "T1", "cluster_id": "C1", "domain_label": "ribozyme", "fold_id": 0},
            {"target_id": "T2", "cluster_id": "C2", "domain_label": "ribozyme", "fold_id": 1},
            {"target_id": "T3", "cluster_id": "C3", "domain_label": "crispr_cas", "fold_id": 0},
            {"target_id": "T4", "cluster_id": "C4", "domain_label": "crispr_cas", "fold_id": 1},
        ]
    ).write_parquet(path)


def test_evaluate_homology_folds_prioritizes_orphans(tmp_path: Path) -> None:
    folds = tmp_path / "folds.parquet"
    metrics = tmp_path / "metrics.parquet"
    retrieval = tmp_path / "retrieval.parquet"
    report = tmp_path / "report.json"
    _write_folds(folds)
    pl.DataFrame(
        [
            {"target_id": "T1", "score": 0.20},
            {"target_id": "T2", "score": 0.80},
            {"target_id": "T3", "score": 0.10},
            {"target_id": "T4", "score": 0.70},
        ]
    ).write_parquet(metrics)
    pl.DataFrame(
        [
            {"target_id": "T1", "final_score": 0.20},
            {"target_id": "T2", "final_score": 0.90},
            {"target_id": "T3", "final_score": 0.30},
            {"target_id": "T4", "final_score": 0.85},
        ]
    ).write_parquet(retrieval)
    out = evaluate_homology_folds(
        repo_root=tmp_path,
        train_folds_path=folds,
        target_metrics_path=metrics,
        report_path=report,
        orphan_labels_path=None,
        retrieval_path=retrieval,
        metric_column="score",
        retrieval_score_column="final_score",
        orphan_score_threshold=0.65,
        orphan_weight=0.70,
    )
    payload = json.loads(out.report_path.read_text(encoding="utf-8"))
    assert payload["overall"]["orphan_count"] == 2
    assert payload["overall"]["non_orphan_count"] == 2
    assert payload["overall"]["priority_score"] == pytest.approx(0.33, abs=1e-6)


def test_evaluate_homology_folds_fails_without_orphans(tmp_path: Path) -> None:
    folds = tmp_path / "folds.parquet"
    metrics = tmp_path / "metrics.parquet"
    retrieval = tmp_path / "retrieval.parquet"
    _write_folds(folds)
    pl.DataFrame(
        [
            {"target_id": "T1", "score": 0.20},
            {"target_id": "T2", "score": 0.80},
            {"target_id": "T3", "score": 0.10},
            {"target_id": "T4", "score": 0.70},
        ]
    ).write_parquet(metrics)
    pl.DataFrame(
        [
            {"target_id": "T1", "final_score": 0.95},
            {"target_id": "T2", "final_score": 0.90},
            {"target_id": "T3", "final_score": 0.88},
            {"target_id": "T4", "final_score": 0.85},
        ]
    ).write_parquet(retrieval)
    with pytest.raises(PipelineError, match="nenhum orphan"):
        evaluate_homology_folds(
            repo_root=tmp_path,
            train_folds_path=folds,
            target_metrics_path=metrics,
            report_path=tmp_path / "report.json",
            orphan_labels_path=None,
            retrieval_path=retrieval,
            metric_column="score",
            retrieval_score_column="final_score",
            orphan_score_threshold=0.65,
            orphan_weight=0.70,
        )
