from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.qa_gnn_ranker import (
    QA_GNN_DEFAULT_FEATURE_NAMES,
    is_qa_gnn_model_file,
    load_qa_gnn_runtime,
    score_candidate_feature_dicts_with_qa_gnn_runtime,
    score_candidates_with_qa_gnn,
    train_qa_gnn_ranker,
)


def _make_train_table(path: Path) -> None:
    rows = []
    for t in range(8):
        target_id = f"T{t:02d}"
        for m in range(10):
            coverage = 0.25 + (0.06 * m) + (0.01 * (t % 3))
            similarity = 0.10 + (0.05 * m) + (0.02 * (t % 2))
            path_length = 20.0 + (2.0 * m) + (1.5 * t)
            step_mean = 1.0 + (0.02 * m)
            step_std = 0.3 + (0.01 * (m % 4))
            radius_gyr = 2.0 + (0.08 * m) + (0.03 * t)
            gap_open_score = -5.0 if (m % 2 == 0) else -3.0
            gap_extend_score = -1.0 if (m % 3 == 0) else -0.5
            label = (0.55 * coverage) + (0.35 * similarity) - (0.02 * step_std) + (0.004 * radius_gyr)
            rows.append(
                {
                    "target_id": target_id,
                    "model_id": m + 1,
                    "coverage": coverage,
                    "similarity": similarity,
                    "path_length": path_length,
                    "step_mean": step_mean,
                    "step_std": step_std,
                    "radius_gyr": radius_gyr,
                    "gap_open_score": gap_open_score,
                    "gap_extend_score": gap_extend_score,
                    "label": label,
                }
            )
    pl.DataFrame(rows).write_parquet(path)


def test_train_and_score_qa_gnn_ranker(tmp_path: Path) -> None:
    train_path = tmp_path / "qa_gnn_train.parquet"
    _make_train_table(train_path)

    model_path = tmp_path / "qa_gnn_model.json"
    weights_path = tmp_path / "qa_gnn_model.pt"
    out = train_qa_gnn_ranker(
        candidates_path=train_path,
        out_model_path=model_path,
        out_weights_path=weights_path,
        feature_names=QA_GNN_DEFAULT_FEATURE_NAMES,
        label_col="label",
        group_col="target_id",
        hidden_dim=24,
        num_layers=2,
        dropout=0.1,
        knn_k=4,
        epochs=20,
        lr=1e-3,
        weight_decay=1e-4,
        val_fraction=0.25,
        seed=123,
        device="cpu",
    )
    assert model_path.exists()
    assert weights_path.exists()
    assert float(out["val_metrics"]["rmse"]) >= 0.0

    scored_path = tmp_path / "qa_gnn_scored.parquet"
    scored = score_candidates_with_qa_gnn(
        candidates_path=train_path,
        model_path=model_path,
        weights_path=weights_path,
        out_scores_path=scored_path,
        device="cpu",
    )
    assert scored_path.exists()
    assert int(scored["rows"]) == 80
    df = pl.read_parquet(scored_path)
    assert "gnn_score" in df.columns
    assert df.get_column("gnn_score").is_not_null().all()


def test_train_qa_gnn_ranker_fails_without_required_column(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.parquet"
    pl.DataFrame(
        {
            "target_id": ["A", "A", "B", "B"],
            "coverage": [0.1, 0.2, 0.3, 0.4],
            "similarity": [0.2, 0.3, 0.4, 0.5],
            "label": [0.1, 0.2, 0.3, 0.4],
        }
    ).write_parquet(bad_path)
    with pytest.raises(PipelineError):
        train_qa_gnn_ranker(
            candidates_path=bad_path,
            out_model_path=tmp_path / "m.json",
            out_weights_path=tmp_path / "m.pt",
            feature_names=QA_GNN_DEFAULT_FEATURE_NAMES,
            label_col="label",
            group_col="target_id",
            epochs=5,
            device="cpu",
        )


def test_qa_gnn_runtime_scoring_and_model_type_detection(tmp_path: Path) -> None:
    train_path = tmp_path / "qa_gnn_train.parquet"
    _make_train_table(train_path)
    model_path = tmp_path / "qa_gnn_model.json"
    weights_path = tmp_path / "qa_gnn_model.pt"
    train_qa_gnn_ranker(
        candidates_path=train_path,
        out_model_path=model_path,
        out_weights_path=weights_path,
        feature_names=QA_GNN_DEFAULT_FEATURE_NAMES,
        label_col="label",
        group_col="target_id",
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        knn_k=3,
        epochs=10,
        lr=1e-3,
        weight_decay=1e-4,
        val_fraction=0.25,
        seed=123,
        device="cpu",
    )
    assert is_qa_gnn_model_file(model_path=model_path, location="tests:test_qa_gnn_runtime_scoring_and_model_type_detection")

    runtime = load_qa_gnn_runtime(
        model_path=model_path,
        weights_path=weights_path,
        device="cpu",
        location="tests:test_qa_gnn_runtime_scoring_and_model_type_detection",
    )
    feature_dicts = [
        {
            "coverage": 0.8,
            "similarity": 0.7,
            "path_length": 25.0,
            "step_mean": 1.2,
            "step_std": 0.35,
            "radius_gyr": 2.8,
            "gap_open_score": -5.0,
            "gap_extend_score": -1.0,
        },
        {
            "coverage": 0.6,
            "similarity": 0.5,
            "path_length": 22.0,
            "step_mean": 1.1,
            "step_std": 0.32,
            "radius_gyr": 2.4,
            "gap_open_score": -3.0,
            "gap_extend_score": -0.5,
        },
    ]
    scores = score_candidate_feature_dicts_with_qa_gnn_runtime(
        feature_dicts=feature_dicts,
        runtime=runtime,
        location="tests:test_qa_gnn_runtime_scoring_and_model_type_detection",
    )
    assert len(scores) == 2
    assert all(isinstance(v, float) for v in scores)

    linear_model_path = tmp_path / "linear_qa_model.json"
    linear_model_path.write_text("{\"version\": 1, \"feature_names\": [\"coverage\"]}\n", encoding="utf-8")
    assert not is_qa_gnn_model_file(
        model_path=linear_model_path,
        location="tests:test_qa_gnn_runtime_scoring_and_model_type_detection",
    )
