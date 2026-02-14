from __future__ import annotations

from pathlib import Path

import polars as pl

from rna3d_local.qa_ranker import (
    QA_FEATURE_NAMES,
    _pair_similarity,
    build_candidate_feature_dict,
    load_qa_model,
    score_candidate_with_model,
    select_candidates_with_diversity,
    train_qa_ranker,
)


def test_train_and_score_qa_ranker(tmp_path: Path) -> None:
    train_path = tmp_path / "qa_train.parquet"
    rows = []
    for i in range(24):
        cov = 0.2 + (0.03 * i)
        sim = 0.1 + (0.02 * i)
        rows.append(
            {
                "coverage": cov,
                "similarity": sim,
                "match_ratio": cov * 0.7,
                "mismatch_ratio": max(0.0, cov * 0.2),
                "chem_compatible_ratio": max(0.0, cov * 0.1),
                "path_length": 10.0 + i,
                "step_mean": 1.0 + (0.01 * i),
                "step_std": 0.2 + (0.005 * i),
                "radius_gyr": 2.0 + (0.02 * i),
                "gap_open_score": -5.0,
                "gap_extend_score": -1.0,
                "label": (0.6 * cov) + (0.4 * sim),
                "target_id": f"T{i % 6}",
            }
        )
    pl.DataFrame(rows).write_parquet(train_path)
    out_model = tmp_path / "qa_model.json"

    res = train_qa_ranker(
        candidates_path=train_path,
        out_model_path=out_model,
        label_col="label",
        group_col="target_id",
        feature_names=QA_FEATURE_NAMES,
        l2_lambda=1.0,
        val_fraction=0.25,
        seed=123,
    )
    assert out_model.exists()
    assert float(res["val_metrics"]["rmse"]) >= 0.0

    model = load_qa_model(model_path=out_model, location="tests/test_qa_ranker.py:test_train_and_score_qa_ranker")
    cand = {
        "coverage": 0.8,
        "similarity": 0.7,
        "match_count": 8,
        "mismatch_count": 2,
        "chem_compatible_count": 1,
        "gap_open_score": -5.0,
        "gap_extend_score": -1.0,
        "coords": [(float(i), 0.0, 0.0) for i in range(10)],
    }
    feat = build_candidate_feature_dict(candidate=cand, target_length=10, location="tests/test_qa_ranker.py:test_train_and_score_qa_ranker")
    score = score_candidate_with_model(candidate_features=feat, model=model, location="tests/test_qa_ranker.py:test_train_and_score_qa_ranker")
    assert isinstance(score, float)


def test_select_candidates_with_diversity_penalizes_duplicates() -> None:
    c1 = {
        "uid": "A",
        "qa_score": 0.95,
        "coords": [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)],
    }
    c2 = {
        "uid": "B",
        "qa_score": 0.94,
        "coords": [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)],
    }
    c3 = {
        "uid": "C",
        "qa_score": 0.80,
        "coords": [(0.0, 0.0, 0.0), (0.0, 10.0, 0.0), (20.0, 20.0, 0.0)],
    }
    selected = select_candidates_with_diversity(
        candidates=[c1, c2, c3],
        n_models=2,
        diversity_lambda=0.5,
        location="tests/test_qa_ranker.py:test_select_candidates_with_diversity_penalizes_duplicates",
    )
    uids = [str(c["uid"]) for c in selected]
    assert "A" in uids
    assert "C" in uids


def test_pair_similarity_is_rotation_invariant() -> None:
    a = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)]
    b = [(0.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 20.0, 0.0)]
    sim = _pair_similarity(
        coords_a=a,
        coords_b=b,
        location="tests/test_qa_ranker.py:test_pair_similarity_is_rotation_invariant",
    )
    assert sim > 0.99
