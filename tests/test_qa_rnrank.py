from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rna3d_local.qa_rnrank import (
    QA_RNRANK_DEFAULT_FEATURE_NAMES,
    is_qa_rnrank_model_file,
    score_candidates_with_qa_rnrank,
    train_qa_rnrank,
)


def _require_torch_for_test() -> None:
    try:
        import torch  # noqa: PLC0415,F401
    except Exception:  # noqa: BLE001
        pytest.skip("PyTorch indisponivel para teste QA RNArank")


def _make_train_table(path: Path) -> None:
    rows: list[dict] = []
    for t in range(10):
        target_id = f"T{t:02d}"
        for m in range(8):
            coverage = 0.2 + (0.08 * m)
            similarity = 0.1 + (0.07 * m)
            path_length = 18.0 + (1.5 * m) + (0.2 * t)
            step_mean = 1.0 + (0.015 * m)
            step_std = 0.25 + (0.01 * (m % 4))
            radius_gyr = 2.0 + (0.05 * m)
            rows.append(
                {
                    "target_id": target_id,
                    "candidate_id": f"tbm:model_{m+1}",
                    "coverage": coverage,
                    "similarity": similarity,
                    "mapped_ratio": min(1.0, coverage),
                    "match_ratio": min(1.0, similarity * 0.8),
                    "mismatch_ratio": max(0.0, coverage - (similarity * 0.8)),
                    "chem_compatible_ratio": 0.05 * (m % 3),
                    "path_length": path_length,
                    "step_mean": step_mean,
                    "step_std": step_std,
                    "radius_gyr": radius_gyr,
                    "gap_open_score": -5.0 if (m % 2 == 0) else -3.0,
                    "gap_extend_score": -1.0 if (m % 3 == 0) else -0.5,
                    "qa_score_base": 0.05 * float(m),
                    "resid_count": 40.0,
                    "dist_off_1": 2.0 + (0.03 * m),
                    "dist_off_2": 2.4 + (0.03 * m),
                    "dist_off_4": 3.0 + (0.02 * m),
                    "dist_off_8": 4.0 + (0.02 * m),
                    "dist_off_16": 5.0 + (0.01 * m),
                    "dist_off_32": 6.0 + (0.01 * m),
                    "label": (0.55 * coverage) + (0.35 * similarity) - (0.05 * step_std) + (0.01 * radius_gyr),
                }
            )
    pl.DataFrame(rows).write_parquet(path)


def test_train_and_score_qa_rnrank(tmp_path: Path) -> None:
    _require_torch_for_test()
    train_path = tmp_path / "qa_rnrank_train.parquet"
    _make_train_table(train_path)

    model_path = tmp_path / "qa_rnrank_model.json"
    weights_path = tmp_path / "qa_rnrank_model.pt"
    out = train_qa_rnrank(
        candidates_path=train_path,
        out_model_path=model_path,
        out_weights_path=weights_path,
        feature_names=QA_RNRANK_DEFAULT_FEATURE_NAMES,
        label_col="label",
        group_col="target_id",
        hidden_dim=32,
        dropout=0.1,
        epochs=20,
        lr=1e-3,
        weight_decay=1e-4,
        val_fraction=0.2,
        rank_weight=0.4,
        regression_weight=0.6,
        seed=123,
        device="cpu",
    )
    assert model_path.exists()
    assert weights_path.exists()
    assert float(out["val_metrics"]["rmse"]) >= 0.0
    assert is_qa_rnrank_model_file(model_path=model_path, location="tests:test_train_and_score_qa_rnrank")

    scored_path = tmp_path / "qa_rnrank_scored.parquet"
    scored = score_candidates_with_qa_rnrank(
        candidates_path=train_path,
        model_path=model_path,
        weights_path=weights_path,
        out_scores_path=scored_path,
        device="cpu",
    )
    assert scored_path.exists()
    assert int(scored["rows"]) == 80
    df = pl.read_parquet(scored_path)
    assert "qa_rnrank_score" in df.columns
    assert df.get_column("qa_rnrank_score").is_not_null().all()
