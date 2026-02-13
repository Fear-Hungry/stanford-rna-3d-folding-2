from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rna3d_local.qa_rnrank import QA_RNRANK_DEFAULT_FEATURE_NAMES, select_top5_global_with_qa_rnrank, train_qa_rnrank


def _require_torch_for_test() -> None:
    try:
        import torch  # noqa: PLC0415,F401
    except Exception:  # noqa: BLE001
        pytest.skip("PyTorch indisponivel para teste de selecao global")


def _make_candidate_pool(path: Path) -> None:
    rows: list[dict] = []
    for t in range(4):
        target_id = f"T{t:02d}"
        seq = "ACGUA"
        for m in range(6):
            coords = [[float(i + m), float(t), 0.0] for i in range(1, len(seq) + 1)]
            coverage = 1.0
            similarity = 0.4 + (0.08 * m)
            step_std = 0.2 + (0.02 * (m % 3))
            rows.append(
                {
                    "target_id": target_id,
                    "source": "tbm",
                    "model_id": m + 1,
                    "candidate_id": f"tbm:model_{m+1}",
                    "template_uid": f"tbm:{target_id}:model_{m+1}",
                    "resid_count": len(seq),
                    "resids": [1, 2, 3, 4, 5],
                    "resnames": list(seq),
                    "sequence": seq,
                    "coords": coords,
                    "coverage": coverage,
                    "similarity": similarity,
                    "mapped_count": len(seq),
                    "match_count": int(round(similarity * len(seq))),
                    "mismatch_count": int(len(seq) - int(round(similarity * len(seq)))),
                    "chem_compatible_count": 0,
                    "mapped_ratio": 1.0,
                    "match_ratio": similarity,
                    "mismatch_ratio": max(0.0, 1.0 - similarity),
                    "chem_compatible_ratio": 0.0,
                    "path_length": 10.0 + (0.5 * m),
                    "step_mean": 2.0 + (0.03 * m),
                    "step_std": step_std,
                    "radius_gyr": 1.5 + (0.05 * m),
                    "dist_off_1": 2.0 + (0.02 * m),
                    "dist_off_2": 2.2 + (0.02 * m),
                    "dist_off_4": 2.8 + (0.01 * m),
                    "dist_off_8": 0.0,
                    "dist_off_16": 0.0,
                    "dist_off_32": 0.0,
                    "gap_open_score": -5.0,
                    "gap_extend_score": -1.0,
                    "qa_score_base": 0.1 * float(m),
                    "label": (0.6 * similarity) + (0.3 * coverage) - (0.1 * step_std),
                }
            )
    pl.DataFrame(rows).write_parquet(path)


def test_select_top5_global_with_qa_rnrank(tmp_path: Path) -> None:
    _require_torch_for_test()
    candidates = tmp_path / "candidate_pool.parquet"
    _make_candidate_pool(candidates)

    model_path = tmp_path / "qa_rnrank_model.json"
    weights_path = tmp_path / "qa_rnrank_model.pt"
    train_qa_rnrank(
        candidates_path=candidates,
        out_model_path=model_path,
        out_weights_path=weights_path,
        feature_names=QA_RNRANK_DEFAULT_FEATURE_NAMES,
        label_col="label",
        group_col="target_id",
        hidden_dim=32,
        dropout=0.1,
        epochs=16,
        lr=1e-3,
        weight_decay=1e-4,
        val_fraction=0.25,
        rank_weight=0.4,
        regression_weight=0.6,
        seed=123,
        device="cpu",
    )

    out_long = tmp_path / "selected_long.parquet"
    out_path, manifest_path = select_top5_global_with_qa_rnrank(
        candidates_path=candidates,
        model_path=model_path,
        weights_path=weights_path,
        out_predictions_path=out_long,
        n_models=5,
        qa_top_pool=6,
        diversity_lambda=0.1,
        device="cpu",
    )
    assert out_path.exists()
    assert manifest_path.exists()
    df = pl.read_parquet(out_path)
    # 4 targets * 5 models * 5 residues
    assert df.height == 100
    per_target_models = (
        df.group_by("target_id")
        .agg(pl.col("model_id").n_unique().alias("n_models"))
        .sort("target_id")
        .get_column("n_models")
        .to_list()
    )
    assert per_target_models == [5, 5, 5, 5]
