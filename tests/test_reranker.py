from __future__ import annotations

from pathlib import Path

import polars as pl

from rna3d_local.reranker import score_template_reranker, train_template_reranker


def test_train_and_score_reranker(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.parquet"
    rows = []
    for target_id in ["Q1", "Q2"]:
        for template_uid, label in [("ext:A", 1.0), ("ext:B", 0.0), ("ext:C", 1.0), ("ext:D", 0.0)]:
            rows.append(
                {
                    "target_id": target_id,
                    "template_uid": template_uid,
                    "cosine_score": 0.9 if label > 0 else 0.2,
                    "family_prior_score": 0.8 if label > 0 else 0.1,
                    "alignment_refine_score": 0.7 if label > 0 else 0.2,
                    "label": label,
                }
            )
    pl.DataFrame(rows).write_parquet(candidates)

    chem = tmp_path / "chemical.parquet"
    pl.DataFrame(
        [
            {"target_id": "Q1", "resid": 1, "p_open": 0.0, "p_paired": 1.0},
            {"target_id": "Q1", "resid": 2, "p_open": 1.0, "p_paired": 0.0},
            {"target_id": "Q1", "resid": 3, "p_open": 0.0, "p_paired": 1.0},
            {"target_id": "Q2", "resid": 1, "p_open": 1.0, "p_paired": 0.0},
            {"target_id": "Q2", "resid": 2, "p_open": 0.0, "p_paired": 1.0},
            {"target_id": "Q2", "resid": 3, "p_open": 1.0, "p_paired": 0.0},
        ]
    ).write_parquet(chem)
    templates = tmp_path / "templates.parquet"
    rows_tpl = []
    template_coords = {
        "ext:A": [(-2.0, 0.0, 0.0), (0.0, 5.0, 0.0), (2.0, 0.0, 0.0)],
        "ext:B": [(-2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
        "ext:C": [(-2.0, 0.0, 0.0), (0.0, 5.0, 0.0), (2.0, 0.0, 0.0)],
        "ext:D": [(-2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
    }
    for template_uid, coords in template_coords.items():
        for resid, (x, y, z) in enumerate(coords, start=1):
            rows_tpl.append(
                {
                    "template_uid": template_uid,
                    "resid": resid,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
    pl.DataFrame(rows_tpl).write_parquet(templates)

    trained = train_template_reranker(
        repo_root=tmp_path,
        candidates_path=candidates,
        chemical_features_path=chem,
        templates_path=templates,
        out_dir=tmp_path / "model",
        labels_path=None,
        epochs=20,
        learning_rate=1e-2,
        seed=42,
    )
    scored_path = tmp_path / "scored.parquet"
    scored = score_template_reranker(
        repo_root=tmp_path,
        candidates_path=candidates,
        chemical_features_path=chem,
        templates_path=templates,
        model_path=trained.model_path,
        config_path=trained.config_path,
        out_path=scored_path,
        top_k=2,
    )
    assert scored.scored_path.exists()
    out = pl.read_parquet(scored.scored_path)
    assert out.height == 4
    assert out.filter(pl.col("rerank_rank") <= 2).height == 4

    scored_full = score_template_reranker(
        repo_root=tmp_path,
        candidates_path=candidates,
        chemical_features_path=chem,
        templates_path=templates,
        model_path=trained.model_path,
        config_path=trained.config_path,
        out_path=tmp_path / "scored_full.parquet",
        top_k=None,
    )
    out_full = pl.read_parquet(scored_full.scored_path)
    q1 = out_full.filter(pl.col("target_id") == "Q1")
    assert int(q1.get_column("chem_p_open_mean").n_unique()) > 1
    assert int(q1.get_column("chem_p_paired_mean").n_unique()) > 1
