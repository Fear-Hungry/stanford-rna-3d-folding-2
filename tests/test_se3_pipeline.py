from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.ensemble.qa_ranker_se3 import rank_se3_ensemble
from rna3d_local.ensemble.select_top5 import select_top5_se3
from rna3d_local.errors import PipelineError
from rna3d_local.se3_pipeline import sample_se3_ensemble, train_se3_generator


def _write_targets(path: Path) -> None:
    pl.DataFrame(
        [
            {"target_id": "T1", "sequence": "ACG", "temporal_cutoff": "2024-01-01"},
            {"target_id": "T2", "sequence": "GUA", "temporal_cutoff": "2024-01-01"},
        ]
    ).write_csv(path)


def _write_pairings(path: Path) -> None:
    rows = []
    for target_id in ["T1", "T2"]:
        for resid in [1, 2, 3]:
            rows.append({"target_id": target_id, "resid": resid, "pair_prob": 0.2 * resid})
    pl.DataFrame(rows).write_parquet(path)


def _write_chem(path: Path) -> None:
    rows = []
    for target_id in ["T1", "T2"]:
        for resid in [1, 2, 3]:
            rows.append({"target_id": target_id, "resid": resid, "p_open": 0.3 + (0.1 * resid), "p_paired": 0.7 - (0.1 * resid)})
    pl.DataFrame(rows).write_parquet(path)


def _write_labels(path: Path) -> None:
    rows = []
    for target_id in ["T1", "T2"]:
        for resid in [1, 2, 3]:
            rows.append({"target_id": target_id, "resid": resid, "x": float(resid), "y": float(resid + 1), "z": float(resid + 2)})
    pl.DataFrame(rows).write_parquet(path)


def _write_config(path: Path) -> None:
    payload = {
        "hidden_dim": 16,
        "num_layers": 1,
        "ipa_heads": 4,
        "diffusion_steps": 6,
        "flow_steps": 6,
        "epochs": 2,
        "learning_rate": 1e-3,
        "method": "both",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_train_sample_rank_select_se3_pipeline(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    pairings = tmp_path / "pairings.parquet"
    chem = tmp_path / "chem.parquet"
    labels = tmp_path / "labels.parquet"
    config = tmp_path / "config.json"
    _write_targets(targets)
    _write_pairings(pairings)
    _write_chem(chem)
    _write_labels(labels)
    _write_config(config)

    trained = train_se3_generator(
        repo_root=tmp_path,
        targets_path=targets,
        pairings_path=pairings,
        chemical_features_path=chem,
        labels_path=labels,
        config_path=config,
        out_dir=tmp_path / "model",
        seed=123,
    )
    sampled = sample_se3_ensemble(
        repo_root=tmp_path,
        model_dir=trained.model_dir,
        targets_path=targets,
        pairings_path=pairings,
        chemical_features_path=chem,
        out_path=tmp_path / "candidates.parquet",
        method="both",
        n_samples=6,
        seed=123,
    )
    ranked = rank_se3_ensemble(
        repo_root=tmp_path,
        candidates_path=sampled.candidates_path,
        out_path=tmp_path / "ranked.parquet",
        qa_config_path=None,
        diversity_lambda=0.2,
    )
    selected = select_top5_se3(
        repo_root=tmp_path,
        ranked_path=ranked.ranked_path,
        out_path=tmp_path / "top5.parquet",
        n_models=5,
        diversity_lambda=0.2,
    )
    out = pl.read_parquet(selected.predictions_path)
    assert out.get_column("target_id").n_unique() == 2
    per_target = out.group_by("target_id").agg(pl.col("model_id").n_unique().alias("n_models"))
    assert per_target.filter(pl.col("n_models") != 5).height == 0


def test_select_top5_se3_fails_when_insufficient_samples(tmp_path: Path) -> None:
    ranked = tmp_path / "ranked.parquet"
    rows = []
    for sample_id in ["s1", "s2", "s3"]:
        for resid, base in enumerate("ACG", start=1):
            rows.append(
                {
                    "target_id": "T1",
                    "sample_id": sample_id,
                    "resid": resid,
                    "resname": base,
                    "x": float(resid),
                    "y": float(resid + 1),
                    "z": float(resid + 2),
                    "final_score": 1.0,
                    "qa_score": 0.8,
                }
            )
    pl.DataFrame(rows).write_parquet(ranked)
    with pytest.raises(PipelineError, match="samples insuficientes"):
        select_top5_se3(
            repo_root=tmp_path,
            ranked_path=ranked,
            out_path=tmp_path / "top5.parquet",
            n_models=5,
            diversity_lambda=0.2,
        )
