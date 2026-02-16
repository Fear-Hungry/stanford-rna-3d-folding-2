from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.ensemble.qa_ranker_se3 import rank_se3_ensemble
from rna3d_local.ensemble.select_top5 import select_top5_se3
from rna3d_local.errors import PipelineError
from rna3d_local.se3_pipeline import sample_se3_ensemble, train_se3_generator
from rna3d_local.training import msa_covariance, thermo_2d


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
            rows.append(
                {
                    "target_id": target_id,
                    "resid": resid,
                    "reactivity_dms": 0.1 * resid,
                    "reactivity_2a3": 0.2 * resid,
                    "p_open": 0.3 + (0.1 * resid),
                    "p_paired": 0.7 - (0.1 * resid),
                }
            )
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
        "thermo_backend": "rnafold",
        "msa_backend": "mmseqs2",
        "mmseqs_db": "/tmp/fake_db",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _patch_external_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_rnafold_pairs(*, sequence: str, target_id: str, rnafold_bin: str, stage: str, location: str):
        length = len(sequence)
        pairs: list[tuple[int, int, float]] = []
        for left in range(1, (length // 2) + 1):
            right = length - left + 1
            if left < right:
                pairs.append((left, right, 0.60))
        return pairs

    def _fake_run_mmseqs_chain_alignments(
        *,
        mmseqs_bin: str,
        mmseqs_db: str,
        chain_sequence: str,
        query_id: str,
        max_msa_sequences: int,
        stage: str,
        location: str,
    ):
        import numpy as np

        mapping = {"A": 0, "C": 1, "G": 2, "U": 3}
        seq = chain_sequence.strip().upper().replace("T", "U")
        row0 = np.array([mapping[ch] for ch in seq], dtype=np.int16)
        row1 = row0.copy()
        if row0.size >= 2:
            canonical = [(0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
            left0 = int(row0[0])
            right0 = int(row0[-1])
            chosen = None
            for left, right in canonical:
                if left != left0 and right != right0:
                    chosen = (left, right)
                    break
            if chosen is None:
                chosen = ((left0 + 1) % 4, (right0 + 1) % 4)
            row1[0] = np.int16(chosen[0])
            row1[-1] = np.int16(chosen[1])
        return np.stack([row0, row1], axis=0)

    monkeypatch.setattr(thermo_2d, "_run_rnafold_pairs", _fake_run_rnafold_pairs)
    monkeypatch.setattr(msa_covariance, "_run_mmseqs_chain_alignments", _fake_run_mmseqs_chain_alignments)


def test_train_sample_rank_select_se3_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_external_backends(monkeypatch)
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


def test_train_sample_se3_with_multichain_sequence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_external_backends(monkeypatch)
    targets = tmp_path / "targets.csv"
    pairings = tmp_path / "pairings.parquet"
    chem = tmp_path / "chem.parquet"
    labels = tmp_path / "labels.parquet"
    config = tmp_path / "config.json"
    pl.DataFrame([{"target_id": "TC", "sequence": "AC|GU", "temporal_cutoff": "2024-01-01"}]).write_csv(targets)
    pl.DataFrame(
        [
            {"target_id": "TC", "resid": 1, "pair_prob": 0.1},
            {"target_id": "TC", "resid": 2, "pair_prob": 0.2},
            {"target_id": "TC", "resid": 3, "pair_prob": 0.3},
            {"target_id": "TC", "resid": 4, "pair_prob": 0.4},
        ]
    ).write_parquet(pairings)
    pl.DataFrame(
        [
            {"target_id": "TC", "resid": 1, "reactivity_dms": 0.1, "reactivity_2a3": 0.2, "p_open": 0.5, "p_paired": 0.5},
            {"target_id": "TC", "resid": 2, "reactivity_dms": 0.2, "reactivity_2a3": 0.3, "p_open": 0.5, "p_paired": 0.5},
            {"target_id": "TC", "resid": 3, "reactivity_dms": 0.3, "reactivity_2a3": 0.4, "p_open": 0.5, "p_paired": 0.5},
            {"target_id": "TC", "resid": 4, "reactivity_dms": 0.4, "reactivity_2a3": 0.5, "p_open": 0.5, "p_paired": 0.5},
        ]
    ).write_parquet(chem)
    pl.DataFrame(
        [
            {"target_id": "TC", "resid": 1, "x": 1.0, "y": 2.0, "z": 3.0},
            {"target_id": "TC", "resid": 2, "x": 2.0, "y": 3.0, "z": 4.0},
            {"target_id": "TC", "resid": 3, "x": 3.0, "y": 4.0, "z": 5.0},
            {"target_id": "TC", "resid": 4, "x": 4.0, "y": 5.0, "z": 6.0},
        ]
    ).write_parquet(labels)
    config.write_text(
        json.dumps(
            {
                "hidden_dim": 16,
                "num_layers": 1,
                "ipa_heads": 4,
                "diffusion_steps": 4,
                "flow_steps": 4,
                "epochs": 1,
                "learning_rate": 1e-3,
                "method": "diffusion",
                "thermo_backend": "rnafold",
                "msa_backend": "mmseqs2",
                "mmseqs_db": "/tmp/fake_db",
                "chain_separator": "|",
            }
        ),
        encoding="utf-8",
    )
    trained = train_se3_generator(
        repo_root=tmp_path,
        targets_path=targets,
        pairings_path=pairings,
        chemical_features_path=chem,
        labels_path=labels,
        config_path=config,
        out_dir=tmp_path / "model",
        seed=7,
    )
    sampled = sample_se3_ensemble(
        repo_root=tmp_path,
        model_dir=trained.model_dir,
        targets_path=targets,
        pairings_path=pairings,
        chemical_features_path=chem,
        out_path=tmp_path / "candidates.parquet",
        method="diffusion",
        n_samples=2,
        seed=7,
    )
    out = pl.read_parquet(sampled.candidates_path)
    assert out.height == 8
