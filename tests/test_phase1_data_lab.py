from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.training.data_lab import prepare_phase1_data_lab


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    targets = tmp_path / "targets.csv"
    pairings = tmp_path / "pairings.parquet"
    chemical = tmp_path / "chemical.parquet"
    labels = tmp_path / "labels.parquet"
    pl.DataFrame([{"target_id": "T1", "sequence": "ACGU", "temporal_cutoff": "2024-01-01"}]).write_csv(targets)
    pl.DataFrame(
        [
            {"target_id": "T1", "resid": 1, "pair_prob": 0.1},
            {"target_id": "T1", "resid": 2, "pair_prob": 0.2},
            {"target_id": "T1", "resid": 3, "pair_prob": 0.3},
            {"target_id": "T1", "resid": 4, "pair_prob": 0.4},
        ]
    ).write_parquet(pairings)
    pl.DataFrame(
        [
            {"target_id": "T1", "resid": 1, "reactivity_dms": 0.1, "reactivity_2a3": 0.2, "p_open": 0.5, "p_paired": 0.5},
            {"target_id": "T1", "resid": 2, "reactivity_dms": 0.2, "reactivity_2a3": 0.3, "p_open": 0.5, "p_paired": 0.5},
            {"target_id": "T1", "resid": 3, "reactivity_dms": 0.3, "reactivity_2a3": 0.4, "p_open": 0.5, "p_paired": 0.5},
            {"target_id": "T1", "resid": 4, "reactivity_dms": 0.4, "reactivity_2a3": 0.5, "p_open": 0.5, "p_paired": 0.5},
        ]
    ).write_parquet(chemical)
    pl.DataFrame(
        [
            {"target_id": "T1", "resid": 1, "x": 1.0, "y": 2.0, "z": 3.0},
            {"target_id": "T1", "resid": 2, "x": 2.0, "y": 3.0, "z": 4.0},
            {"target_id": "T1", "resid": 3, "x": 3.0, "y": 4.0, "z": 5.0},
            {"target_id": "T1", "resid": 4, "x": 4.0, "y": 5.0, "z": 6.0},
        ]
    ).write_parquet(labels)
    return targets, pairings, chemical, labels


def test_prepare_phase1_data_lab_contract(tmp_path: Path) -> None:
    targets, pairings, chemical, labels = _write_inputs(tmp_path)
    out_dir = tmp_path / "phase1_lab"
    if importlib.util.find_spec("zarr") is None:
        with pytest.raises(PipelineError, match="dependencia zarr ausente"):
            prepare_phase1_data_lab(
                repo_root=tmp_path,
                targets_path=targets,
                pairings_path=pairings,
                chemical_features_path=chemical,
                labels_path=labels,
                out_dir=out_dir,
                thermo_backend="mock",
                rnafold_bin="RNAfold",
                linearfold_bin="linearfold",
                msa_backend="mock",
                mmseqs_bin="mmseqs",
                mmseqs_db="",
                chain_separator="|",
                max_msa_sequences=16,
                max_cov_positions=64,
                max_cov_pairs=512,
                num_workers=4,
            )
    else:
        result = prepare_phase1_data_lab(
            repo_root=tmp_path,
            targets_path=targets,
            pairings_path=pairings,
            chemical_features_path=chemical,
            labels_path=labels,
            out_dir=out_dir,
            thermo_backend="mock",
            rnafold_bin="RNAfold",
            linearfold_bin="linearfold",
            msa_backend="mock",
            mmseqs_bin="mmseqs",
            mmseqs_db="",
            chain_separator="|",
            max_msa_sequences=16,
            max_cov_positions=64,
            max_cov_pairs=512,
            num_workers=4,
        )
        assert result.store_path.exists()
        assert result.store_manifest_path.exists()
        assert result.manifest_path.exists()
