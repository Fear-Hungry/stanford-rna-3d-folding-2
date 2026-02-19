from __future__ import annotations

import polars as pl
import pytest
import torch

from rna3d_local.errors import PipelineError
from rna3d_local.training.chemical_mapping import compute_chemical_exposure_mapping


def _targets() -> pl.DataFrame:
    return pl.DataFrame([{"target_id": "T1", "sequence": "AC|GU", "temporal_cutoff": "2024-01-01"}])


def _chemical_complete() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"target_id": "T1", "resid": 1, "reactivity_dms": 0.1, "reactivity_2a3": 0.2},
            {"target_id": "T1", "resid": 2, "reactivity_dms": 0.2, "reactivity_2a3": 0.3},
            {"target_id": "T1", "resid": 3, "reactivity_dms": 0.8, "reactivity_2a3": 0.9},
            {"target_id": "T1", "resid": 4, "reactivity_dms": 0.9, "reactivity_2a3": 1.0},
        ]
    )


def _labels_complete() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"target_id": "T1", "resid": 1, "x": 1.0, "y": 0.0, "z": 0.0},
            {"target_id": "T1", "resid": 2, "x": 2.0, "y": 0.0, "z": 0.0},
            {"target_id": "T1", "resid": 3, "x": 4.0, "y": 0.0, "z": 0.0},
            {"target_id": "T1", "resid": 4, "x": 7.0, "y": 0.0, "z": 0.0},
        ]
    )


def test_compute_chemical_mapping_ignores_pdb_geometry_for_exposure() -> None:
    out_with_pdb = compute_chemical_exposure_mapping(
        targets=_targets(),
        chemical_features=_chemical_complete(),
        pdb_labels=_labels_complete(),
        chain_separator="|",
        stage="TEST",
        location="tests/test_chemical_mapping.py:test_compute_chemical_mapping_ignores_pdb_geometry_for_exposure",
    )
    out_without_pdb = compute_chemical_exposure_mapping(
        targets=_targets(),
        chemical_features=_chemical_complete(),
        pdb_labels=None,
        chain_separator="|",
        stage="TEST",
        location="tests/test_chemical_mapping.py:test_compute_chemical_mapping_ignores_pdb_geometry_for_exposure",
    )
    target = out_with_pdb["T1"]
    assert target.source == "quickstart_only"
    assert int(target.exposure.numel()) == 4
    assert float(target.exposure.min().item()) >= 0.0
    assert float(target.exposure.max().item()) <= 1.0
    assert bool(torch.allclose(target.exposure, out_without_pdb["T1"].exposure))


def test_compute_chemical_mapping_without_pdb_uses_quickstart_only() -> None:
    out = compute_chemical_exposure_mapping(
        targets=_targets(),
        chemical_features=_chemical_complete(),
        pdb_labels=None,
        chain_separator="|",
        stage="TEST",
        location="tests/test_chemical_mapping.py:test_compute_chemical_mapping_without_pdb_uses_quickstart_only",
    )
    assert out["T1"].source == "quickstart_only"


def test_compute_chemical_mapping_fails_on_incomplete_coverage() -> None:
    bad_chemical = _chemical_complete().filter(pl.col("resid") != 4)
    with pytest.raises(PipelineError, match="cobertura completa"):
        compute_chemical_exposure_mapping(
            targets=_targets(),
            chemical_features=bad_chemical,
            pdb_labels=_labels_complete(),
            chain_separator="|",
            stage="TEST",
            location="tests/test_chemical_mapping.py:test_compute_chemical_mapping_fails_on_incomplete_coverage",
        )
