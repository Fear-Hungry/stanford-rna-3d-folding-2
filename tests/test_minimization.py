from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local import minimization
from rna3d_local.minimization import minimize_ensemble


def _write_predictions(path: Path) -> None:
    rows = []
    for target_id in ["T1", "T2"]:
        for model_id in [1, 2]:
            for resid, base in enumerate("ACGU", start=1):
                rows.append(
                    {
                        "target_id": target_id,
                        "model_id": model_id,
                        "resid": resid,
                        "resname": base,
                        "x": float(model_id * resid),
                        "y": float((model_id * resid) + 1),
                        "z": float((model_id * resid) + 2),
                        "source": "test",
                        "confidence": 0.8,
                    }
                )
    pl.DataFrame(rows).write_parquet(path)


def test_minimize_ensemble_openmm_preserves_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_minimize_openmm(
        *,
        coords_angstrom: np.ndarray,
        residue_index: np.ndarray,
        max_iterations: int,
        bond_length_angstrom: float,
        bond_force_k: float,
        angle_force_k: float,
        angle_target_deg: float,
        vdw_min_distance_angstrom: float,
        vdw_epsilon: float,
        position_restraint_k: float,
        openmm_platform: str | None,
        stage: str,
        location: str,
    ) -> np.ndarray:
        return np.asarray(coords_angstrom, dtype=np.float64)

    monkeypatch.setattr(minimization, "_minimize_openmm", _fake_minimize_openmm)
    pred = tmp_path / "pred.parquet"
    out = tmp_path / "pred_min.parquet"
    _write_predictions(pred)
    result = minimize_ensemble(
        repo_root=tmp_path,
        predictions_path=pred,
        out_path=out,
        backend="openmm",
        max_iterations=20,
        bond_length_angstrom=5.9,
        bond_force_k=60.0,
        angle_force_k=8.0,
        angle_target_deg=120.0,
        vdw_min_distance_angstrom=2.1,
        vdw_epsilon=0.2,
        position_restraint_k=800.0,
        openmm_platform=None,
    )
    refined = pl.read_parquet(result.predictions_path)
    assert refined.height == 16
    assert refined.get_column("target_id").n_unique() == 2
    assert set(["target_id", "model_id", "resid", "resname", "x", "y", "z", "refinement_backend", "refinement_steps", "refinement_position_restraint_k"]).issubset(set(refined.columns))
    assert refined.filter(pl.col("refinement_backend") != "openmm").height == 0
    key_dup = refined.group_by(["target_id", "model_id", "resid"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    assert key_dup.height == 0


def test_minimize_ensemble_fails_on_duplicate_key(tmp_path: Path) -> None:
    pred = tmp_path / "pred_dup.parquet"
    rows = [
        {"target_id": "T1", "model_id": 1, "resid": 1, "resname": "A", "x": 0.0, "y": 0.0, "z": 0.0},
        {"target_id": "T1", "model_id": 1, "resid": 1, "resname": "A", "x": 0.1, "y": 0.1, "z": 0.1},
    ]
    pl.DataFrame(rows).write_parquet(pred)
    with pytest.raises(PipelineError, match="chave duplicada"):
        minimize_ensemble(
            repo_root=tmp_path,
            predictions_path=pred,
            out_path=tmp_path / "out.parquet",
            backend="openmm",
            max_iterations=5,
            bond_length_angstrom=5.9,
            bond_force_k=60.0,
            angle_force_k=8.0,
            angle_target_deg=120.0,
            vdw_min_distance_angstrom=2.1,
            vdw_epsilon=0.2,
            position_restraint_k=800.0,
            openmm_platform=None,
        )


def test_minimize_ensemble_pyrosetta_backend_fails_actionable(tmp_path: Path) -> None:
    pred = tmp_path / "pred.parquet"
    _write_predictions(pred)
    with pytest.raises(PipelineError, match="pyrosetta"):
        minimize_ensemble(
            repo_root=tmp_path,
            predictions_path=pred,
            out_path=tmp_path / "out.parquet",
            backend="pyrosetta",
            max_iterations=5,
            bond_length_angstrom=5.9,
            bond_force_k=60.0,
            angle_force_k=8.0,
            angle_target_deg=120.0,
            vdw_min_distance_angstrom=2.1,
            vdw_epsilon=0.2,
            position_restraint_k=800.0,
            openmm_platform=None,
        )


def test_minimize_ensemble_fails_when_iterations_exceed_budget(tmp_path: Path) -> None:
    pred = tmp_path / "pred.parquet"
    _write_predictions(pred)
    with pytest.raises(PipelineError, match="max_iterations deve ser <= 100"):
        minimize_ensemble(
            repo_root=tmp_path,
            predictions_path=pred,
            out_path=tmp_path / "out.parquet",
            backend="openmm",
            max_iterations=120,
            bond_length_angstrom=5.9,
            bond_force_k=60.0,
            angle_force_k=8.0,
            angle_target_deg=120.0,
            vdw_min_distance_angstrom=2.1,
            vdw_epsilon=0.2,
            position_restraint_k=800.0,
            openmm_platform=None,
        )
