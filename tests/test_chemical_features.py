from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rna3d_local.chemical_features import prepare_chemical_features
from rna3d_local.errors import PipelineError


def test_prepare_chemical_features_from_reactivity_schema(tmp_path: Path) -> None:
    quickstart = tmp_path / "quickstart.csv"
    pl.DataFrame(
        [
            {"target_id": "T1", "resid": 1, "dms": 0.1, "2a3": 0.6},
            {"target_id": "T1", "resid": 2, "dms": 0.8, "2a3": 0.2},
            {"target_id": "T2", "resid": 1, "dms": 0.3, "2a3": 0.7},
            {"target_id": "T2", "resid": 2, "dms": 0.5, "2a3": 0.4},
        ]
    ).write_csv(quickstart)
    out = tmp_path / "chemical.parquet"

    result = prepare_chemical_features(repo_root=tmp_path, quickstart_path=quickstart, out_path=out)
    assert result.features_path.exists()
    features = pl.read_parquet(result.features_path)
    assert features.height == 4
    assert set(features.columns) == {"target_id", "resid", "reactivity_dms", "reactivity_2a3", "p_open", "p_paired"}


def test_prepare_chemical_features_from_template_quickstart_schema(tmp_path: Path) -> None:
    quickstart = tmp_path / "quickstart_templates.csv"
    pl.DataFrame(
        [
            {"ID": "R1:1", "resname": "A", "resid": 1, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0, "x_2": 1.0, "y_2": 0.0, "z_2": 0.0},
            {"ID": "R1:2", "resname": "C", "resid": 2, "x_1": 0.0, "y_1": 1.0, "z_1": 0.0, "x_2": 1.0, "y_2": 1.0, "z_2": 0.0},
            {"ID": "R2:1", "resname": "G", "resid": 1, "x_1": 2.0, "y_1": 0.0, "z_1": 0.0, "x_2": 3.0, "y_2": 0.0, "z_2": 0.0},
            {"ID": "R2:2", "resname": "U", "resid": 2, "x_1": 2.0, "y_1": 1.0, "z_1": 0.0, "x_2": 3.0, "y_2": 1.0, "z_2": 0.0},
        ]
    ).write_csv(quickstart)
    out = tmp_path / "chemical.parquet"

    result = prepare_chemical_features(repo_root=tmp_path, quickstart_path=quickstart, out_path=out)
    assert result.features_path.exists()
    features = pl.read_parquet(result.features_path)
    assert features.height == 4
    assert features.get_column("target_id").n_unique() == 2
    assert (features.get_column("p_open") >= 0.0).all()
    assert (features.get_column("p_open") <= 1.0).all()


def test_prepare_chemical_features_from_template_quickstart_single_triplet(tmp_path: Path) -> None:
    quickstart = tmp_path / "quickstart_templates_single.csv"
    pl.DataFrame(
        [
            {"ID": "R1:1", "resname": "A", "resid": 1, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0},
            {"ID": "R1:2", "resname": "C", "resid": 2, "x_1": 2.0, "y_1": 0.0, "z_1": 0.0},
            {"ID": "R2:1", "resname": "G", "resid": 1, "x_1": 1.0, "y_1": 1.0, "z_1": 0.0},
            {"ID": "R2:2", "resname": "U", "resid": 2, "x_1": 3.0, "y_1": 1.0, "z_1": 0.0},
        ]
    ).write_csv(quickstart)
    out = tmp_path / "chemical.parquet"

    result = prepare_chemical_features(repo_root=tmp_path, quickstart_path=quickstart, out_path=out)
    features = pl.read_parquet(result.features_path)
    assert features.height == 4
    assert features.filter(pl.col("reactivity_dms").is_null() | pl.col("reactivity_2a3").is_null()).height == 0


def test_prepare_chemical_features_from_template_quickstart_plain_xyz_and_underscore_id(tmp_path: Path) -> None:
    quickstart = tmp_path / "quickstart_templates_plain.csv"
    pl.DataFrame(
        [
            {"ID": "R1_1", "resname": "A", "resid": 1, "x": 0.0, "y": 0.0, "z": 0.0},
            {"ID": "R1_2", "resname": "C", "resid": 2, "x": 1.0, "y": 0.0, "z": 0.0},
            {"ID": "R2_1", "resname": "G", "resid": 1, "x": 2.0, "y": 0.0, "z": 0.0},
            {"ID": "R2_2", "resname": "U", "resid": 2, "x": 3.0, "y": 0.0, "z": 0.0},
        ]
    ).write_csv(quickstart)
    out = tmp_path / "chemical.parquet"

    result = prepare_chemical_features(repo_root=tmp_path, quickstart_path=quickstart, out_path=out)
    features = pl.read_parquet(result.features_path)
    assert features.height == 4
    assert set(features.get_column("target_id").unique().to_list()) == {"R1", "R2"}
    assert (features.get_column("p_open") >= 0.0).all()
    assert (features.get_column("p_open") <= 1.0).all()


def test_prepare_chemical_features_fails_for_unsupported_schema(tmp_path: Path) -> None:
    quickstart = tmp_path / "invalid.csv"
    pl.DataFrame([{"foo": "x", "bar": 1}]).write_csv(quickstart)
    out = tmp_path / "chemical.parquet"

    with pytest.raises(PipelineError, match="schema quickstart nao suportado"):
        prepare_chemical_features(repo_root=tmp_path, quickstart_path=quickstart, out_path=out)
