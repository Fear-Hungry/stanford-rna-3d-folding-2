from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rna3d_local.ensemble import blend_predictions
from rna3d_local.errors import PipelineError


def _write_parquet(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_parquet(path)


def test_blend_predictions_dynamic_by_coverage_changes_weight_per_row(tmp_path: Path) -> None:
    tbm = tmp_path / "tbm.parquet"
    rnp = tmp_path / "rnp.parquet"
    out_static = tmp_path / "out_static.parquet"
    out_dynamic = tmp_path / "out_dynamic.parquet"

    _write_parquet(
        tbm,
        [
            {"ID": "Q1_1", "model_id": 1, "resid": 1, "resname": "A", "x": 10.0, "y": 0.0, "z": 0.0, "coverage": 0.9},
            {"ID": "Q2_1", "model_id": 1, "resid": 1, "resname": "A", "x": 10.0, "y": 0.0, "z": 0.0, "coverage": 0.1},
        ],
    )
    _write_parquet(
        rnp,
        [
            {"ID": "Q1_1", "model_id": 1, "resid": 1, "resname": "A", "x": 0.0, "y": 0.0, "z": 0.0, "coverage": 0.1},
            {"ID": "Q2_1", "model_id": 1, "resid": 1, "resname": "A", "x": 0.0, "y": 0.0, "z": 0.0, "coverage": 0.9},
        ],
    )

    blend_predictions(
        tbm_predictions_path=tbm,
        rnapro_predictions_path=rnp,
        out_path=out_static,
        tbm_weight=1.0,
        rnapro_weight=1.0,
        dynamic_by_coverage=False,
    )
    blend_predictions(
        tbm_predictions_path=tbm,
        rnapro_predictions_path=rnp,
        out_path=out_dynamic,
        tbm_weight=1.0,
        rnapro_weight=1.0,
        dynamic_by_coverage=True,
        coverage_power=1.0,
    )

    static_df = pl.read_parquet(out_static).sort("ID")
    dynamic_df = pl.read_parquet(out_dynamic).sort("ID")
    assert static_df.get_column("x").to_list() == [5.0, 5.0]
    assert dynamic_df.get_column("x").to_list() == [9.0, 1.0]


def test_blend_predictions_dynamic_rejects_invalid_coverage(tmp_path: Path) -> None:
    tbm = tmp_path / "tbm_bad.parquet"
    rnp = tmp_path / "rnp_bad.parquet"
    out = tmp_path / "out.parquet"

    _write_parquet(
        tbm,
        [{"ID": "Q1_1", "model_id": 1, "resid": 1, "resname": "A", "x": 1.0, "y": 2.0, "z": 3.0, "coverage": -0.1}],
    )
    _write_parquet(
        rnp,
        [{"ID": "Q1_1", "model_id": 1, "resid": 1, "resname": "A", "x": 1.0, "y": 2.0, "z": 3.0, "coverage": 0.4}],
    )

    with pytest.raises(PipelineError):
        blend_predictions(
            tbm_predictions_path=tbm,
            rnapro_predictions_path=rnp,
            out_path=out,
            dynamic_by_coverage=True,
        )

