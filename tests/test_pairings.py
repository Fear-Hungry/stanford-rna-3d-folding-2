from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.pairings import derive_pairings_from_chemical


def test_derive_pairings_from_chemical_success(tmp_path: Path) -> None:
    chem = tmp_path / "chem.parquet"
    out = tmp_path / "pairings.parquet"
    pl.DataFrame(
        [
            {"target_id": "T1", "resid": 1, "p_paired": 0.1},
            {"target_id": "T1", "resid": 2, "p_paired": 0.9},
            {"target_id": "T2", "resid": 1, "p_paired": 0.5},
        ]
    ).write_parquet(chem)
    result = derive_pairings_from_chemical(repo_root=tmp_path, chemical_features_path=chem, out_path=out)
    assert result.pairings_path.exists()
    df = pl.read_parquet(result.pairings_path)
    assert df.columns == ["target_id", "resid", "pair_prob"]
    assert df.height == 3


def test_derive_pairings_from_chemical_fails_on_out_of_range(tmp_path: Path) -> None:
    chem = tmp_path / "chem.parquet"
    out = tmp_path / "pairings.parquet"
    pl.DataFrame(
        [
            {"target_id": "T1", "resid": 1, "p_paired": -0.1},
            {"target_id": "T1", "resid": 2, "p_paired": 0.5},
        ]
    ).write_parquet(chem)
    with pytest.raises(PipelineError, match="pair_prob fora de \\[0,1\\]"):
        derive_pairings_from_chemical(repo_root=tmp_path, chemical_features_path=chem, out_path=out)

