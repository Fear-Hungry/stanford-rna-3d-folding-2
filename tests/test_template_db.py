from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.template_db import build_template_db


def _write_csv(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_csv(path)


def test_build_template_db_success(tmp_path: Path) -> None:
    external = tmp_path / "external_templates.csv"
    _write_csv(
        external,
        [
            {
                "template_id": "T1",
                "sequence": "ACG",
                "release_date": "2020-01-01",
                "resid": 1,
                "resname": "A",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "source": "ext",
            },
            {
                "template_id": "T1",
                "sequence": "ACG",
                "release_date": "2020-01-01",
                "resid": 2,
                "resname": "C",
                "x": 1.0,
                "y": 0.0,
                "z": 0.0,
                "source": "ext",
            },
            {
                "template_id": "T1",
                "sequence": "ACG",
                "release_date": "2020-01-01",
                "resid": 3,
                "resname": "G",
                "x": 2.0,
                "y": 0.0,
                "z": 0.0,
                "source": "ext",
            },
        ],
    )
    out = build_template_db(repo_root=tmp_path, external_templates_path=external, out_dir=tmp_path / "template_db")
    assert out.templates_path.exists()
    assert out.template_index_path.exists()
    index = pl.read_parquet(out.template_index_path)
    assert index.height == 1
    assert set(index.columns) >= {"template_uid", "sequence", "release_date"}


def test_build_template_db_fails_missing_column(tmp_path: Path) -> None:
    external = tmp_path / "external_templates.csv"
    _write_csv(
        external,
        [
            {
                "template_id": "T1",
                "sequence": "ACG",
                "release_date": "2020-01-01",
                "resid": 1,
                "resname": "A",
                "x": 0.0,
                "y": 0.0,
                # z missing
                "source": "ext",
            }
        ],
    )
    with pytest.raises(PipelineError, match="sem coluna obrigatoria"):
        build_template_db(repo_root=tmp_path, external_templates_path=external, out_dir=tmp_path / "template_db")
