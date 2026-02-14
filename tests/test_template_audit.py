from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.template_audit import audit_external_templates


def _write_external_templates(path: Path, *, release_date: str) -> None:
    pl.DataFrame(
        [
            {
                "template_id": "EXT1",
                "sequence": "ACG",
                "release_date": release_date,
                "resid": 1,
                "resname": "A",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "source": "ext",
            },
            {
                "template_id": "EXT1",
                "sequence": "ACG",
                "release_date": release_date,
                "resid": 2,
                "resname": "C",
                "x": 1.0,
                "y": 0.0,
                "z": 0.0,
                "source": "ext",
            },
            {
                "template_id": "EXT1",
                "sequence": "ACG",
                "release_date": release_date,
                "resid": 3,
                "resname": "G",
                "x": 2.0,
                "y": 0.0,
                "z": 0.0,
                "source": "ext",
            },
        ]
    ).write_csv(path)


def test_audit_external_templates_success(tmp_path: Path) -> None:
    external_templates = tmp_path / "external_templates.csv"
    _write_external_templates(external_templates, release_date="2019-01-01")

    report = audit_external_templates(
        external_templates_path=external_templates,
        out_report_path=tmp_path / "audit.json",
    )
    assert report["n_rows"] == 3
    assert report["n_templates"] == 1
    assert (tmp_path / "audit.json").exists()


def test_audit_external_templates_rejects_future_release_date(tmp_path: Path) -> None:
    external_templates = tmp_path / "external_templates.csv"
    future_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    _write_external_templates(external_templates, release_date=future_date)

    with pytest.raises(PipelineError):
        audit_external_templates(external_templates_path=external_templates)
