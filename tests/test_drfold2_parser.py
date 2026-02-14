from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl
from rna3d_local.drfold2 import _validate_target_rows, extract_target_coordinates_from_pdb
from rna3d_local.errors import PipelineError


def _write_pdb(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ATOM      1  C4'   A A   1      11.000  13.000  14.000  1.00 20.00           C",
                "ATOM      2  C4'   C A   2      12.000  14.000  15.000  1.00 20.00           C",
                "ATOM      3  C4'   G A   3      13.000  15.000  16.000  1.00 20.00           C",
                "TER",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_pdb_with_c1_and_c4(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ATOM      1  C4'   A A   1      11.000  13.000  14.000  1.00 20.00           C",
                "ATOM      2  C1'   A A   1      21.000  23.000  24.000  1.00 20.00           C",
                "ATOM      3  C4'   C A   2      12.000  14.000  15.000  1.00 20.00           C",
                "ATOM      4  C1'   C A   2      22.000  24.000  25.000  1.00 20.00           C",
                "ATOM      5  C4'   G A   3      13.000  15.000  16.000  1.00 20.00           C",
                "ATOM      6  C1'   G A   3      23.000  25.000  26.000  1.00 20.00           C",
                "TER",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_extract_target_coordinates_from_pdb_ok(tmp_path: Path) -> None:
    pdb = tmp_path / "m.pdb"
    _write_pdb(pdb)
    coords = extract_target_coordinates_from_pdb(
        pdb_path=pdb,
        target_sequence="ACG",
        location="tests/test_drfold2_parser.py:test_extract_target_coordinates_from_pdb_ok",
    )
    assert len(coords) == 3
    assert coords[0] == (11.0, 13.0, 14.0)
    assert coords[2] == (13.0, 15.0, 16.0)


def test_extract_target_coordinates_from_pdb_prefers_c1_atom(tmp_path: Path) -> None:
    pdb = tmp_path / "m_c1_first.pdb"
    _write_pdb_with_c1_and_c4(pdb)
    coords = extract_target_coordinates_from_pdb(
        pdb_path=pdb,
        target_sequence="ACG",
        location="tests/test_drfold2_parser.py:test_extract_target_coordinates_from_pdb_prefers_c1_atom",
    )
    assert len(coords) == 3
    assert coords[0] == (21.0, 23.0, 24.0)
    assert coords[2] == (23.0, 25.0, 26.0)


def test_extract_target_coordinates_from_pdb_fails_on_length_mismatch(tmp_path: Path) -> None:
    pdb = tmp_path / "m.pdb"
    _write_pdb(pdb)
    with pytest.raises(PipelineError):
        extract_target_coordinates_from_pdb(
            pdb_path=pdb,
            target_sequence="AC",
            location="tests/test_drfold2_parser.py:test_extract_target_coordinates_from_pdb_fails_on_length_mismatch",
        )


def test_validate_target_rows_rejects_null_or_placeholder_target_id() -> None:
    rows = pl.DataFrame(
        {
            "target_id": ["Q1", None, "None", "nan"],
            "sequence": ["ACGU", "ACGU", "ACGU", "ACGU"],
        }
    )
    with pytest.raises(PipelineError):
        _validate_target_rows(targets=rows, location="tests/test_drfold2_parser.py:test_validate_target_rows_rejects_null_or_placeholder_target_id")
