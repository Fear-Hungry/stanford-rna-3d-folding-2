from __future__ import annotations

from pathlib import Path

import pytest

from rna3d_local.drfold2 import extract_target_coordinates_from_pdb
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


def test_extract_target_coordinates_from_pdb_fails_on_length_mismatch(tmp_path: Path) -> None:
    pdb = tmp_path / "m.pdb"
    _write_pdb(pdb)
    with pytest.raises(PipelineError):
        extract_target_coordinates_from_pdb(
            pdb_path=pdb,
            target_sequence="AC",
            location="tests/test_drfold2_parser.py:test_extract_target_coordinates_from_pdb_fails_on_length_mismatch",
        )

