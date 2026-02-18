from __future__ import annotations

from pathlib import Path

import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.runners.boltz1 import _extract_c1_from_pdb, _normalize_plddt as _normalize_boltz_plddt
from rna3d_local.runners.chai1 import _normalize_plddt as _normalize_chai_plddt


def _pdb_c1_line(*, serial: int, resname: str, chain: str, resid: int, x: float, y: float, z: float, bfactor: float) -> str:
    return (
        f"ATOM  {serial:5d}  C1' {resname:>3} {chain:1}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{bfactor:6.2f}           C"
    )


def test_boltz_extract_c1_from_pdb_reads_confidence_from_bfactor(tmp_path: Path) -> None:
    pdb_path = tmp_path / "model.pdb"
    lines = [
        _pdb_c1_line(serial=1, resname="A", chain="A", resid=1, x=1.0, y=2.0, z=3.0, bfactor=80.0),
        _pdb_c1_line(serial=2, resname="C", chain="A", resid=2, x=4.0, y=5.0, z=6.0, bfactor=60.0),
        "END",
    ]
    pdb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    coords, confidence = _extract_c1_from_pdb(
        pdb_path=pdb_path,
        chain_order=["A"],
        expected_seq_by_chain=["AC"],
        stage="TEST",
        location="tests/test_model_confidence_extraction.py:test_boltz_extract_c1_from_pdb_reads_confidence_from_bfactor",
        target_id="T1",
    )
    assert len(coords) == 2
    assert pytest.approx(confidence, rel=1e-6, abs=1e-6) == 0.70


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.85, 0.85),
        (85.0, 0.85),
        (100.0, 1.0),
    ],
)
def test_plddt_normalization_accepts_expected_ranges(raw: float, expected: float) -> None:
    chai = _normalize_chai_plddt(
        raw,
        stage="TEST",
        location="tests/test_model_confidence_extraction.py:test_plddt_normalization_accepts_expected_ranges:chai",
        target_id="T1",
    )
    boltz = _normalize_boltz_plddt(
        raw,
        stage="TEST",
        location="tests/test_model_confidence_extraction.py:test_plddt_normalization_accepts_expected_ranges:boltz",
        target_id="T1",
    )
    assert pytest.approx(chai, rel=1e-6, abs=1e-6) == expected
    assert pytest.approx(boltz, rel=1e-6, abs=1e-6) == expected


def test_plddt_normalization_rejects_invalid_values() -> None:
    with pytest.raises(PipelineError, match="fora do intervalo esperado"):
        _normalize_chai_plddt(
            150.0,
            stage="TEST",
            location="tests/test_model_confidence_extraction.py:test_plddt_normalization_rejects_invalid_values:chai",
            target_id="T1",
        )
    with pytest.raises(PipelineError, match="negativo"):
        _normalize_boltz_plddt(
            -1.0,
            stage="TEST",
            location="tests/test_model_confidence_extraction.py:test_plddt_normalization_rejects_invalid_values:boltz",
            target_id="T1",
        )
