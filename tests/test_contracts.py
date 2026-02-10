from __future__ import annotations

from pathlib import Path

import pytest

from rna3d_local.contracts import validate_submission_against_sample
from rna3d_local.errors import PipelineError


def _write(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")


def test_contract_fails_on_column_mismatch(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    _write(sample, "ID,resname,resid,x_1,y_1,z_1\nA_1,A,1,0,0,0\n")
    _write(sub, "ID,resname,resid,x_1,y_1\nA_1,A,1,0,0\n")
    with pytest.raises(PipelineError) as e:
        validate_submission_against_sample(sample_path=sample, submission_path=sub)
    msg = str(e.value)
    assert msg.startswith("[VALIDATE]")
    assert "colunas da submissao nao batem" in msg


def test_contract_fails_on_missing_keys(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    _write(sample, "ID,resname,resid,x_1,y_1,z_1\nA_1,A,1,0,0,0\nA_2,C,2,0,0,0\n")
    _write(sub, "ID,resname,resid,x_1,y_1,z_1\nA_1,A,1,0,0,0\n")
    with pytest.raises(PipelineError) as e:
        validate_submission_against_sample(sample_path=sample, submission_path=sub)
    msg = str(e.value)
    assert msg.startswith("[VALIDATE]")
    assert "chaves da submissao nao batem" in msg
    assert "missing=1" in msg


def test_contract_fails_on_duplicate_keys(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    _write(sample, "ID,resname,resid,x_1,y_1,z_1\nA_1,A,1,0,0,0\nA_2,C,2,0,0,0\n")
    _write(sub, "ID,resname,resid,x_1,y_1,z_1\nA_1,A,1,0,0,0\nA_1,A,1,0,0,0\n")
    with pytest.raises(PipelineError) as e:
        validate_submission_against_sample(sample_path=sample, submission_path=sub)
    msg = str(e.value)
    assert msg.startswith("[VALIDATE]")
    assert "chaves duplicadas" in msg


def test_contract_fails_on_nulls(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    _write(sample, "ID,resname,resid,x_1,y_1,z_1\nA_1,A,1,0,0,0\n")
    _write(sub, "ID,resname,resid,x_1,y_1,z_1\nA_1,A,1,,0,0\n")
    with pytest.raises(PipelineError) as e:
        validate_submission_against_sample(sample_path=sample, submission_path=sub)
    msg = str(e.value)
    assert msg.startswith("[VALIDATE]")
    assert "valores nulos" in msg

