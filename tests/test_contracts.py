from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rna3d_local.contracts import validate_submission_against_sample
from rna3d_local.contracts import validate_solution_against_sample
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


def test_contract_fails_on_invalid_key_order(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    _write(sample, "ID,resname,resid,x_1,y_1,z_1\nA_1,A,1,0,0,0\nA_2,C,2,0,0,0\n")
    _write(sub, "ID,resname,resid,x_1,y_1,z_1\nA_2,C,2,0,0,0\nA_1,A,1,0,0,0\n")
    with pytest.raises(PipelineError) as e:
        validate_submission_against_sample(sample_path=sample, submission_path=sub)
    msg = str(e.value)
    assert msg.startswith("[VALIDATE]")
    assert "ordem de chaves da submissao nao bate" in msg


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


def test_contract_fails_on_non_numeric_coordinate(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    _write(sample, "ID,resname,resid,x_1,y_1,z_1\nA_1,A,1,0,0,0\n")
    _write(sub, "ID,resname,resid,x_1,y_1,z_1\nA_1,A,1,abc,0,0\n")
    with pytest.raises(PipelineError) as e:
        validate_submission_against_sample(sample_path=sample, submission_path=sub)
    msg = str(e.value)
    assert msg.startswith("[VALIDATE]")
    assert "nao numerico" in msg


def test_contract_fails_on_non_finite_coordinate(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    _write(sample, "ID,resname,resid,x_1,y_1,z_1\nA_1,A,1,0,0,0\n")
    _write(sub, "ID,resname,resid,x_1,y_1,z_1\nA_1,A,1,inf,0,0\n")
    with pytest.raises(PipelineError) as e:
        validate_submission_against_sample(sample_path=sample, submission_path=sub)
    msg = str(e.value)
    assert msg.startswith("[VALIDATE]")
    assert "nao-finito" in msg


def test_contract_fails_on_out_of_range_coordinate(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    _write(sample, "ID,resname,resid,x_1,y_1,z_1\nA_1,A,1,0,0,0\n")
    _write(sub, "ID,resname,resid,x_1,y_1,z_1\nA_1,A,1,1000001,0,0\n")
    with pytest.raises(PipelineError) as e:
        validate_submission_against_sample(sample_path=sample, submission_path=sub)
    msg = str(e.value)
    assert msg.startswith("[VALIDATE]")
    assert "fora da faixa plausivel" in msg


def test_validate_solution_accepts_parquet_with_matching_keys(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sol = tmp_path / "solution.parquet"
    _write(sample, "ID,resname,resid,x_1,y_1,z_1\nA_1,A,1,0,0,0\nA_2,C,2,0,0,0\n")
    pl.DataFrame(
        [
            {"ID": "A_1", "resname": "A", "resid": 1, "x_1": 0.1, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "A_2", "resname": "C", "resid": 2, "x_1": 0.2, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
        ]
    ).write_parquet(sol)

    validate_solution_against_sample(sample_path=sample, solution_path=sol)


def test_validate_solution_fails_on_mismatch_keys_for_parquet(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sol = tmp_path / "solution.parquet"
    _write(sample, "ID,resname,resid,x_1,y_1,z_1\nA_1,A,1,0,0,0\nA_2,C,2,0,0,0\n")
    pl.DataFrame(
        [
            {"ID": "A_1", "resname": "A", "resid": 1, "x_1": 0.1, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "A_3", "resname": "G", "resid": 3, "x_1": 0.2, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
        ]
    ).write_parquet(sol)

    with pytest.raises(PipelineError) as e:
        validate_solution_against_sample(sample_path=sample, solution_path=sol)
    msg = str(e.value)
    assert msg.startswith("[VALIDATE]")
    assert "chaves da solucao nao batem" in msg


def test_validate_solution_fails_on_duplicate_keys_for_parquet(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sol = tmp_path / "solution.parquet"
    _write(sample, "ID,resname,resid,x_1,y_1,z_1\nA_1,A,1,0,0,0\nA_2,C,2,0,0,0\n")
    pl.DataFrame(
        [
            {"ID": "A_1", "resname": "A", "resid": 1, "x_1": 0.1, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "A_1", "resname": "A", "resid": 1, "x_1": 0.1, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
        ]
    ).write_parquet(sol)

    with pytest.raises(PipelineError) as e:
        validate_solution_against_sample(sample_path=sample, solution_path=sol)
    msg = str(e.value)
    assert msg.startswith("[VALIDATE]")
    assert "chaves duplicadas em solucao" in msg
