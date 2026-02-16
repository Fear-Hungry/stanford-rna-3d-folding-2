from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from .errors import raise_error
from .io_tables import read_table


def require_columns(df: pl.DataFrame, required: list[str], *, stage: str, location: str, label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise_error(
            stage,
            location,
            f"{label} sem coluna obrigatoria",
            impact=str(len(missing)),
            examples=missing[:8],
        )


def require_non_null(df: pl.DataFrame, columns: list[str], *, stage: str, location: str, label: str) -> None:
    bad: list[str] = []
    for column in columns:
        if column not in df.columns:
            bad.append(f"{column}:missing")
            continue
        null_count = int(df.get_column(column).null_count())
        if null_count > 0:
            bad.append(f"{column}:{null_count}")
    if bad:
        raise_error(stage, location, f"{label} com valores nulos", impact=str(len(bad)), examples=bad[:8])


def parse_date_column(df: pl.DataFrame, column: str, *, stage: str, location: str, label: str) -> pl.DataFrame:
    if column not in df.columns:
        raise_error(stage, location, f"{label} sem coluna obrigatoria", impact="1", examples=[column])
    parsed = df.with_columns(pl.col(column).cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias(column))
    bad = int(parsed.get_column(column).null_count())
    if bad > 0:
        examples = parsed.filter(pl.col(column).is_null()).head(8)
        ids = [str(item) for item in examples.to_dicts()]
        raise_error(stage, location, f"{label} com data invalida", impact=str(bad), examples=ids[:8])
    return parsed


def validate_submission_against_sample(*, sample_path: Path, submission_path: Path) -> pl.DataFrame:
    location = "src/rna3d_local/contracts.py:validate_submission_against_sample"
    stage = "CHECK_SUBMISSION"
    sample = read_table(sample_path, stage=stage, location=location)
    submission = read_table(submission_path, stage=stage, location=location)

    if sample.columns != submission.columns:
        raise_error(
            stage,
            location,
            "colunas da submissao nao batem com sample",
            impact=str(abs(len(sample.columns) - len(submission.columns))),
            examples=[f"sample={sample.columns[:6]}", f"submission={submission.columns[:6]}"],
        )
    if sample.height != submission.height:
        raise_error(
            stage,
            location,
            "numero de linhas divergente de sample",
            impact=str(abs(sample.height - submission.height)),
            examples=[f"sample={sample.height}", f"submission={submission.height}"],
        )

    id_col = "ID"
    if id_col not in sample.columns:
        raise_error(stage, location, "sample sem coluna ID", impact="1", examples=["ID"])
    sample_ids = sample.get_column(id_col).cast(pl.Utf8).to_list()
    submission_ids = submission.get_column(id_col).cast(pl.Utf8).to_list()
    if sample_ids != submission_ids:
        mismatches = []
        for idx, (expected, got) in enumerate(zip(sample_ids, submission_ids)):
            if expected != got:
                mismatches.append(f"{idx}:{expected}!={got}")
            if len(mismatches) >= 8:
                break
        raise_error(
            stage,
            location,
            "chaves/ordem da submissao nao batem com sample",
            impact=str(sum(1 for a, b in zip(sample_ids, submission_ids) if a != b)),
            examples=mismatches,
        )

    coord_columns = [column for column in submission.columns if column.startswith(("x_", "y_", "z_"))]
    bad_values: list[str] = []
    for column in coord_columns:
        for row_index, value in enumerate(submission.get_column(column).to_list()):
            try:
                numeric = float(value)
            except Exception:
                bad_values.append(f"{column}@{row_index}:non-numeric")
                if len(bad_values) >= 8:
                    break
                continue
            if not math.isfinite(numeric):
                bad_values.append(f"{column}@{row_index}:non-finite")
            elif abs(numeric) > 1e6:
                bad_values.append(f"{column}@{row_index}:out-of-range")
            if len(bad_values) >= 8:
                break
        if len(bad_values) >= 8:
            break
    if bad_values:
        raise_error(stage, location, "coordenadas invalidas na submissao", impact=str(len(bad_values)), examples=bad_values)
    return submission
