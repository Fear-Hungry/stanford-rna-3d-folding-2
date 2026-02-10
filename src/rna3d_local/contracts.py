from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl

from .errors import raise_error


KEY_COL = "ID"


def _read_csv(path: Path, *, location: str) -> pl.DataFrame:
    if not path.exists():
        raise_error("VALIDATE", location, "arquivo nao encontrado", impact="1", examples=[str(path)])
    try:
        return pl.read_csv(path, infer_schema_length=1000, ignore_errors=False)
    except Exception as e:  # noqa: BLE001 - must wrap with actionable error
        raise_error("VALIDATE", location, "falha ao ler CSV", impact="1", examples=[f"{path.name}:{type(e).__name__}:{e}"])
    raise AssertionError("unreachable")


def _sample_keyset(df: pl.DataFrame, *, location: str) -> set[str]:
    if KEY_COL not in df.columns:
        raise_error("VALIDATE", location, "coluna chave ausente", impact="1", examples=[KEY_COL])
    s = df.get_column(KEY_COL)
    if s.null_count() > 0:
        raise_error("VALIDATE", location, "chave com valores nulos", impact=str(s.null_count()), examples=[])
    return set(s.to_list())


def _ensure_no_nulls(df: pl.DataFrame, *, key_examples: Iterable[str], location: str) -> None:
    null_counts = {c: int(df.get_column(c).null_count()) for c in df.columns}
    bad = {c: n for c, n in null_counts.items() if n > 0}
    if bad:
        # surface the worst offender first
        worst_col, worst_n = max(bad.items(), key=lambda kv: kv[1])
        raise_error(
            "VALIDATE",
            location,
            f"valores nulos detectados na coluna {worst_col}",
            impact=str(worst_n),
            examples=list(key_examples),
        )


def validate_submission_against_sample(*, sample_path: Path, submission_path: Path) -> None:
    """
    Strict contract validation (fail-fast):
    - same columns (names + order)
    - unique keys
    - keyset matches exactly
    - no nulls
    """
    location = "src/rna3d_local/contracts.py:validate_submission_against_sample"
    sample = _read_csv(sample_path, location=location)
    sub = _read_csv(submission_path, location=location)

    if sub.columns != sample.columns:
        missing = [c for c in sample.columns if c not in sub.columns]
        extra = [c for c in sub.columns if c not in sample.columns]
        raise_error(
            "VALIDATE",
            location,
            "colunas da submissao nao batem com sample_submission",
            impact=f"missing={len(missing)} extra={len(extra)}",
            examples=(missing + extra)[:8],
        )

    if KEY_COL not in sub.columns:
        raise_error("VALIDATE", location, "coluna chave ausente", impact="1", examples=[KEY_COL])

    # duplicates
    dup_mask = sub.get_column(KEY_COL).is_duplicated()
    dup_n = int(dup_mask.sum())
    if dup_n > 0:
        dup_ids = sub.filter(dup_mask).get_column(KEY_COL).unique().head(8).to_list()
        raise_error(
            "VALIDATE",
            location,
            "chaves duplicadas na submissao",
            impact=str(dup_n),
            examples=[str(x) for x in dup_ids],
        )

    sample_keys = _sample_keyset(sample, location=location)
    sub_keys = _sample_keyset(sub, location=location)

    missing_keys = list(sample_keys - sub_keys)
    extra_keys = list(sub_keys - sample_keys)
    if missing_keys or extra_keys:
        raise_error(
            "VALIDATE",
            location,
            "chaves da submissao nao batem com sample_submission",
            impact=f"missing={len(missing_keys)} extra={len(extra_keys)}",
            examples=[*(missing_keys[:4]), *(extra_keys[:4])],
        )

    # nulls: report with example IDs that have any nulls
    any_null = sub.select(pl.any_horizontal(pl.all().is_null()).alias("_any_null")).get_column("_any_null")
    any_null_n = int(any_null.sum())
    if any_null_n > 0:
        ids = sub.filter(any_null).get_column(KEY_COL).head(8).to_list()
        raise_error(
            "VALIDATE",
            location,
            "linhas com valores nulos (submissao incompleta)",
            impact=str(any_null_n),
            examples=[str(x) for x in ids],
        )


def validate_solution_against_sample(*, sample_path: Path, solution_path: Path) -> None:
    """
    Validates that a solution file matches the sample keyset and has required columns.
    Used to gate local scoring datasets (fail-fast).
    """
    location = "src/rna3d_local/contracts.py:validate_solution_against_sample"
    sample = _read_csv(sample_path, location=location)
    sol = _read_csv(solution_path, location=location)

    if KEY_COL not in sol.columns:
        raise_error("VALIDATE", location, "solucao sem coluna chave", impact="1", examples=[KEY_COL])

    sample_keys = _sample_keyset(sample, location=location)
    sol_keys = _sample_keyset(sol, location=location)
    missing_keys = list(sample_keys - sol_keys)
    extra_keys = list(sol_keys - sample_keys)
    if missing_keys or extra_keys:
        raise_error(
            "VALIDATE",
            location,
            "chaves da solucao nao batem com sample_submission",
            impact=f"missing={len(missing_keys)} extra={len(extra_keys)}",
            examples=[*(missing_keys[:4]), *(extra_keys[:4])],
        )

