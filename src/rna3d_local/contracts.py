from __future__ import annotations

from pathlib import Path

import polars as pl

from .errors import raise_error


KEY_COL = "ID"


def _scan_table(path: Path, *, location: str) -> pl.LazyFrame:
    if not path.exists():
        raise_error("VALIDATE", location, "arquivo nao encontrado", impact="1", examples=[str(path)])
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pl.scan_csv(path, infer_schema_length=1000, ignore_errors=False)
        if suffix == ".parquet":
            return pl.scan_parquet(path)
        raise_error(
            "VALIDATE",
            location,
            "formato nao suportado para tabela (use CSV ou Parquet)",
            impact="1",
            examples=[str(path)],
        )
    except Exception as e:  # noqa: BLE001
        raise_error("VALIDATE", location, "falha ao abrir tabela", impact="1", examples=[f"{path.name}:{type(e).__name__}:{e}"])
    raise AssertionError("unreachable")


def _table_columns(path: Path, *, location: str) -> list[str]:
    lf = _scan_table(path, location=location)
    try:
        return list(lf.collect_schema().names())
    except Exception as e:  # noqa: BLE001
        raise_error(
            "VALIDATE",
            location,
            "falha ao ler cabecalho da tabela",
            impact="1",
            examples=[f"{path.name}:{type(e).__name__}:{e}"],
        )
    raise AssertionError("unreachable")


def _read_key_column(path: Path, *, location: str) -> pl.DataFrame:
    lf = _scan_table(path, location=location)
    try:
        return lf.select(pl.col(KEY_COL).cast(pl.Utf8).alias(KEY_COL)).collect()
    except Exception as e:  # noqa: BLE001
        raise_error(
            "VALIDATE",
            location,
            "falha ao carregar coluna chave",
            impact="1",
            examples=[f"{path.name}:{type(e).__name__}:{e}"],
        )
    raise AssertionError("unreachable")


def _ensure_key_not_null(df: pl.DataFrame, *, location: str) -> None:
    null_n = int(df.get_column(KEY_COL).null_count())
    if null_n > 0:
        raise_error("VALIDATE", location, "chave com valores nulos", impact=str(null_n), examples=[])


def _validate_no_duplicate_keys(*, keys_df: pl.DataFrame, location: str, table_name: str) -> None:
    dup_lf = keys_df.lazy().group_by(KEY_COL).len().filter(pl.col("len") > 1)
    try:
        dup_n = int(dup_lf.select((pl.col("len") - 1).sum().fill_null(0).alias("_dup")).collect().item(0, 0))
        dup_ids = dup_lf.select(pl.col(KEY_COL)).limit(8).collect().get_column(KEY_COL).to_list()
    except Exception as e:  # noqa: BLE001
        raise_error(
            "VALIDATE",
            location,
            "falha ao validar duplicatas de chave",
            impact="1",
            examples=[f"{table_name}:{type(e).__name__}:{e}"],
        )
    if dup_n > 0:
        raise_error(
            "VALIDATE",
            location,
            f"chaves duplicadas em {table_name}",
            impact=str(dup_n),
            examples=[str(x) for x in dup_ids],
        )


def _keyset_diff(
    *,
    expected_keys: pl.DataFrame,
    actual_keys: pl.DataFrame,
    location: str,
    mismatch_msg: str,
) -> None:
    expected_unique = expected_keys.lazy().unique()
    actual_unique = actual_keys.lazy().unique()

    missing_lf = expected_unique.join(actual_unique, on=KEY_COL, how="anti").select(pl.col(KEY_COL))
    extra_lf = actual_unique.join(expected_unique, on=KEY_COL, how="anti").select(pl.col(KEY_COL))

    try:
        missing_n = int(missing_lf.select(pl.len().alias("_n")).collect().item(0, 0))
        extra_n = int(extra_lf.select(pl.len().alias("_n")).collect().item(0, 0))
        missing_examples = missing_lf.limit(4).collect().get_column(KEY_COL).to_list()
        extra_examples = extra_lf.limit(4).collect().get_column(KEY_COL).to_list()
    except Exception as e:  # noqa: BLE001
        raise_error("VALIDATE", location, "falha ao comparar conjuntos de chaves", impact="1", examples=[f"{type(e).__name__}:{e}"])

    if missing_n or extra_n:
        raise_error(
            "VALIDATE",
            location,
            mismatch_msg,
            impact=f"missing={missing_n} extra={extra_n}",
            examples=[*(missing_examples[:4]), *(extra_examples[:4])],
        )


def _validate_key_order(
    *,
    expected_keys: pl.DataFrame,
    actual_keys: pl.DataFrame,
    location: str,
    mismatch_msg: str,
) -> None:
    try:
        exp = expected_keys.lazy().with_row_index("_rn")
        act = actual_keys.lazy().with_row_index("_rn")
        joined = exp.join(act, on="_rn", how="inner", suffix="_actual")
        mism_lf = joined.filter(pl.col(KEY_COL) != pl.col(f"{KEY_COL}_actual"))
        mismatch_n = int(mism_lf.select(pl.len().alias("_n")).collect().item(0, 0))
        examples_df = mism_lf.select(
            pl.col(KEY_COL).alias("expected"),
            pl.col(f"{KEY_COL}_actual").alias("actual"),
        ).limit(4).collect()
        examples = [f"expected={e} actual={a}" for e, a in examples_df.iter_rows()]
    except Exception as e:  # noqa: BLE001
        raise_error("VALIDATE", location, "falha ao validar ordem de chaves", impact="1", examples=[f"{type(e).__name__}:{e}"])
    if mismatch_n > 0:
        raise_error("VALIDATE", location, mismatch_msg, impact=f"mismatch={mismatch_n}", examples=examples)


def _validate_no_null_rows(*, table_path: Path, columns: list[str], location: str) -> None:
    lf = _scan_table(table_path, location=location)
    any_null = pl.any_horizontal([pl.col(c).is_null() for c in columns]).alias("_any_null")
    try:
        null_n = int(lf.select(any_null.sum().cast(pl.Int64).alias("_n")).collect().item(0, 0))
        if null_n == 0:
            return
        ids = (
            lf.filter(any_null)
            .select(pl.col(KEY_COL).cast(pl.Utf8).alias(KEY_COL))
            .limit(8)
            .collect()
            .get_column(KEY_COL)
            .to_list()
        )
    except Exception as e:  # noqa: BLE001
        raise_error("VALIDATE", location, "falha ao validar nulos", impact="1", examples=[f"{type(e).__name__}:{e}"])
    raise_error(
        "VALIDATE",
        location,
        "linhas com valores nulos (submissao incompleta)",
        impact=str(null_n),
        examples=[str(x) for x in ids],
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

    sample_cols = _table_columns(sample_path, location=location)
    sub_cols = _table_columns(submission_path, location=location)

    if sub_cols != sample_cols:
        missing = [c for c in sample_cols if c not in sub_cols]
        extra = [c for c in sub_cols if c not in sample_cols]
        raise_error(
            "VALIDATE",
            location,
            "colunas da submissao nao batem com sample_submission",
            impact=f"missing={len(missing)} extra={len(extra)}",
            examples=(missing + extra)[:8],
        )

    if KEY_COL not in sub_cols:
        raise_error("VALIDATE", location, "coluna chave ausente", impact="1", examples=[KEY_COL])

    sample_keys = _read_key_column(sample_path, location=location)
    sub_keys = _read_key_column(submission_path, location=location)

    _ensure_key_not_null(sample_keys, location=location)
    _ensure_key_not_null(sub_keys, location=location)
    _validate_no_duplicate_keys(keys_df=sample_keys, location=location, table_name="sample_submission")
    _validate_no_duplicate_keys(keys_df=sub_keys, location=location, table_name="submissao")

    _keyset_diff(
        expected_keys=sample_keys,
        actual_keys=sub_keys,
        location=location,
        mismatch_msg="chaves da submissao nao batem com sample_submission",
    )
    _validate_key_order(
        expected_keys=sample_keys,
        actual_keys=sub_keys,
        location=location,
        mismatch_msg="ordem de chaves da submissao nao bate com sample_submission",
    )

    _validate_no_null_rows(table_path=submission_path, columns=sub_cols, location=location)


def validate_solution_against_sample(*, sample_path: Path, solution_path: Path) -> None:
    """
    Validates that a solution file matches the sample keyset and has required columns.
    Used to gate local scoring datasets (fail-fast).
    """
    location = "src/rna3d_local/contracts.py:validate_solution_against_sample"
    sol_cols = _table_columns(solution_path, location=location)
    if KEY_COL not in sol_cols:
        raise_error("VALIDATE", location, "solucao sem coluna chave", impact="1", examples=[KEY_COL])

    sample_keys = _read_key_column(sample_path, location=location)
    sol_keys = _read_key_column(solution_path, location=location)

    _ensure_key_not_null(sample_keys, location=location)
    _ensure_key_not_null(sol_keys, location=location)
    _validate_no_duplicate_keys(keys_df=sample_keys, location=location, table_name="sample_submission")
    _validate_no_duplicate_keys(keys_df=sol_keys, location=location, table_name="solucao")

    _keyset_diff(
        expected_keys=sample_keys,
        actual_keys=sol_keys,
        location=location,
        mismatch_msg="chaves da solucao nao batem com sample_submission",
    )
    _validate_key_order(
        expected_keys=sample_keys,
        actual_keys=sol_keys,
        location=location,
        mismatch_msg="ordem de chaves da solucao nao bate com sample_submission",
    )
