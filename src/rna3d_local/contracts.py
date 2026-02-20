from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import polars as pl

from .errors import raise_error


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

    coord_abs_max_raw = os.environ.get("RNA3D_SUBMISSION_COORD_ABS_MAX", "1000").strip()
    try:
        coord_abs_max = float(coord_abs_max_raw)
    except Exception:
        raise_error(stage, location, "RNA3D_SUBMISSION_COORD_ABS_MAX invalido", impact="1", examples=[coord_abs_max_raw])
    if not math.isfinite(coord_abs_max) or coord_abs_max <= 0:
        raise_error(stage, location, "RNA3D_SUBMISSION_COORD_ABS_MAX invalido", impact="1", examples=[coord_abs_max_raw])

    try:
        with sample_path.open("r", encoding="utf-8", newline="") as f_sample, submission_path.open("r", encoding="utf-8", newline="") as f_sub:
            sample_reader = csv.reader(f_sample)
            sub_reader = csv.reader(f_sub)
            try:
                sample_header = next(sample_reader)
            except StopIteration:
                raise_error(stage, location, "sample vazio", impact="1", examples=[str(sample_path)])
            try:
                sub_header = next(sub_reader)
            except StopIteration:
                raise_error(stage, location, "submissao vazia", impact="1", examples=[str(submission_path)])

            if sample_header != sub_header:
                raise_error(
                    stage,
                    location,
                    "colunas da submissao nao batem com sample",
                    impact=str(abs(len(sample_header) - len(sub_header))),
                    examples=[f"sample={sample_header[:6]}", f"submission={sub_header[:6]}"],
                )
            if "ID" not in sample_header:
                raise_error(stage, location, "sample sem coluna ID", impact="1", examples=["ID"])
            id_idx = sample_header.index("ID")

            coord_idxs = [idx for idx, name in enumerate(sample_header) if name.startswith(("x_", "y_", "z_"))]
            fixed_idxs = [idx for idx, _name in enumerate(sample_header) if idx not in coord_idxs]

            mismatches: list[str] = []
            mismatch_count = 0
            fixed_mismatches: list[str] = []
            fixed_mismatch_count = 0
            bad_values: list[str] = []
            sample_rows = 0
            sub_rows = 0

            stop_early_for_bad_coords = False
            for row_index, (srow, prow) in enumerate(zip(sample_reader, sub_reader, strict=False)):
                sample_rows += 1
                sub_rows += 1
                if len(srow) != len(sample_header) or len(prow) != len(sample_header):
                    raise_error(
                        stage,
                        location,
                        "linha com numero de colunas divergente do header",
                        impact="1",
                        examples=[f"row={row_index}", f"sample_cols={len(srow)}", f"sub_cols={len(prow)}"],
                    )

                expected_id = srow[id_idx]
                got_id = prow[id_idx]
                if expected_id != got_id:
                    mismatch_count += 1
                    if len(mismatches) < 8:
                        mismatches.append(f"{row_index}:{expected_id}!={got_id}")
                for cidx in fixed_idxs:
                    if cidx == id_idx:
                        continue
                    if srow[cidx] != prow[cidx]:
                        fixed_mismatch_count += 1
                        if len(fixed_mismatches) < 8:
                            fixed_mismatches.append(
                                f"{sample_header[cidx]}@{row_index}:{srow[cidx]}!={prow[cidx]}"
                            )

                for cidx in coord_idxs:
                    value = prow[cidx]
                    col_name = sample_header[cidx]
                    try:
                        numeric = float(value)
                    except Exception:
                        bad_values.append(f"{col_name}@{row_index}:non-numeric")
                        if len(bad_values) >= 8:
                            stop_early_for_bad_coords = True
                            break
                        continue
                    if not math.isfinite(numeric):
                        bad_values.append(f"{col_name}@{row_index}:non-finite")
                    elif abs(numeric) > coord_abs_max:
                        bad_values.append(f"{col_name}@{row_index}:abs>{coord_abs_max:g}")
                    if len(bad_values) >= 8:
                        stop_early_for_bad_coords = True
                        break
                if stop_early_for_bad_coords:
                    break

            if bad_values:
                raise_error(stage, location, "coordenadas invalidas na submissao", impact=str(len(bad_values)), examples=bad_values)

            # Count trailing rows if lengths diverge.
            for _ in sample_reader:
                sample_rows += 1
            for _ in sub_reader:
                sub_rows += 1

            if sample_rows != sub_rows:
                raise_error(
                    stage,
                    location,
                    "numero de linhas divergente de sample",
                    impact=str(abs(sample_rows - sub_rows)),
                    examples=[f"sample={sample_rows}", f"submission={sub_rows}"],
                )
            if mismatch_count > 0:
                raise_error(
                    stage,
                    location,
                    "chaves/ordem da submissao nao batem com sample",
                    impact=str(mismatch_count),
                    examples=mismatches,
                )
            if fixed_mismatch_count > 0:
                raise_error(
                    stage,
                    location,
                    "valores fixos da submissao nao batem com sample",
                    impact=str(fixed_mismatch_count),
                    examples=fixed_mismatches,
                )
    except OSError as exc:
        raise_error(stage, location, "falha ao ler CSV para validacao streaming", impact="1", examples=[f"{type(exc).__name__}:{exc}"])

    # NOTE: return value is not used by callers; avoid loading full CSV into RAM.
    return pl.DataFrame()
