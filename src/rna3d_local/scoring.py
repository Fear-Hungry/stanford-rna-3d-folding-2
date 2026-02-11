from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow.dataset as ds

from .bigdata import (
    DEFAULT_MAX_ROWS_IN_MEMORY,
    DEFAULT_MEMORY_BUDGET_MB,
    assert_memory_budget,
    assert_row_budget,
)
from .contracts import validate_submission_against_sample, validate_solution_against_sample
from .errors import PipelineError, raise_error

DEFAULT_SCORE_CHUNK_ROWS = 100_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_metric(metric_py: Path):
    location = "src/rna3d_local/scoring.py:_load_metric"
    if not metric_py.exists():
        raise_error("SCORE", location, "metric.py nao encontrado", impact="1", examples=[str(metric_py)])
    spec = importlib.util.spec_from_file_location("tm_score_metric", metric_py)
    if spec is None or spec.loader is None:
        raise_error("SCORE", location, "falha ao carregar spec do metric.py", impact="1", examples=[str(metric_py)])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    if not hasattr(mod, "score"):
        raise_error("SCORE", location, "metric.py nao exporta score()", impact="1", examples=[str(metric_py)])
    return mod


def _target_from_id_series(s: pd.Series, *, location: str) -> pd.Series:
    # Use regex extract (instead of rsplit) to keep string dtype and avoid
    # building a large object-dtype temporary on big folds.
    tid = s.astype("string").str.extract(r"^(.*)_\d+$", expand=False)
    if tid.isna().any():
        bad = s[tid.isna()].head(8).astype(str).tolist()
        raise_error("SCORE", location, "falha ao derivar target_id de ID", impact=str(int(tid.isna().sum())), examples=bad)
    return tid


def _iter_table_chunks(*, path: Path, location: str, chunk_size: int) -> Iterator[pd.DataFrame]:
    if chunk_size <= 0:
        raise_error("SCORE", location, "chunk_size invalido (deve ser > 0)", impact="1", examples=[str(chunk_size)])
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            scanner = ds.dataset(path, format="parquet").scanner(batch_size=int(chunk_size))
            for batch in scanner.to_batches():
                yield batch.to_pandas()
            return
        except Exception as e:  # noqa: BLE001
            raise_error("SCORE", location, "falha ao ler parquet em lotes", impact="1", examples=[f"{path.name}:{type(e).__name__}:{e}"])
        raise AssertionError("unreachable")
    if suffix == ".csv":
        try:
            for chunk in pd.read_csv(path, chunksize=int(chunk_size), dtype_backend="pyarrow"):
                yield chunk
            return
        except Exception as e:  # noqa: BLE001
            raise_error("SCORE", location, "falha ao ler csv em lotes", impact="1", examples=[f"{path.name}:{type(e).__name__}:{e}"])
        raise AssertionError("unreachable")
    raise_error("SCORE", location, "formato nao suportado para score (use CSV ou Parquet)", impact="1", examples=[str(path)])
    raise AssertionError("unreachable")


def _iter_target_groups(
    *,
    path: Path,
    table_label: str,
    location: str,
    chunk_size: int,
    max_rows_in_memory: int,
) -> Iterator[tuple[str, pd.DataFrame]]:
    active_target: str | None = None
    active_parts: list[pd.DataFrame] = []
    seen_targets: set[str] = set()

    for chunk in _iter_table_chunks(path=path, location=location, chunk_size=chunk_size):
        if chunk.empty:
            continue
        if "ID" not in chunk.columns:
            raise_error("SCORE", location, "tabela sem coluna obrigatoria ID", impact="1", examples=[f"{table_label}:ID"])
        try:
            chunk = chunk.copy()
            chunk["__target_id"] = _target_from_id_series(chunk["ID"], location=location)
            chunk.reset_index(drop=True, inplace=True)
            target_series = chunk["__target_id"].astype("string")
            boundary_mask = target_series.ne(target_series.shift(1)).fillna(True)
            starts = list(target_series.index[boundary_mask])
            starts.append(len(chunk))
        except PipelineError:
            raise
        except Exception as e:  # noqa: BLE001
            raise_error(
                "SCORE",
                location,
                "falha ao agrupar linhas por target_id",
                impact="1",
                examples=[f"{table_label}:{type(e).__name__}:{e}"],
            )

        for i in range(len(starts) - 1):
            start = starts[i]
            end = starts[i + 1]
            g = chunk.iloc[start:end].drop(columns=["__target_id"])
            tid = str(target_series.iloc[start])
            if active_target is None:
                active_target = tid
                active_parts = [g]
                continue
            if tid == active_target:
                active_parts.append(g)
                continue
            if tid in seen_targets:
                raise_error(
                    "SCORE",
                    location,
                    f"target_id nao contiguo em {table_label}; use dados ordenados por ID",
                    impact="1",
                    examples=[tid],
                )
            merged = pd.concat(active_parts, ignore_index=True)
            assert_row_budget(
                stage="SCORE",
                location=location,
                rows=int(len(merged)),
                max_rows_in_memory=max_rows_in_memory,
                label=f"{table_label}:{active_target}",
            )
            yield active_target, merged
            seen_targets.add(active_target)
            active_target = tid
            active_parts = [g]

    if active_target is None:
        raise_error("SCORE", location, f"tabela sem linhas para score ({table_label})", impact="0", examples=[str(path)])
    if active_target in seen_targets:
        raise_error(
            "SCORE",
            location,
            f"target_id nao contiguo em {table_label}; estado final invalido",
            impact="1",
            examples=[active_target],
        )
    merged = pd.concat(active_parts, ignore_index=True)
    assert_row_budget(
        stage="SCORE",
        location=location,
        rows=int(len(merged)),
        max_rows_in_memory=max_rows_in_memory,
        label=f"{table_label}:{active_target}",
    )
    yield active_target, merged


def _ensure_group_ids_match(*, target_id: str, sol_df: pd.DataFrame, sub_df: pd.DataFrame, location: str) -> None:
    if len(sol_df) != len(sub_df):
        raise_error(
            "SCORE",
            location,
            "quantidade de linhas por target difere entre solucao e submissao",
            impact=f"target={target_id} solution={len(sol_df)} submission={len(sub_df)}",
            examples=[target_id],
        )
    sol_ids = sol_df["ID"].astype("string").reset_index(drop=True)
    sub_ids = sub_df["ID"].astype("string").reset_index(drop=True)
    mism = sol_ids.ne(sub_ids)
    if bool(mism.any()):
        mism_n = int(mism.sum())
        first_idx = int(mism[mism].index[0])
        ex_sol = str(sol_ids.iloc[first_idx])
        ex_sub = str(sub_ids.iloc[first_idx])
        raise_error(
            "SCORE",
            location,
            "ordem/chaves diferem entre solucao e submissao no target",
            impact=f"target={target_id} mismatches={mism_n}",
            examples=[f"expected={ex_sol}", f"actual={ex_sub}"],
        )


def _iter_aligned_target_groups(
    *,
    solution: Path,
    submission: Path,
    location: str,
    chunk_size: int,
    max_rows_in_memory: int,
) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    sol_iter = _iter_target_groups(
        path=solution,
        table_label="solution",
        location=location,
        chunk_size=chunk_size,
        max_rows_in_memory=max_rows_in_memory,
    )
    sub_iter = _iter_target_groups(
        path=submission,
        table_label="submission",
        location=location,
        chunk_size=chunk_size,
        max_rows_in_memory=max_rows_in_memory,
    )
    sentinel = object()
    while True:
        sol_item = next(sol_iter, sentinel)
        sub_item = next(sub_iter, sentinel)
        if sol_item is sentinel and sub_item is sentinel:
            return
        if sol_item is sentinel:
            sub_tid = str(sub_item[0]) if sub_item is not sentinel else "?"
            raise_error(
                "SCORE",
                location,
                "submissao possui targets extras em relacao a solucao",
                impact="1",
                examples=[sub_tid],
            )
        if sub_item is sentinel:
            sol_tid = str(sol_item[0]) if sol_item is not sentinel else "?"
            raise_error(
                "SCORE",
                location,
                "solucao possui targets extras em relacao a submissao",
                impact="1",
                examples=[sol_tid],
            )
        sol_tid, sol_df = sol_item
        sub_tid, sub_df = sub_item
        if sol_tid != sub_tid:
            raise_error(
                "SCORE",
                location,
                "targets da solucao e submissao nao batem na leitura em lote",
                impact="1",
                examples=[f"solution={sol_tid}", f"submission={sub_tid}"],
            )
        _ensure_group_ids_match(target_id=sol_tid, sol_df=sol_df, sub_df=sub_df, location=location)
        yield sol_tid, sol_df, sub_df


def _coord_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("x_") or c.startswith("y_") or c.startswith("z_")]


def _prepare_metric_frame(*, df: pd.DataFrame, coord_cols: list[str], location: str, context: str) -> pd.DataFrame:
    if not coord_cols:
        return df
    try:
        return df.astype({c: "float64" for c in coord_cols}, copy=False)
    except Exception as e:  # noqa: BLE001
        raise_error(
            "SCORE",
            location,
            "falha ao converter coordenadas para formato do metric",
            impact="1",
            examples=[context, f"{type(e).__name__}:{e}"],
        )
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class ScoreResult:
    score: float
    per_target: dict[str, float] | None


def _score_per_target_groups(
    *,
    metric,
    solution: Path,
    submission: Path,
    usalign_bin: Path,
    location: str,
    memory_budget_mb: int,
    max_rows_in_memory: int,
    chunk_size: int,
    keep_per_target: bool,
) -> ScoreResult:
    results: dict[str, float] = {}
    total = 0.0
    n_targets = 0

    for target_id, sol_df, sub_df in _iter_aligned_target_groups(
        solution=solution,
        submission=submission,
        location=location,
        chunk_size=chunk_size,
        max_rows_in_memory=max_rows_in_memory,
    ):
        sol_coord_cols = _coord_columns(sol_df)
        sub_coord_cols = _coord_columns(sub_df)
        gsol = _prepare_metric_frame(
            df=sol_df,
            coord_cols=sol_coord_cols,
            location=location,
            context=f"target={target_id},table=solution",
        )
        gsub = _prepare_metric_frame(
            df=sub_df,
            coord_cols=sub_coord_cols,
            location=location,
            context=f"target={target_id},table=submission",
        )
        try:
            target_score = float(metric.score(gsol, gsub, "ID", usalign_bin_hint=str(usalign_bin)))
        except Exception as e:  # noqa: BLE001
            raise_error(
                "SCORE",
                location,
                "falha no metric.score para target",
                impact="1",
                examples=[f"target={target_id}", f"{type(e).__name__}:{e}"],
            )
        if keep_per_target:
            results[target_id] = target_score
        total += target_score
        n_targets += 1
        assert_memory_budget(
            stage="SCORE",
            location=location,
            budget_mb=memory_budget_mb,
            context_examples=[f"target={target_id}", f"n_targets={n_targets}"],
        )

    if n_targets == 0:
        raise_error("SCORE", location, "nenhum target encontrado para score", impact="0", examples=[])
    overall = float(total / n_targets)
    return ScoreResult(score=overall, per_target=results if keep_per_target else None)


def score_submission(
    *,
    sample_submission: Path,
    solution: Path,
    submission: Path,
    metric_py: Path,
    usalign_bin: Path,
    per_target: bool,
    keep_tmp: bool,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    max_rows_in_memory: int = DEFAULT_MAX_ROWS_IN_MEMORY,
    chunk_size: int = DEFAULT_SCORE_CHUNK_ROWS,
) -> ScoreResult:
    """
    Runs Kaggle-identical scoring (vendored TM-score metric) with strict gating.
    """
    location = "src/rna3d_local/scoring.py:score_submission"
    assert_memory_budget(stage="SCORE", location=location, budget_mb=memory_budget_mb)
    for p in (sample_submission, solution, submission, metric_py, usalign_bin):
        if not p.exists():
            raise_error("SCORE", location, "arquivo obrigatorio ausente", impact="1", examples=[str(p)])

    # contract gating
    validate_submission_against_sample(sample_path=sample_submission, submission_path=submission)
    validate_solution_against_sample(sample_path=sample_submission, solution_path=solution)

    metric = _load_metric(metric_py)

    # Run metric in isolated working dir (it writes many *.pdb).
    work = Path(tempfile.mkdtemp(prefix="rna3d_score_"))
    old = Path.cwd()
    try:
        os.chdir(work)
        return _score_per_target_groups(
            metric=metric,
            solution=solution,
            submission=submission,
            usalign_bin=usalign_bin,
            location=location,
            memory_budget_mb=memory_budget_mb,
            max_rows_in_memory=max_rows_in_memory,
            chunk_size=chunk_size,
            keep_per_target=per_target,
        )
    finally:
        os.chdir(old)
        if keep_tmp:
            # preserve for debugging
            dbg = Path("runs") / "debug_metric_tmp"
            dbg.mkdir(parents=True, exist_ok=True)
            for p in work.glob("*.pdb"):
                p.replace(dbg / p.name)
        shutil.rmtree(work, ignore_errors=True)


def write_score_artifacts(*, out_dir: Path, result: ScoreResult, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"created_utc": _utc_now(), "score": result.score, "meta": meta}
    (out_dir / "score.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result.per_target is not None:
        df = pd.DataFrame([{"target_id": k, "score": v} for k, v in result.per_target.items()]).sort_values("target_id")
        df.to_csv(out_dir / "per_target.csv", index=False)
