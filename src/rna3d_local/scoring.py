from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .contracts import validate_submission_against_sample, validate_solution_against_sample
from .errors import raise_error


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


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


@dataclass(frozen=True)
class ScoreResult:
    score: float
    per_target: dict[str, float] | None


def score_submission(
    *,
    sample_submission: Path,
    solution: Path,
    submission: Path,
    metric_py: Path,
    usalign_bin: Path,
    per_target: bool,
    keep_tmp: bool,
) -> ScoreResult:
    """
    Runs Kaggle-identical scoring (vendored TM-score metric) with strict gating.
    """
    location = "src/rna3d_local/scoring.py:score_submission"
    for p in (sample_submission, solution, submission, metric_py, usalign_bin):
        if not p.exists():
            raise_error("SCORE", location, "arquivo obrigatorio ausente", impact="1", examples=[str(p)])

    # contract gating
    validate_submission_against_sample(sample_path=sample_submission, submission_path=submission)
    validate_solution_against_sample(sample_path=sample_submission, solution_path=solution)

    metric = _load_metric(metric_py)

    sol_df = _read_frame(solution)
    sub_df = _read_frame(submission)

    # Run metric in isolated working dir (it writes many *.pdb).
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        old = Path.cwd()
        try:
            os.chdir(work)
            if per_target:
                # metric.score groups internally by target_id; feed per target slices for diagnostics.
                sol_df = sol_df.copy()
                sub_df = sub_df.copy()
                sol_df["target_id"] = sol_df["ID"].apply(lambda x: "_".join(str(x).split("_")[:-1]))
                sub_df["target_id"] = sub_df["ID"].apply(lambda x: "_".join(str(x).split("_")[:-1]))
                results: dict[str, float] = {}
                for tid, gsol in sol_df.groupby("target_id"):
                    gsub = sub_df[sub_df["target_id"] == tid]
                    # metric expects full columns, including chain/copy in solution.
                    results[str(tid)] = float(metric.score(gsol, gsub, "ID", usalign_bin_hint=str(usalign_bin)))
                overall = float(sum(results.values()) / len(results)) if results else 0.0
                return ScoreResult(score=overall, per_target=results)
            overall = float(metric.score(sol_df, sub_df, "ID", usalign_bin_hint=str(usalign_bin)))
            return ScoreResult(score=overall, per_target=None)
        finally:
            os.chdir(old)
            if keep_tmp:
                # preserve for debugging
                dbg = Path("runs") / "debug_metric_tmp"
                dbg.mkdir(parents=True, exist_ok=True)
                for p in work.glob("*.pdb"):
                    p.replace(dbg / p.name)


def write_score_artifacts(*, out_dir: Path, result: ScoreResult, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"created_utc": _utc_now(), "score": result.score, "meta": meta}
    (out_dir / "score.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result.per_target is not None:
        df = pd.DataFrame([{"target_id": k, "score": v} for k, v in result.per_target.items()]).sort_values("target_id")
        df.to_csv(out_dir / "per_target.csv", index=False)

