from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from ..errors import raise_error
from ..utils import utc_now_iso, write_json


@dataclass(frozen=True)
class KaggleOfficialScoreResult:
    score: float
    score_json_path: Path
    report_path: Path


def _require_pandas(*, stage: str, location: str):
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "dependencia pandas ausente para metric.py oficial", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    return pd


def _load_metric_module(*, metric_path: Path, stage: str, location: str) -> ModuleType:
    if not metric_path.exists():
        raise_error(stage, location, "metric.py oficial ausente", impact="1", examples=[str(metric_path)])
    if not metric_path.is_file():
        raise_error(stage, location, "metric_path nao e arquivo", impact="1", examples=[str(metric_path)])
    spec = importlib.util.spec_from_file_location("rna3d_kaggle_official_metric", str(metric_path))
    if spec is None or spec.loader is None:
        raise_error(stage, location, "falha ao carregar spec do metric.py", impact="1", examples=[str(metric_path)])
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "falha ao importar metric.py oficial", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    return module


def _validate_ids(*, sol_df: Any, sub_df: Any, stage: str, location: str) -> None:
    for label, df in [("ground_truth", sol_df), ("submission", sub_df)]:
        if "ID" not in getattr(df, "columns", []):
            raise_error(stage, location, f"{label} sem coluna ID", impact="1", examples=["ID"])
        dup = df["ID"].duplicated()
        try:
            has_dup = bool(dup.any())
        except Exception:
            has_dup = False
        if has_dup:
            examples = [str(x) for x in df.loc[dup, "ID"].head(8).tolist()]
            raise_error(stage, location, f"{label} com ID duplicado", impact=str(int(dup.sum())), examples=examples)
    sol_ids = set(str(x) for x in sol_df["ID"].tolist())
    sub_ids = set(str(x) for x in sub_df["ID"].tolist())
    missing = sorted(sol_ids - sub_ids)
    extra = sorted(sub_ids - sol_ids)
    if missing or extra:
        examples = [f"missing:{x}" for x in missing[:4]] + [f"extra:{x}" for x in extra[:4]]
        raise_error(stage, location, "chaves da submissao nao batem com ground_truth", impact=str(len(missing) + len(extra)), examples=examples)


def score_local_kaggle_official(
    *,
    ground_truth_path: Path,
    submission_path: Path,
    score_json_path: Path,
    report_path: Path,
    metric_path: Path | None = None,
    row_id_column_name: str = "ID",
) -> KaggleOfficialScoreResult:
    stage = "SCORE_LOCAL_KAGGLE_OFFICIAL"
    location = "src/rna3d_local/evaluation/kaggle_oracle.py:score_local_kaggle_official"
    pd = _require_pandas(stage=stage, location=location)
    default_metric = Path(__file__).resolve().parent / "kaggle_official" / "metric.py"
    metric_file = default_metric if metric_path is None else Path(metric_path).resolve()
    metric = _load_metric_module(metric_path=metric_file, stage=stage, location=location)
    scorer = getattr(metric, "score", None)
    if not callable(scorer):
        raise_error(stage, location, "metric.py oficial sem funcao score()", impact="1", examples=[str(metric_file)])
    if str(row_id_column_name) != "ID":
        raise_error(stage, location, "row_id_column_name invalido (esperado ID)", impact="1", examples=[str(row_id_column_name)])

    try:
        sol_df = pd.read_csv(str(ground_truth_path))
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "falha ao ler ground_truth CSV", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    try:
        sub_df = pd.read_csv(str(submission_path))
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "falha ao ler submission CSV", impact="1", examples=[f"{type(exc).__name__}:{exc}"])

    _validate_ids(sol_df=sol_df, sub_df=sub_df, stage=stage, location=location)
    try:
        score = float(scorer(sol_df, sub_df, row_id_column_name="ID"))
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "metric.py oficial falhou ao computar score", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    if score != score or score in {float("inf"), float("-inf")}:
        raise_error(stage, location, "metric.py retornou score nao-finito", impact="1", examples=[str(score)])

    write_json(
        score_json_path,
        {
            "created_utc": utc_now_iso(),
            "metric": "kaggle_official_metric",
            "score": float(score),
            "score_source": "kaggle_official_metric_py",
        },
    )
    write_json(
        report_path,
        {
            "created_utc": utc_now_iso(),
            "metric": "kaggle_official_metric",
            "score": float(score),
            "paths": {
                "ground_truth": str(ground_truth_path.resolve()),
                "submission": str(submission_path.resolve()),
                "metric_py": str(metric_file),
                "score_json": str(score_json_path.resolve()),
                "report": str(report_path.resolve()),
            },
        },
    )
    return KaggleOfficialScoreResult(score=float(score), score_json_path=score_json_path, report_path=report_path)
