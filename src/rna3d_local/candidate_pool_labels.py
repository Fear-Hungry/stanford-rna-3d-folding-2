from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from .bigdata import (
    DEFAULT_MAX_ROWS_IN_MEMORY,
    DEFAULT_MEMORY_BUDGET_MB,
    assert_memory_budget,
    assert_row_budget,
    collect_streaming,
)
from .candidate_pool_common import _utc_now
from .errors import raise_error
from .utils import sha256_file

LABEL_METHOD_TM_SCORE_USALIGN = "tm_score_usalign"
LABEL_METHOD_RMSD_KABSCH = "rmsd_kabsch"
LABEL_METHOD_CHOICES: tuple[str, ...] = (
    LABEL_METHOD_TM_SCORE_USALIGN,
    LABEL_METHOD_RMSD_KABSCH,
)
_TM_SCORE_INACTIVE_COORD = -1_000_001.0

def _extract_target_id_from_row_id(*, row_id: object, location: str) -> str:
    text = str(row_id).strip()
    if not text:
        raise_error("POOL", location, "ID vazio no arquivo de solucoes", impact="1", examples=[str(row_id)])
    if "_" not in text:
        raise_error("POOL", location, "ID com formato inesperado (esperado <target_id>_<resid>)", impact="1", examples=[text])
    target_id = text.rsplit("_", maxsplit=1)[0].strip()
    if not target_id:
        raise_error("POOL", location, "target_id extraido vazio do ID", impact="1", examples=[text])
    return target_id


def _normalize_label_method(*, label_method: str, location: str) -> str:
    method = str(label_method or "").strip().lower()
    if method not in LABEL_METHOD_CHOICES:
        raise_error(
            "POOL",
            location,
            "label_method invalido",
            impact="1",
            examples=[str(label_method), f"choices={','.join(LABEL_METHOD_CHOICES)}"],
        )
    return method


def _resolve_label_source_name(*, label_source_name: str | None, label_method: str, location: str) -> str:
    if label_source_name is not None and str(label_source_name).strip():
        return str(label_source_name).strip()
    if label_method == LABEL_METHOD_TM_SCORE_USALIGN:
        return "solution_tm_score_usalign"
    if label_method == LABEL_METHOD_RMSD_KABSCH:
        return "solution_rmsd_kabsch_inv1"
    raise_error("POOL", location, "label_method sem source padrao", impact="1", examples=[str(label_method)])
    raise AssertionError("unreachable")


def _extract_solution_coords(
    *,
    solution_path: Path,
    location: str,
    require_resname: bool,
) -> dict[str, dict[str, list]]:
    if not solution_path.exists():
        raise_error("POOL", location, "arquivo de solucao inexistente para labels", impact="1", examples=[str(solution_path)])

    suffix = solution_path.suffix.lower()
    if suffix == ".parquet":
        lf = pl.scan_parquet(solution_path)
    elif suffix == ".csv":
        lf = pl.scan_csv(solution_path)
    else:
        raise_error(
            "POOL",
            location,
            "formato de solucao invalido para labels (suportado: parquet/csv)",
            impact="1",
            examples=[str(solution_path)],
        )

    cols = set(lf.collect_schema().names())
    required = {"ID", "resid", "x_1", "y_1", "z_1"}
    if bool(require_resname):
        required.add("resname")
    missing = [c for c in required if c not in cols]
    if missing:
        raise_error(
            "POOL",
            location,
            "solucao sem coluna obrigatoria para label",
            impact=str(len(missing)),
            examples=missing[:8],
        )

    resname_expr = (
        pl.col("resname").cast(pl.Utf8).str.strip_chars().str.to_uppercase()
        if "resname" in cols
        else pl.lit(None, dtype=pl.Utf8)
    )
    rows = collect_streaming(
        lf=lf.select(
            pl.col("ID").cast(pl.Utf8),
            pl.col("resid").cast(pl.Int32),
            resname_expr.alias("resname"),
            pl.col("x_1").cast(pl.Float64),
            pl.col("y_1").cast(pl.Float64),
            pl.col("z_1").cast(pl.Float64),
        ),
        stage="POOL",
        location=location,
    )
    if rows.height == 0:
        raise_error("POOL", location, "solucao vazia para label", impact="0", examples=[str(solution_path)])

    rows = rows.with_columns(
        pl.col("ID").map_elements(
            lambda v: _extract_target_id_from_row_id(row_id=v, location=location),
            return_dtype=pl.Utf8,
        ).alias("target_id")
    )
    rows = rows.with_columns(pl.col("resid").alias("resid_idx"))
    grouped = rows.group_by("target_id").agg(
        pl.col("resid_idx").sort().alias("resids"),
        pl.col("resname").sort_by("resid_idx").alias("resnames"),
        pl.col("x_1").sort_by("resid_idx").alias("x_values"),
        pl.col("y_1").sort_by("resid_idx").alias("y_values"),
        pl.col("z_1").sort_by("resid_idx").alias("z_values"),
    )

    solution_map: dict[str, dict[str, list]] = {}
    invalid_examples: list[str] = []
    invalid_count = 0

    for row in grouped.iter_rows(named=True):
        tid = str(row["target_id"])
        resids_raw = row["resids"]
        resnames_raw = row["resnames"]
        xv = row["x_values"]
        yv = row["y_values"]
        zv = row["z_values"]
        if not (len(resids_raw) == len(resnames_raw) == len(xv) == len(yv) == len(zv)):
            invalid_count += 1
            if len(invalid_examples) < 8:
                invalid_examples.append(f"{tid}:shape_mismatch")
            continue
        if len(resids_raw) == 0 or len(xv) == 0 or len(yv) == 0 or len(zv) == 0:
            invalid_count += 1
            if len(invalid_examples) < 8:
                invalid_examples.append(f"{tid}:empty_target")
            continue
        resids = [int(v) for v in resids_raw]
        expected = list(range(1, len(resids) + 1))
        if resids != expected:
            invalid_count += 1
            if len(invalid_examples) < 8:
                invalid_examples.append(f"{tid}:non_contiguous_resid")
            continue
        resnames: list[str] = []
        for idx, raw_resname in enumerate(resnames_raw, start=1):
            text = "" if raw_resname is None else str(raw_resname).strip().upper()
            if bool(require_resname):
                if text not in {"A", "C", "G", "U"}:
                    invalid_count += 1
                    if len(invalid_examples) < 8:
                        invalid_examples.append(f"{tid}:{idx}:invalid_resname={text}")
                    text = ""
            if not text:
                text = "A"
            resnames.append(text)

        coords: list[tuple[float, float, float]] = []
        for idx, values in enumerate(zip(xv, yv, zv, strict=True), start=1):
            try:
                x = float(values[0])
                y = float(values[1])
                z = float(values[2])
            except (TypeError, ValueError):
                if len(invalid_examples) < 8:
                    invalid_examples.append(f"{tid}:{idx}:invalid_coord")
                continue
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                if len(invalid_examples) < 8:
                    invalid_examples.append(f"{tid}:{idx}:non_finite_coord")
                continue
            coords.append((x, y, z))
        if len(coords) != len(resids):
            invalid_count += 1
            if len(invalid_examples) < 8:
                invalid_examples.append(f"{tid}:coord_count_mismatch")
            continue
        solution_map[str(tid)] = {
            "coords": coords,
            "resids": resids,
            "resnames": resnames,
        }

    if invalid_count > 0:
        raise_error(
            "POOL",
            location,
            "solucao invalida para label",
            impact=str(invalid_count),
            examples=invalid_examples,
        )
    return solution_map


def _extract_candidate_coords(*, candidate: dict[str, Any], location: str) -> list[tuple[float, float, float]]:
    tid = str(candidate["target_id"])
    model_id = int(candidate["model_id"])
    raw_coords = candidate.get("coords")
    if raw_coords is None:
        raise_error("POOL", location, "coords ausentes no candidato", impact="1", examples=[f"{tid}:{model_id}"])
    coords: list[tuple[float, float, float]] = []
    for idx, values in enumerate(raw_coords, start=1):
        if values is None or len(values) != 3:
            raise_error(
                "POOL",
                location,
                "coords invalidas no candidato",
                impact="1",
                examples=[f"{tid}:{model_id}:resid={idx}"],
            )
        x_coord = float(values[0])
        y_coord = float(values[1])
        z_coord = float(values[2])
        if not (math.isfinite(x_coord) and math.isfinite(y_coord) and math.isfinite(z_coord)):
            raise_error(
                "POOL",
                location,
                "coords nao finitas no candidato",
                impact="1",
                examples=[f"{tid}:{model_id}:resid={idx}"],
            )
        coords.append((x_coord, y_coord, z_coord))
    return coords


def _extract_candidate_resids(*, candidate: dict[str, Any], location: str) -> list[int]:
    tid = str(candidate["target_id"])
    model_id = int(candidate["model_id"])
    raw_resids = candidate.get("resids")
    if raw_resids is None:
        raise_error("POOL", location, "resids ausentes no candidato", impact="1", examples=[f"{tid}:{model_id}"])
    resids = [int(value) for value in raw_resids]
    expected = list(range(1, len(resids) + 1))
    if resids != expected:
        raise_error(
            "POOL",
            location,
            "resids do candidato devem ser contiguos iniciando em 1",
            impact="1",
            examples=[f"{tid}:{model_id}"],
        )
    return resids


def _extract_candidate_resnames(*, candidate: dict[str, Any], expected_len: int, location: str) -> list[str]:
    tid = str(candidate["target_id"])
    model_id = int(candidate["model_id"])
    raw_resnames = candidate.get("resnames")
    if raw_resnames is None:
        raise_error("POOL", location, "resnames ausentes no candidato", impact="1", examples=[f"{tid}:{model_id}"])
    if len(raw_resnames) != int(expected_len):
        raise_error(
            "POOL",
            location,
            "resnames com tamanho divergente do numero de residuos",
            impact=f"candidate={model_id} got={len(raw_resnames)} expected={expected_len}",
            examples=[tid],
        )
    resnames: list[str] = []
    for idx, raw_resname in enumerate(raw_resnames, start=1):
        text = str(raw_resname).strip().upper()
        if text not in {"A", "C", "G", "U"}:
            raise_error(
                "POOL",
                location,
                "resname invalido no candidato para TM-score",
                impact="1",
                examples=[f"{tid}:{model_id}:resid={idx}:resname={text}"],
            )
        resnames.append(text)
    return resnames


def _kabsch_rmsd(*, coords: list[tuple[float, float, float]], refs: list[tuple[float, float, float]], location: str, target_id: str, model_id: int) -> float:
    if len(coords) != len(refs):
        raise_error(
            "POOL",
            location,
            "comprimento de residuos do candidato divergente da solucao",
            impact=f"target={target_id} candidate={model_id} cand={len(coords)} sol={len(refs)}",
            examples=[f"{target_id}:{model_id}"],
        )
    if len(coords) <= 0:
        raise_error("POOL", location, "candidato sem residuos para calcular label", impact="0", examples=[f"{target_id}:{model_id}"])

    pred_matrix = np.asarray(coords, dtype=np.float64)
    ref_matrix = np.asarray(refs, dtype=np.float64)
    if pred_matrix.shape != ref_matrix.shape or pred_matrix.shape[1] != 3:
        raise_error(
            "POOL",
            location,
            "shape invalido para Kabsch",
            impact="1",
            examples=[f"{target_id}:{model_id}:pred={pred_matrix.shape}:ref={ref_matrix.shape}"],
        )

    pred_centered = pred_matrix - pred_matrix.mean(axis=0)
    ref_centered = ref_matrix - ref_matrix.mean(axis=0)
    covariance = pred_centered.T @ ref_centered
    try:
        left_vectors, _singular_values, right_vectors_t = np.linalg.svd(covariance)
    except Exception as error:  # noqa: BLE001
        raise_error(
            "POOL",
            location,
            "falha na decomposicao SVD para Kabsch",
            impact="1",
            examples=[f"{type(error).__name__}:{error}", f"{target_id}:{model_id}"],
        )
    rotation = left_vectors @ right_vectors_t
    determinant = float(np.linalg.det(rotation))
    if determinant < 0.0:
        left_vectors[:, -1] *= -1.0
        rotation = left_vectors @ right_vectors_t

    aligned = pred_centered @ rotation
    diff = aligned - ref_centered
    squared = np.sum(diff * diff, axis=1)
    rmsd = float(np.sqrt(float(np.mean(squared))))
    if not math.isfinite(rmsd):
        raise_error(
            "POOL",
            location,
            "rmsd_kabsch nao finito para candidato",
            impact=f"{target_id}:{model_id}",
            examples=[f"rmsd={rmsd}"],
        )
    return rmsd


def _load_tm_metric(*, metric_py_path: Path, location: str) -> Any:
    if not metric_py_path.exists():
        raise_error("POOL", location, "metric.py ausente para label TM-score", impact="1", examples=[str(metric_py_path)])
    spec = importlib.util.spec_from_file_location("tm_score_metric_labeling", metric_py_path)
    if spec is None or spec.loader is None:
        raise_error("POOL", location, "falha ao carregar metric.py para labels", impact="1", examples=[str(metric_py_path)])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "score"):
        raise_error("POOL", location, "metric.py nao exporta score() para labels", impact="1", examples=[str(metric_py_path)])
    return module


def _build_metric_frame(
    *,
    target_id: str,
    resids: list[int],
    resnames: list[str],
    coords: list[tuple[float, float, float]],
    n_models: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for resid, resname, coord in zip(resids, resnames, coords, strict=True):
        row: dict[str, float | int | str] = {
            "ID": f"{target_id}_{int(resid)}",
            "resid": int(resid),
            "resname": str(resname),
        }
        for model_id in range(1, int(n_models) + 1):
            x_col = f"x_{model_id}"
            y_col = f"y_{model_id}"
            z_col = f"z_{model_id}"
            if model_id == 1:
                row[x_col] = float(coord[0])
                row[y_col] = float(coord[1])
                row[z_col] = float(coord[2])
            else:
                row[x_col] = float(_TM_SCORE_INACTIVE_COORD)
                row[y_col] = float(_TM_SCORE_INACTIVE_COORD)
                row[z_col] = float(_TM_SCORE_INACTIVE_COORD)
        rows.append(row)
    return pd.DataFrame(rows)


def _solve_candidate_pool_label_rmsd_kabsch(*, candidate: dict[str, Any], reference: dict[str, dict[str, list]], location: str) -> tuple[float, float]:
    tid = str(candidate["target_id"])
    if tid not in reference:
        raise_error("POOL", location, "target ausente na solucao de treino", impact="1", examples=[tid])

    coords = _extract_candidate_coords(candidate=candidate, location=location)
    refs = [tuple(value) for value in reference[tid]["coords"]]
    rmsd = _kabsch_rmsd(
        coords=coords,
        refs=refs,
        location=location,
        target_id=tid,
        model_id=int(candidate["model_id"]),
    )
    label = 1.0 / (1.0 + float(rmsd))
    return float(label), float(rmsd)


def _solve_candidate_pool_label_tm_score_usalign(
    *,
    candidate: dict[str, Any],
    reference: dict[str, dict[str, list]],
    reference_metric_frames: dict[str, pd.DataFrame],
    metric_module: Any,
    usalign_bin_path: Path,
    location: str,
) -> tuple[float, float]:
    tid = str(candidate["target_id"])
    model_id = int(candidate["model_id"])
    if tid not in reference:
        raise_error("POOL", location, "target ausente na solucao de treino", impact="1", examples=[tid])
    if tid not in reference_metric_frames:
        raise_error("POOL", location, "target sem frame nativo para TM-score", impact="1", examples=[tid])

    coords = _extract_candidate_coords(candidate=candidate, location=location)
    resids = _extract_candidate_resids(candidate=candidate, location=location)
    resnames = _extract_candidate_resnames(candidate=candidate, expected_len=len(coords), location=location)
    refs = [tuple(value) for value in reference[tid]["coords"]]
    rmsd = _kabsch_rmsd(coords=coords, refs=refs, location=location, target_id=tid, model_id=model_id)

    pred_frame = _build_metric_frame(
        target_id=tid,
        resids=resids,
        resnames=resnames,
        coords=coords,
        n_models=5,
    )
    try:
        tm_score = float(
            metric_module.score(
                reference_metric_frames[tid],
                pred_frame,
                "ID",
                usalign_bin_hint=str(usalign_bin_path),
            )
        )
    except Exception as error:  # noqa: BLE001
        raise_error(
            "POOL",
            location,
            "falha ao calcular TM-score/USalign para label",
            impact="1",
            examples=[f"{tid}:{model_id}", f"{type(error).__name__}:{error}"],
        )
    if not math.isfinite(tm_score):
        raise_error(
            "POOL",
            location,
            "tm_score_usalign nao finito para candidato",
            impact="1",
            examples=[f"{tid}:{model_id}", f"tm_score={tm_score}"],
        )
    if tm_score < 0.0 or tm_score > 1.0:
        raise_error(
            "POOL",
            location,
            "tm_score_usalign fora de [0,1]",
            impact="1",
            examples=[f"{tid}:{model_id}", f"tm_score={tm_score}"],
        )
    return float(tm_score), float(rmsd)


def add_labels_to_candidate_pool(
    *,
    candidate_pool_path: Path,
    solution_path: Path,
    out_path: Path,
    label_col: str = "label",
    label_source_col: str = "label_source",
    label_source_name: str | None = None,
    label_method: str = LABEL_METHOD_TM_SCORE_USALIGN,
    metric_py_path: Path | None = None,
    usalign_bin_path: Path | None = None,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    max_rows_in_memory: int = DEFAULT_MAX_ROWS_IN_MEMORY,
) -> tuple[Path, Path]:
    location = "src/rna3d_local/candidate_pool_labels.py:add_labels_to_candidate_pool"
    assert_memory_budget(stage="POOL", location=location, budget_mb=memory_budget_mb)
    label_method_norm = _normalize_label_method(label_method=label_method, location=location)

    if label_col == label_source_col:
        raise_error("POOL", location, "label_col e label_source_col devem ser diferentes", impact="1", examples=[label_col])
    label_source_name_final = _resolve_label_source_name(
        label_source_name=label_source_name,
        label_method=label_method_norm,
        location=location,
    )

    if not candidate_pool_path.exists():
        raise_error("POOL", location, "candidate_pool ausente", impact="1", examples=[str(candidate_pool_path)])
    if label_source_col == "label":
        raise_error("POOL", location, "label_source_col reservado para metadado de label", impact="1", examples=[label_source_col])

    pool = collect_streaming(
        lf=pl.scan_parquet(candidate_pool_path),
        stage="POOL",
        location=location,
    )
    required_pool = {"target_id", "model_id", "resid_count", "coords", "resids"}
    missing_pool = [c for c in required_pool if c not in set(pool.columns)]
    if missing_pool:
        raise_error("POOL", location, "candidate_pool sem coluna obrigatoria", impact=str(len(missing_pool)), examples=missing_pool[:8])
    if label_col in pool.columns:
        raise_error(
            "POOL",
            location,
            "candidate_pool ja possui coluna de label",
            impact="1",
            examples=[str(candidate_pool_path)],
        )
    if label_source_col in pool.columns:
        raise_error(
            "POOL",
            location,
            "candidate_pool ja possui coluna de label_source",
            impact="1",
            examples=[str(candidate_pool_path)],
        )
    assert_row_budget(stage="POOL", location=location, rows=int(pool.height), max_rows_in_memory=max_rows_in_memory, label="candidate_pool")
    if pool.height == 0:
        raise_error("POOL", location, "candidate_pool vazio", impact="0", examples=[str(candidate_pool_path)])

    if label_method_norm == LABEL_METHOD_TM_SCORE_USALIGN and "resnames" not in set(pool.columns):
        raise_error(
            "POOL",
            location,
            "candidate_pool sem coluna obrigatoria para label tm_score_usalign",
            impact="1",
            examples=["resnames"],
        )

    solution_map = _extract_solution_coords(
        solution_path=solution_path,
        location=location,
        require_resname=(label_method_norm == LABEL_METHOD_TM_SCORE_USALIGN),
    )
    pool_targets = set(pool.get_column("target_id").to_list())
    missing_targets = sorted([t for t in pool_targets if str(t) not in solution_map])
    if missing_targets:
        raise_error(
            "POOL",
            location,
            "targets do candidate_pool ausentes na solucao para rotulagem",
            impact=str(len(missing_targets)),
            examples=missing_targets[:8],
        )

    metric_module = None
    metric_py_final: Path | None = None
    usalign_bin_final: Path | None = None
    reference_metric_frames: dict[str, pd.DataFrame] = {}
    if label_method_norm == LABEL_METHOD_TM_SCORE_USALIGN:
        if metric_py_path is None:
            raise_error("POOL", location, "metric_py_path obrigatorio para label tm_score_usalign", impact="1", examples=["metric_py_path"])
        if usalign_bin_path is None:
            raise_error("POOL", location, "usalign_bin_path obrigatorio para label tm_score_usalign", impact="1", examples=["usalign_bin_path"])
        metric_py_final = metric_py_path.resolve()
        usalign_bin_final = usalign_bin_path.resolve()
        if not metric_py_final.exists():
            raise_error("POOL", location, "metric.py ausente para label tm_score_usalign", impact="1", examples=[str(metric_py_final)])
        if not usalign_bin_final.exists():
            raise_error("POOL", location, "USalign ausente para label tm_score_usalign", impact="1", examples=[str(usalign_bin_final)])
        metric_module = _load_tm_metric(metric_py_path=metric_py_final, location=location)
        for target_id, ref_values in solution_map.items():
            reference_metric_frames[target_id] = _build_metric_frame(
                target_id=str(target_id),
                resids=[int(value) for value in ref_values["resids"]],
                resnames=[str(value) for value in ref_values["resnames"]],
                coords=[tuple(value) for value in ref_values["coords"]],
                n_models=40,
            )

    labels: list[float] = []
    sources: list[str] = []
    rmsd_values: list[float] = []

    temp_work_dir: Path | None = None
    old_cwd = Path.cwd()
    if label_method_norm == LABEL_METHOD_TM_SCORE_USALIGN:
        temp_work_dir = Path(tempfile.mkdtemp(prefix="rna3d_pool_labels_"))
        os.chdir(temp_work_dir)
    try:
        for row in pool.iter_rows(named=True):
            if label_method_norm == LABEL_METHOD_TM_SCORE_USALIGN:
                assert metric_module is not None
                assert usalign_bin_final is not None
                label, rmsd = _solve_candidate_pool_label_tm_score_usalign(
                    candidate=row,
                    reference=solution_map,
                    reference_metric_frames=reference_metric_frames,
                    metric_module=metric_module,
                    usalign_bin_path=usalign_bin_final,
                    location=location,
                )
            else:
                label, rmsd = _solve_candidate_pool_label_rmsd_kabsch(
                    candidate=row,
                    reference=solution_map,
                    location=location,
                )
            labels.append(float(label))
            rmsd_values.append(float(rmsd))
            sources.append(str(label_source_name_final))
    finally:
        if temp_work_dir is not None:
            os.chdir(old_cwd)
            shutil.rmtree(temp_work_dir, ignore_errors=True)

    out_df = pool.with_columns(
        pl.Series(name=label_col, values=labels, dtype=pl.Float64),
        pl.Series(name=label_source_col, values=sources, dtype=pl.Utf8),
    )
    out_df = out_df.sort(["target_id", "source", "model_id"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    out_df.write_parquet(out_path)
    assert_memory_budget(stage="POOL", location=location, budget_mb=memory_budget_mb)

    manifest = {
        "created_utc": _utc_now(),
        "paths": {
            "candidate_pool": str(candidate_pool_path),
            "solution": str(solution_path),
            "labeled_candidate_pool": str(out_path),
        },
        "label": {
            "label_col": str(label_col),
            "label_source_col": str(label_source_col),
            "label_source": str(label_source_name_final),
            "label_method": str(label_method_norm),
            "labeled_targets": int(len(solution_map)),
            "metric_py": None if metric_py_final is None else str(metric_py_final),
            "usalign_bin": None if usalign_bin_final is None else str(usalign_bin_final),
        },
        "stats": {
            "rows": int(out_df.height),
            "targets": int(out_df.select("target_id").n_unique()),
            "features_with_label": int(len(out_df.columns)),
            "label_min": float(min(labels)),
            "label_max": float(max(labels)),
            "rmsd_kabsch_mean": float(sum(rmsd_values) / float(len(rmsd_values))),
        },
        "sha256": {
            "labeled_candidate_pool.parquet": sha256_file(out_path),
            **({} if metric_py_final is None else {"metric.py": sha256_file(metric_py_final)}),
            **({} if usalign_bin_final is None else {"USalign": sha256_file(usalign_bin_final)}),
        },
    }
    manifest_path = out_path.parent / "candidate_pool_labels_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path, manifest_path




__all__ = [
    "LABEL_METHOD_CHOICES",
    "LABEL_METHOD_RMSD_KABSCH",
    "LABEL_METHOD_TM_SCORE_USALIGN",
    "add_labels_to_candidate_pool",
]
