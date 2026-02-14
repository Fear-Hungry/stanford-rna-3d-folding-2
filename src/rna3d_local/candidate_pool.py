from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from .bigdata import (
    DEFAULT_MAX_ROWS_IN_MEMORY,
    DEFAULT_MEMORY_BUDGET_MB,
    TableReadConfig,
    assert_memory_budget,
    assert_row_budget,
    collect_streaming,
    scan_table,
)
from .compute_backend import resolve_compute_backend
from .errors import raise_error
from .qa_ranker import candidate_geometry_features
from .utils import sha256_file

CANDIDATE_POOL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "target_id",
    "ID",
    "resid",
    "resname",
    "model_id",
    "x",
    "y",
    "z",
    "coverage",
    "similarity",
    "template_uid",
)

CANDIDATE_POOL_DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    "coverage",
    "similarity",
    "mapped_ratio",
    "match_ratio",
    "mismatch_ratio",
    "chem_compatible_ratio",
    "path_length",
    "step_mean",
    "step_std",
    "radius_gyr",
    "gap_open_score",
    "gap_extend_score",
    "qa_score_base",
    "resid_count",
    "dist_off_1",
    "dist_off_2",
    "dist_off_4",
    "dist_off_8",
    "dist_off_16",
    "dist_off_32",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        x = float(value)
        if not (x == x):  # NaN
            return float(default)
        if x in (float("inf"), float("-inf")):
            return float(default)
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _mean_distance_for_offset(*, coords: list[tuple[float, float, float]], offset: int) -> float:
    n = len(coords)
    if n <= int(offset):
        return 0.0
    total = 0.0
    count = 0
    for i in range(0, n - int(offset)):
        a = coords[i]
        b = coords[i + int(offset)]
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        dz = float(a[2]) - float(b[2])
        total += float((dx * dx + dy * dy + dz * dz) ** 0.5)
        count += 1
    if count <= 0:
        return 0.0
    return float(total / float(count))


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


def _extract_solution_coords(*, solution_path: Path, location: str) -> dict[str, list[tuple[float, float, float]]]:
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
    missing = [c for c in required if c not in cols]
    if missing:
        raise_error(
            "POOL",
            location,
            "solucao sem coluna obrigatoria para label",
            impact=str(len(missing)),
            examples=missing[:8],
        )

    rows = collect_streaming(
        lf=lf.select(
            pl.col("ID").cast(pl.Utf8),
            pl.col("resid").cast(pl.Int32),
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
        pl.col("x_1").sort_by("resid_idx").alias("x_values"),
        pl.col("y_1").sort_by("resid_idx").alias("y_values"),
        pl.col("z_1").sort_by("resid_idx").alias("z_values"),
    )

    solution_map: dict[str, list[tuple[float, float, float]]] = {}
    invalid_examples: list[str] = []
    invalid_count = 0

    for row in grouped.iter_rows(named=True):
        tid = str(row["target_id"])
        resids_raw = row["resids"]
        xv = row["x_values"]
        yv = row["y_values"]
        zv = row["z_values"]
        if not (len(xv) == len(yv) == len(zv)):
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
        solution_map[str(tid)] = coords

    if invalid_count > 0:
        raise_error(
            "POOL",
            location,
            "solucao invalida para label",
            impact=str(invalid_count),
            examples=invalid_examples,
        )
    return solution_map


def _solve_candidate_pool_label(*, candidate: dict[str, Any], reference: dict[str, list[tuple[float, float, float]]], location: str) -> tuple[float, float]:
    tid = str(candidate["target_id"])
    if tid not in reference:
        raise_error("POOL", location, "target ausente na solucao de treino", impact="1", examples=[tid])

    coords = candidate["coords"]
    refs = reference[tid]
    if coords is None:
        raise_error("POOL", location, "coords ausentes no candidato", impact="1", examples=[f"{tid}:{int(candidate['model_id'])}"])
    if len(coords) != len(refs):
        raise_error(
            "POOL",
            location,
            "comprimento de residuos do candidato divergente da solucao",
            impact=f"target={tid} candidate={int(candidate['model_id'])} cand={len(coords)} sol={len(refs)}",
            examples=[f"{tid}:{int(candidate['model_id'])}"],
        )

    acc = 0.0
    for i, c in enumerate(coords):
        px = float(c[0])
        py = float(c[1])
        pz = float(c[2])
        rx, ry, rz = refs[i]
        dx = px - rx
        dy = py - ry
        dz = pz - rz
        acc += float(dx * dx + dy * dy + dz * dz)
    rmsd = math.sqrt(acc / max(1, len(coords)))
    if not math.isfinite(rmsd):
        raise_error(
            "POOL",
            location,
            "label nao finito para candidato",
            impact=f"{tid}:{int(candidate['model_id'])}",
            examples=[f"rmsd={rmsd}"],
        )
    label = 1.0 / (1.0 + float(rmsd))
    return float(label), float(rmsd)


def add_labels_to_candidate_pool(
    *,
    candidate_pool_path: Path,
    solution_path: Path,
    out_path: Path,
    label_col: str = "label",
    label_source_col: str = "label_source",
    label_source_name: str = "solution_rmsd_inv1",
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    max_rows_in_memory: int = DEFAULT_MAX_ROWS_IN_MEMORY,
) -> tuple[Path, Path]:
    location = "src/rna3d_local/candidate_pool.py:add_labels_to_candidate_pool"
    assert_memory_budget(stage="POOL", location=location, budget_mb=memory_budget_mb)

    if label_col == label_source_col:
        raise_error("POOL", location, "label_col e label_source_col devem ser diferentes", impact="1", examples=[label_col])
    if not str(label_source_name).strip():
        raise_error("POOL", location, "label_source_name vazio", impact="1", examples=[str(label_source_name)])

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

    solution_map = _extract_solution_coords(solution_path=solution_path, location=location)
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

    labels: list[float] = []
    sources: list[str] = []
    for row in pool.iter_rows(named=True):
        label, _ = _solve_candidate_pool_label(candidate=row, reference=solution_map, location=location)
        labels.append(label)
        sources.append(str(label_source_name))

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
            "label_source": str(label_source_name),
            "labeled_targets": int(len(solution_map)),
        },
        "stats": {
            "rows": int(out_df.height),
            "targets": int(out_df.select("target_id").n_unique()),
            "features_with_label": int(len(out_df.columns)),
        },
        "sha256": {
            "labeled_candidate_pool.parquet": sha256_file(out_path),
        },
    }
    manifest_path = out_path.parent / "candidate_pool_labels_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path, manifest_path


def _parse_prediction_entry(*, raw: str, location: str) -> tuple[str | None, Path]:
    token = str(raw).strip()
    if not token:
        raise_error("POOL", location, "entrada de predicao vazia", impact="1", examples=[str(raw)])
    if "=" in token:
        left, right = token.split("=", 1)
        source = str(left).strip()
        path_raw = str(right).strip()
        if not source:
            raise_error("POOL", location, "source vazio na entrada source=path", impact="1", examples=[token])
        if not path_raw:
            raise_error("POOL", location, "path vazio na entrada source=path", impact="1", examples=[token])
        return source, Path(path_raw)
    return None, Path(token)


def _normalize_prediction_lazy(
    *,
    source_override: str | None,
    path: Path,
    location: str,
) -> pl.LazyFrame:
    lf = scan_table(config=TableReadConfig(path=path, stage="POOL", location=location, columns=None))
    cols = set(lf.collect_schema().names())
    missing = [c for c in CANDIDATE_POOL_REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise_error(
            "POOL",
            location,
            "predicoes sem coluna obrigatoria para candidate pool",
            impact=f"path={path} missing={len(missing)}",
            examples=missing[:8],
        )
    if source_override is None and "branch" not in cols:
        raise_error(
            "POOL",
            location,
            "coluna branch obrigatoria quando source nao e informado",
            impact="1",
            examples=[str(path)],
        )

    source_expr = pl.lit(str(source_override)).alias("source") if source_override is not None else pl.col("branch").cast(pl.Utf8).alias("source")

    def _opt_float(name: str, default: float) -> pl.Expr:
        if name in cols:
            return pl.col(name).cast(pl.Float64).alias(name)
        return pl.lit(float(default), dtype=pl.Float64).alias(name)

    def _opt_int(name: str, default: int) -> pl.Expr:
        if name in cols:
            return pl.col(name).cast(pl.Int64).alias(name)
        return pl.lit(int(default), dtype=pl.Int64).alias(name)

    normalized = lf.select(
        pl.col("target_id").cast(pl.Utf8).alias("target_id"),
        source_expr,
        pl.col("model_id").cast(pl.Int64).alias("model_id"),
        pl.col("template_uid").cast(pl.Utf8).alias("template_uid"),
        pl.col("ID").cast(pl.Utf8).alias("ID"),
        pl.col("resid").cast(pl.Int64).alias("resid"),
        pl.col("resname").cast(pl.Utf8).alias("resname"),
        pl.col("x").cast(pl.Float64).alias("x"),
        pl.col("y").cast(pl.Float64).alias("y"),
        pl.col("z").cast(pl.Float64).alias("z"),
        pl.col("coverage").cast(pl.Float64).alias("coverage"),
        pl.col("similarity").cast(pl.Float64).alias("similarity"),
        _opt_int("mapped_count", 0),
        _opt_int("match_count", 0),
        _opt_int("mismatch_count", 0),
        _opt_int("chem_compatible_count", 0),
        _opt_float("gap_open_score", 0.0),
        _opt_float("gap_extend_score", 0.0),
        _opt_float("qa_score", 0.0).alias("qa_score_base"),
    )
    return normalized


def parse_prediction_entries(*, raw_entries: list[str], repo_root: Path, location: str) -> list[tuple[str | None, Path]]:
    if not raw_entries:
        raise_error("POOL", location, "nenhuma entrada de predicao informada", impact="0", examples=[])
    parsed: list[tuple[str | None, Path]] = []
    for raw in raw_entries:
        source, path_rel = _parse_prediction_entry(raw=raw, location=location)
        path = path_rel if path_rel.is_absolute() else (repo_root / path_rel).resolve()
        if not path.exists():
            raise_error("POOL", location, "arquivo de predicoes nao encontrado", impact="1", examples=[str(path)])
        parsed.append((source, path))
    return parsed


def build_candidate_pool_from_predictions(
    *,
    repo_root: Path,
    prediction_entries: list[tuple[str | None, Path]],
    out_path: Path,
    compute_backend: str = "auto",
    gpu_memory_budget_mb: int = 12_288,
    gpu_precision: str = "fp32",
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    max_rows_in_memory: int = DEFAULT_MAX_ROWS_IN_MEMORY,
) -> tuple[Path, Path]:
    location = "src/rna3d_local/candidate_pool.py:build_candidate_pool_from_predictions"
    assert_memory_budget(stage="POOL", location=location, budget_mb=memory_budget_mb)
    backend = resolve_compute_backend(
        requested=str(compute_backend),
        precision=str(gpu_precision),
        gpu_memory_budget_mb=int(gpu_memory_budget_mb),
        stage="POOL",
        location=location,
    )

    if len(prediction_entries) == 0:
        raise_error("POOL", location, "prediction_entries vazio", impact="0", examples=[])

    normalized_lfs: list[pl.LazyFrame] = []
    for source_override, path in prediction_entries:
        normalized_lfs.append(_normalize_prediction_lazy(source_override=source_override, path=path, location=location))
    all_lf = pl.concat(normalized_lfs, how="vertical_relaxed")

    grouped = collect_streaming(
        lf=all_lf.group_by("target_id", "source", "model_id", "template_uid")
        .agg(
            pl.len().alias("resid_count"),
            pl.col("resid").n_unique().alias("resid_unique_count"),
            pl.col("resid").sort().alias("resids"),
            pl.col("resname").sort_by("resid").alias("resnames"),
            pl.col("x").sort_by("resid").alias("x_values"),
            pl.col("y").sort_by("resid").alias("y_values"),
            pl.col("z").sort_by("resid").alias("z_values"),
            pl.col("coverage").mean().alias("coverage"),
            pl.col("similarity").mean().alias("similarity"),
            pl.col("mapped_count").mean().alias("mapped_count"),
            pl.col("match_count").mean().alias("match_count"),
            pl.col("mismatch_count").mean().alias("mismatch_count"),
            pl.col("chem_compatible_count").mean().alias("chem_compatible_count"),
            pl.col("gap_open_score").mean().alias("gap_open_score"),
            pl.col("gap_extend_score").mean().alias("gap_extend_score"),
            pl.col("qa_score_base").mean().alias("qa_score_base"),
        )
        .sort(["target_id", "source", "model_id"]),
        stage="POOL",
        location=location,
    )
    assert_row_budget(
        stage="POOL",
        location=location,
        rows=int(grouped.height),
        max_rows_in_memory=max_rows_in_memory,
        label="candidate_groups",
    )

    if grouped.height == 0:
        raise_error("POOL", location, "nenhum candidato agregado a partir das predicoes", impact="0", examples=[])

    rows_out: list[dict] = []
    invalid_count = 0
    invalid_examples: list[str] = []
    for row in grouped.iter_rows(named=True):
        tid = str(row["target_id"])
        source = str(row["source"])
        model_id = int(row["model_id"])
        template_uid = str(row["template_uid"])
        resid_count = int(row["resid_count"])
        resid_unique = int(row["resid_unique_count"])
        if resid_count <= 0 or resid_unique <= 0:
            invalid_count += 1
            if len(invalid_examples) < 8:
                invalid_examples.append(f"{tid}:{source}:{model_id}:empty")
            continue
        if resid_count != resid_unique:
            invalid_count += 1
            if len(invalid_examples) < 8:
                invalid_examples.append(f"{tid}:{source}:{model_id}:dup_resid")
            continue

        resids = [int(v) for v in row["resids"]]
        expected = list(range(1, resid_count + 1))
        if resids != expected:
            invalid_count += 1
            if len(invalid_examples) < 8:
                invalid_examples.append(f"{tid}:{source}:{model_id}:non_contiguous_resid")
            continue

        resnames = [str(v) for v in row["resnames"]]
        xv = [float(v) for v in row["x_values"]]
        yv = [float(v) for v in row["y_values"]]
        zv = [float(v) for v in row["z_values"]]
        if not (len(resnames) == len(xv) == len(yv) == len(zv) == resid_count):
            invalid_count += 1
            if len(invalid_examples) < 8:
                invalid_examples.append(f"{tid}:{source}:{model_id}:shape_mismatch")
            continue
        coords_tuple = [(float(x), float(y), float(z)) for x, y, z in zip(xv, yv, zv, strict=True)]
        coords = [[float(x), float(y), float(z)] for x, y, z in coords_tuple]

        coverage = max(0.0, min(1.0, _safe_float(row["coverage"], default=0.0)))
        similarity = max(0.0, min(1.0, _safe_float(row["similarity"], default=0.0)))
        mapped_count = int(round(_safe_float(row["mapped_count"], default=float(coverage * float(resid_count)))))
        if mapped_count < 0:
            mapped_count = 0
        if mapped_count > resid_count:
            mapped_count = resid_count
        match_count = int(round(_safe_float(row["match_count"], default=float(similarity * float(mapped_count)))))
        if match_count < 0:
            match_count = 0
        if match_count > mapped_count:
            match_count = mapped_count
        mismatch_count = int(round(_safe_float(row["mismatch_count"], default=float(mapped_count - match_count))))
        if mismatch_count < 0:
            mismatch_count = 0
        if mismatch_count > mapped_count:
            mismatch_count = mapped_count
        chem_compatible_count = int(round(_safe_float(row["chem_compatible_count"], default=0.0)))
        if chem_compatible_count < 0:
            chem_compatible_count = 0
        if chem_compatible_count > mismatch_count:
            chem_compatible_count = mismatch_count

        geom = candidate_geometry_features(coords=coords_tuple, location=location)
        mapped_ratio = float(mapped_count) / float(resid_count)
        match_ratio = float(match_count) / float(resid_count)
        mismatch_ratio = float(mismatch_count) / float(resid_count)
        chem_ratio = float(chem_compatible_count) / float(resid_count)
        candidate_id = f"{source}:model_{model_id}"

        rows_out.append(
            {
                "target_id": tid,
                "source": source,
                "model_id": int(model_id),
                "candidate_id": candidate_id,
                "template_uid": template_uid,
                "resid_count": int(resid_count),
                "resids": resids,
                "resnames": resnames,
                "sequence": "".join(resnames),
                "coords": coords,
                "coverage": float(coverage),
                "similarity": float(similarity),
                "mapped_count": int(mapped_count),
                "match_count": int(match_count),
                "mismatch_count": int(mismatch_count),
                "chem_compatible_count": int(chem_compatible_count),
                "mapped_ratio": float(mapped_ratio),
                "match_ratio": float(match_ratio),
                "mismatch_ratio": float(mismatch_ratio),
                "chem_compatible_ratio": float(chem_ratio),
                "path_length": float(geom["path_length"]),
                "step_mean": float(geom["step_mean"]),
                "step_std": float(geom["step_std"]),
                "radius_gyr": float(geom["radius_gyr"]),
                "dist_off_1": _mean_distance_for_offset(coords=coords_tuple, offset=1),
                "dist_off_2": _mean_distance_for_offset(coords=coords_tuple, offset=2),
                "dist_off_4": _mean_distance_for_offset(coords=coords_tuple, offset=4),
                "dist_off_8": _mean_distance_for_offset(coords=coords_tuple, offset=8),
                "dist_off_16": _mean_distance_for_offset(coords=coords_tuple, offset=16),
                "dist_off_32": _mean_distance_for_offset(coords=coords_tuple, offset=32),
                "gap_open_score": _safe_float(row["gap_open_score"], default=0.0),
                "gap_extend_score": _safe_float(row["gap_extend_score"], default=0.0),
                "qa_score_base": _safe_float(row["qa_score_base"], default=0.0),
            }
        )

    if invalid_count > 0:
        raise_error(
            "POOL",
            location,
            "candidatos invalidos durante agregacao do candidate pool",
            impact=str(invalid_count),
            examples=invalid_examples,
        )
    if len(rows_out) == 0:
        raise_error("POOL", location, "candidate pool vazio apos validacao", impact="0", examples=[])

    out_df = pl.DataFrame(rows_out).sort(["target_id", "source", "model_id"])
    dup = (
        out_df.group_by("target_id", "candidate_id")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
        .select("target_id", "candidate_id")
        .head(8)
    )
    if dup.height > 0:
        raise_error(
            "POOL",
            location,
            "candidate_id duplicado por target no pool final",
            impact=str(int(dup.height)),
            examples=[f"{a}:{b}" for a, b in dup.iter_rows()],
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out_path)
    assert_memory_budget(stage="POOL", location=location, budget_mb=memory_budget_mb)

    manifest = {
        "created_utc": _utc_now(),
        "paths": {
            "candidate_pool": _rel(out_path, repo_root),
            "prediction_sources": [
                {
                    "source": source if source is not None else "branch",
                    "path": _rel(path, repo_root),
                }
                for source, path in prediction_entries
            ],
        },
        "stats": {
            "rows": int(out_df.height),
            "targets": int(out_df.select("target_id").n_unique()),
            "sources": int(out_df.select("source").n_unique()),
        },
        "compute": backend.to_manifest_dict(),
        "feature_names": list(CANDIDATE_POOL_DEFAULT_FEATURE_NAMES),
        "sha256": {"candidate_pool.parquet": sha256_file(out_path)},
    }
    manifest_path = out_path.parent / "candidate_pool_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path, manifest_path


__all__ = [
    "CANDIDATE_POOL_DEFAULT_FEATURE_NAMES",
    "CANDIDATE_POOL_REQUIRED_COLUMNS",
    "build_candidate_pool_from_predictions",
    "add_labels_to_candidate_pool",
    "parse_prediction_entries",
]
