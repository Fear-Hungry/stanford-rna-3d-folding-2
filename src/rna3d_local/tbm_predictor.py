from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

from .alignment import compute_coverage, map_target_to_template_positions, project_target_coordinates
from .bigdata import (
    DEFAULT_MAX_ROWS_IN_MEMORY,
    DEFAULT_MEMORY_BUDGET_MB,
    TableReadConfig,
    assert_memory_budget,
    assert_row_budget,
    collect_streaming,
    scan_table,
)
from .errors import raise_error
from .utils import sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class TbmPredictionResult:
    predictions_path: Path
    manifest_path: Path


def _build_template_maps(templates_df: pl.DataFrame, *, location: str) -> tuple[dict[str, str], dict[str, dict[int, tuple[float, float, float]]]]:
    required = {"template_uid", "sequence", "resid", "x", "y", "z"}
    miss = [c for c in required if c not in templates_df.columns]
    if miss:
        raise_error("TBM", location, "templates sem coluna obrigatoria", impact=str(len(miss)), examples=miss[:8])

    seq_map: dict[str, str] = {}
    coord_map: dict[str, dict[int, tuple[float, float, float]]] = {}
    for uid, seq, resid, x, y, z in templates_df.select("template_uid", "sequence", "resid", "x", "y", "z").iter_rows():
        uid = str(uid)
        seq = str(seq)
        resid = int(resid)
        coord = (float(x), float(y), float(z))
        prev = seq_map.get(uid)
        if prev is None:
            seq_map[uid] = seq
        elif prev != seq:
            raise_error(
                "TBM",
                location,
                "sequence inconsistente dentro do mesmo template_uid",
                impact="1",
                examples=[uid],
            )
        coord_map.setdefault(uid, {})
        if resid in coord_map[uid]:
            raise_error(
                "TBM",
                location,
                "resid duplicado no template",
                impact="1",
                examples=[f"{uid}:{resid}"],
            )
        coord_map[uid][resid] = coord
    return seq_map, coord_map


def predict_tbm(
    *,
    repo_root: Path,
    retrieval_candidates_path: Path,
    templates_path: Path,
    target_sequences_path: Path,
    out_path: Path,
    n_models: int,
    min_coverage: float = 0.35,
    chunk_size: int = 200_000,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    max_rows_in_memory: int = DEFAULT_MAX_ROWS_IN_MEMORY,
) -> TbmPredictionResult:
    """
    Generate per-residue predictions from top template candidates.
    """
    location = "src/rna3d_local/tbm_predictor.py:predict_tbm"
    assert_memory_budget(stage="TBM", location=location, budget_mb=memory_budget_mb)
    for p in (retrieval_candidates_path, templates_path, target_sequences_path):
        if not p.exists():
            raise_error("TBM", location, "arquivo obrigatorio ausente", impact="1", examples=[str(p)])
    if n_models <= 0:
        raise_error("TBM", location, "n_models deve ser > 0", impact="1", examples=[str(n_models)])
    if min_coverage <= 0 or min_coverage > 1:
        raise_error("TBM", location, "min_coverage invalido (0,1]", impact="1", examples=[str(min_coverage)])
    if chunk_size <= 0:
        raise_error("TBM", location, "chunk_size deve ser > 0", impact="1", examples=[str(chunk_size)])

    retrieval_df = collect_streaming(
        lf=scan_table(
            config=TableReadConfig(
                path=retrieval_candidates_path,
                stage="TBM",
                location=location,
                columns=("target_id", "template_uid", "rank", "similarity"),
            )
        ),
        stage="TBM",
        location=location,
    )
    retrieval_df = retrieval_df.sort(["target_id", "rank"])
    assert_row_budget(
        stage="TBM",
        location=location,
        rows=int(retrieval_df.height),
        max_rows_in_memory=max_rows_in_memory,
        label="retrieval_candidates",
    )

    templates_df = collect_streaming(
        lf=scan_table(
            config=TableReadConfig(
                path=templates_path,
                stage="TBM",
                location=location,
                columns=("template_uid", "sequence", "resid", "x", "y", "z"),
            )
        ),
        stage="TBM",
        location=location,
    )
    assert_row_budget(
        stage="TBM",
        location=location,
        rows=int(templates_df.height),
        max_rows_in_memory=max_rows_in_memory,
        label="templates",
    )
    template_sequences, template_coords = _build_template_maps(templates_df, location=location)

    targets = collect_streaming(
        lf=scan_table(
            config=TableReadConfig(
                path=target_sequences_path,
                stage="TBM",
                location=location,
                columns=("target_id", "sequence"),
            )
        ),
        stage="TBM",
        location=location,
    )
    targets = targets.select(pl.col("target_id").cast(pl.Utf8), pl.col("sequence").cast(pl.Utf8)).sort("target_id")
    assert_row_budget(
        stage="TBM",
        location=location,
        rows=int(targets.height),
        max_rows_in_memory=max_rows_in_memory,
        label="targets",
    )

    candidates_by_target: dict[str, list[tuple[str, int, float]]] = {}
    for tid, uid, rank, similarity in retrieval_df.select("target_id", "template_uid", "rank", "similarity").iter_rows():
        candidates_by_target.setdefault(str(tid), []).append((str(uid), int(rank), float(similarity)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_out_path.exists():
        tmp_out_path.unlink()
    buffer: list[dict] = []
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    target_written: set[str] = set()
    finalized = False

    def _flush() -> None:
        nonlocal writer, rows_written
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer)
        if writer is None:
            writer = pq.ParquetWriter(str(tmp_out_path), table.schema, compression="zstd")
        writer.write_table(table)
        rows_written += len(buffer)
        buffer.clear()

    missing_candidates_count = 0
    missing_candidates_examples: list[str] = []
    low_coverage_count = 0
    low_coverage_examples: list[str] = []

    try:
        for tid, seq in targets.select("target_id", "sequence").iter_rows():
            tid = str(tid)
            seq = str(seq)
            candidates = sorted(candidates_by_target.get(tid, []), key=lambda x: x[1])
            if len(candidates) < n_models:
                missing_candidates_count += 1
                if len(missing_candidates_examples) < 8:
                    missing_candidates_examples.append(f"{tid}:{len(candidates)}")
                continue
            target_length = len(seq)
            wrote_target = False
            for model_id, (uid, rank, similarity) in enumerate(candidates[:n_models], start=1):
                tpl_seq = template_sequences.get(uid)
                tpl_coords = template_coords.get(uid)
                if tpl_seq is None or tpl_coords is None:
                    raise_error("TBM", location, "template_uid sem dados de template", impact="1", examples=[uid])

                mapping = map_target_to_template_positions(
                    target_sequence=seq,
                    template_sequence=tpl_seq,
                    location=location,
                )
                valid_mapped = {
                    tpos: spos
                    for tpos, spos in mapping.items()
                    if spos in tpl_coords
                }
                cov = compute_coverage(mapped_positions=len(valid_mapped), target_length=target_length, location=location)
                if cov < min_coverage:
                    low_coverage_count += 1
                    if len(low_coverage_examples) < 8:
                        low_coverage_examples.append(f"{tid}:{uid}:{cov:.4f}")
                    continue
                coords = project_target_coordinates(
                    target_length=target_length,
                    mapping=valid_mapped,
                    template_coordinates=tpl_coords,
                    location=location,
                )
                for resid, (base, xyz) in enumerate(zip(seq, coords, strict=True), start=1):
                    buffer.append(
                        {
                            "branch": "tbm",
                            "target_id": tid,
                            "ID": f"{tid}_{resid}",
                            "resid": resid,
                            "resname": base,
                            "model_id": model_id,
                            "x": float(xyz[0]),
                            "y": float(xyz[1]),
                            "z": float(xyz[2]),
                            "template_uid": uid,
                            "template_rank": int(rank),
                            "similarity": float(similarity),
                            "coverage": float(cov),
                        }
                    )
                wrote_target = True
                if len(buffer) >= chunk_size:
                    _flush()
                    assert_memory_budget(stage="TBM", location=location, budget_mb=memory_budget_mb)
            if wrote_target:
                target_written.add(tid)

        _flush()
        if writer is not None:
            writer.close()
            writer = None
        assert_memory_budget(stage="TBM", location=location, budget_mb=memory_budget_mb)

        if missing_candidates_count > 0:
            raise_error(
                "TBM",
                location,
                "alvos sem candidatos suficientes para n_models",
                impact=str(missing_candidates_count),
                examples=missing_candidates_examples,
            )
        if low_coverage_count > 0:
            raise_error(
                "TBM",
                location,
                "cobertura abaixo do minimo",
                impact=str(low_coverage_count),
                examples=low_coverage_examples,
            )
        if rows_written == 0:
            raise_error("TBM", location, "nenhuma predicao gerada", impact="0", examples=[])

        if out_path.exists():
            out_path.unlink()
        tmp_out_path.replace(out_path)
        finalized = True
    finally:
        if writer is not None:
            writer.close()
        if not finalized and tmp_out_path.exists():
            tmp_out_path.unlink()

    manifest = {
        "created_utc": _utc_now(),
        "paths": {
            "retrieval_candidates": _rel(retrieval_candidates_path, repo_root),
            "templates": _rel(templates_path, repo_root),
            "target_sequences": _rel(target_sequences_path, repo_root),
            "predictions_long": _rel(out_path, repo_root),
        },
        "params": {"n_models": n_models, "min_coverage": min_coverage},
        "stats": {"n_rows": int(rows_written), "n_targets": int(len(target_written)), "chunk_size": int(chunk_size)},
        "sha256": {"predictions_long.parquet": sha256_file(out_path)},
    }
    manifest_path = out_path.parent / "tbm_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TbmPredictionResult(predictions_path=out_path, manifest_path=manifest_path)
