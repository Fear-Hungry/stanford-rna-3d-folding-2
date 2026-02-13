from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

from .alignment import compute_coverage, map_target_to_template_alignment, project_target_coordinates
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
from .qa_gnn_ranker import (
    is_qa_gnn_model_file,
    load_qa_gnn_runtime,
    score_candidate_feature_dicts_with_qa_gnn_runtime,
)
from .qa_ranker import build_candidate_feature_dict, load_qa_model, score_candidate_with_model, select_candidates_with_diversity
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


def _build_template_maps(
    templates_df: pl.DataFrame,
    template_index_df: pl.DataFrame,
    *,
    location: str,
) -> tuple[dict[str, str], dict[str, dict[int, tuple[float, float, float]]]]:
    required = {"template_uid", "resid", "x", "y", "z"}
    miss = [c for c in required if c not in templates_df.columns]
    if miss:
        raise_error("TBM", location, "templates sem coluna obrigatoria", impact=str(len(miss)), examples=miss[:8])
    required_idx = {"template_uid", "sequence"}
    miss_idx = [c for c in required_idx if c not in template_index_df.columns]
    if miss_idx:
        raise_error(
            "TBM",
            location,
            "template_index sem coluna obrigatoria",
            impact=str(len(miss_idx)),
            examples=miss_idx[:8],
        )

    seq_map: dict[str, str] = {}
    coord_map: dict[str, dict[int, tuple[float, float, float]]] = {}
    for uid, seq in template_index_df.select("template_uid", "sequence").iter_rows():
        uid = str(uid)
        seq = str(seq)
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

    for uid, resid, x, y, z in templates_df.select("template_uid", "resid", "x", "y", "z").iter_rows():
        uid = str(uid)
        resid = int(resid)
        coord = (float(x), float(y), float(z))
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

    missing_seq = [uid for uid in coord_map if uid not in seq_map]
    if missing_seq:
        raise_error(
            "TBM",
            location,
            "template_uid sem sequence correspondente no template_index",
            impact=str(len(missing_seq)),
            examples=missing_seq[:8],
        )

    return seq_map, coord_map


def _build_alignment_profiles(
    *,
    gap_open_scores: tuple[float, ...],
    gap_extend_scores: tuple[float, ...],
    max_variants_per_template: int,
    location: str,
) -> list[tuple[float, float]]:
    if not gap_open_scores:
        raise_error("TBM", location, "gap_open_scores vazio", impact="1", examples=[])
    if not gap_extend_scores:
        raise_error("TBM", location, "gap_extend_scores vazio", impact="1", examples=[])
    if max_variants_per_template <= 0:
        raise_error(
            "TBM",
            location,
            "max_variants_per_template deve ser > 0",
            impact="1",
            examples=[str(max_variants_per_template)],
        )
    out: list[tuple[float, float]] = []
    for go in gap_open_scores:
        for ge in gap_extend_scores:
            out.append((float(go), float(ge)))
    out = out[: int(max_variants_per_template)]
    if not out:
        raise_error("TBM", location, "nenhum perfil de alinhamento gerado", impact="1", examples=[])
    return out


def _jitter(
    *,
    uid: str,
    resid: int,
    axis: int,
    variant_id: int,
    scale: float,
) -> float:
    if scale <= 0.0:
        return 0.0
    key = f"{uid}|{variant_id}|{resid}|{axis}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    # Deterministic jitter in [-scale, +scale].
    raw = int.from_bytes(digest[:4], "big", signed=False) / float(2**32 - 1)
    return ((raw * 2.0) - 1.0) * scale


def predict_tbm(
    *,
    repo_root: Path,
    retrieval_candidates_path: Path,
    templates_path: Path,
    target_sequences_path: Path,
    out_path: Path,
    n_models: int,
    min_coverage: float = 0.35,
    rerank_pool_size: int = 64,
    gap_open_scores: tuple[float, ...] = (-5.0,),
    gap_extend_scores: tuple[float, ...] = (-1.0,),
    max_variants_per_template: int = 1,
    perturbation_scale: float = 0.0,
    mapping_mode: str = "hybrid",
    projection_mode: str = "template_warped",
    qa_model_path: Path | None = None,
    qa_device: str = "cuda",
    qa_top_pool: int = 40,
    diversity_lambda: float = 0.15,
    chunk_size: int = 200_000,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    max_rows_in_memory: int = DEFAULT_MAX_ROWS_IN_MEMORY,
) -> TbmPredictionResult:
    """
    Generate per-residue predictions from top template candidates.
    """
    location = "src/rna3d_local/tbm_predictor.py:predict_tbm"
    assert_memory_budget(stage="TBM", location=location, budget_mb=memory_budget_mb)
    template_index_path = templates_path.parent / "template_index.parquet"
    for p in (retrieval_candidates_path, templates_path, template_index_path, target_sequences_path):
        if not p.exists():
            raise_error("TBM", location, "arquivo obrigatorio ausente", impact="1", examples=[str(p)])
    if n_models <= 0:
        raise_error("TBM", location, "n_models deve ser > 0", impact="1", examples=[str(n_models)])
    if min_coverage <= 0 or min_coverage > 1:
        raise_error("TBM", location, "min_coverage invalido (0,1]", impact="1", examples=[str(min_coverage)])
    if rerank_pool_size <= 0:
        raise_error("TBM", location, "rerank_pool_size deve ser > 0", impact="1", examples=[str(rerank_pool_size)])
    if perturbation_scale < 0.0:
        raise_error("TBM", location, "perturbation_scale invalido (>=0)", impact="1", examples=[str(perturbation_scale)])
    if qa_top_pool <= 0:
        raise_error("TBM", location, "qa_top_pool deve ser > 0", impact="1", examples=[str(qa_top_pool)])
    if diversity_lambda < 0.0:
        raise_error("TBM", location, "diversity_lambda invalido (>=0)", impact="1", examples=[str(diversity_lambda)])
    if chunk_size <= 0:
        raise_error("TBM", location, "chunk_size deve ser > 0", impact="1", examples=[str(chunk_size)])
    align_profiles = _build_alignment_profiles(
        gap_open_scores=gap_open_scores,
        gap_extend_scores=gap_extend_scores,
        max_variants_per_template=int(max_variants_per_template),
        location=location,
    )

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
                columns=("template_uid", "resid", "x", "y", "z"),
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
    template_index_df = collect_streaming(
        lf=scan_table(
            config=TableReadConfig(
                path=template_index_path,
                stage="TBM",
                location=location,
                columns=("template_uid", "sequence"),
            )
        ),
        stage="TBM",
        location=location,
    )
    template_sequences, template_coords = _build_template_maps(
        templates_df,
        template_index_df,
        location=location,
    )

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
    insufficient_coverage_count = 0
    insufficient_coverage_examples: list[str] = []
    selected_mapped_total = 0
    selected_match_total = 0
    selected_mismatch_total = 0
    selected_chem_total = 0
    selected_candidates_total = 0
    qa_model_type = "heuristic"
    qa_model = None
    qa_gnn_runtime = None
    qa_model_sha256 = None
    qa_model_weights_sha256 = None
    if qa_model_path is not None:
        if is_qa_gnn_model_file(model_path=qa_model_path, location=location):
            qa_gnn_runtime = load_qa_gnn_runtime(
                model_path=qa_model_path,
                weights_path=None,
                device=str(qa_device),
                location=location,
            )
            qa_model_type = "qa_gnn"
            qa_model_weights_sha256 = sha256_file(qa_gnn_runtime.weights_path)
        else:
            qa_model = load_qa_model(model_path=qa_model_path, location=location)
            qa_model_type = "linear"
        qa_model_sha256 = sha256_file(qa_model_path)

    try:
        for tid, seq in targets.select("target_id", "sequence").iter_rows():
            tid = str(tid)
            seq = str(seq)
            candidates_all = sorted(candidates_by_target.get(tid, []), key=lambda x: x[1])
            if len(candidates_all) == 0:
                missing_candidates_count += 1
                if len(missing_candidates_examples) < 8:
                    missing_candidates_examples.append(f"{tid}:{len(candidates_all)}")
                continue
            use_pool = max(int(rerank_pool_size), int(n_models))
            candidates = candidates_all[:use_pool]
            target_length = len(seq)
            valid_candidates: list[dict] = []
            for uid, rank, similarity in candidates:
                tpl_seq = template_sequences.get(uid)
                tpl_coords = template_coords.get(uid)
                if tpl_seq is None or tpl_coords is None:
                    raise_error("TBM", location, "template_uid sem dados de template", impact="1", examples=[uid])

                for variant_id, (gap_open, gap_extend) in enumerate(align_profiles, start=1):
                    mapping_info = map_target_to_template_alignment(
                        target_sequence=seq,
                        template_sequence=tpl_seq,
                        location=location,
                        open_gap_score=float(gap_open),
                        extend_gap_score=float(gap_extend),
                        mapping_mode=str(mapping_mode),
                    )
                    match_count = 0
                    mismatch_count = 0
                    chem_compatible_count = 0
                    valid_mapped = {
                        tpos: spos
                        for tpos, spos in mapping_info.target_to_template.items()
                        if int(spos) in tpl_coords
                    }
                    for tpos in valid_mapped.keys():
                        ptype = mapping_info.pair_types.get(int(tpos), "mismatch")
                        if ptype == "match":
                            match_count += 1
                        elif ptype == "chem_compatible_mismatch":
                            mismatch_count += 1
                            chem_compatible_count += 1
                        else:
                            mismatch_count += 1
                    cov = compute_coverage(mapped_positions=len(valid_mapped), target_length=target_length, location=location)
                    if cov < min_coverage:
                        continue
                    coords = project_target_coordinates(
                        target_length=target_length,
                        mapping=valid_mapped,
                        template_coordinates=tpl_coords,
                        location=location,
                        projection_mode=str(projection_mode),
                    )
                    if perturbation_scale > 0.0:
                        jittered: list[tuple[float, float, float]] = []
                        for resid, xyz in enumerate(coords, start=1):
                            jx = _jitter(uid=uid, resid=resid, axis=0, variant_id=variant_id, scale=float(perturbation_scale))
                            jy = _jitter(uid=uid, resid=resid, axis=1, variant_id=variant_id, scale=float(perturbation_scale))
                            jz = _jitter(uid=uid, resid=resid, axis=2, variant_id=variant_id, scale=float(perturbation_scale))
                            jittered.append((float(xyz[0] + jx), float(xyz[1] + jy), float(xyz[2] + jz)))
                        coords_use = jittered
                    else:
                        coords_use = coords
                    valid_candidates.append(
                        {
                            "uid": uid,
                            "rank": int(rank),
                            "similarity": float(similarity),
                            "coverage": float(cov),
                            "gap_open_score": float(gap_open),
                            "gap_extend_score": float(gap_extend),
                            "variant_id": int(variant_id),
                            "mapped_count": int(len(valid_mapped)),
                            "match_count": int(match_count),
                            "mismatch_count": int(mismatch_count),
                            "chem_compatible_count": int(chem_compatible_count),
                            "coords": coords_use,
                        }
                    )

            if len(valid_candidates) < n_models:
                insufficient_coverage_count += 1
                if len(insufficient_coverage_examples) < 8:
                    insufficient_coverage_examples.append(f"{tid}:{len(valid_candidates)}/{n_models}")
                continue

            features_batch: list[dict[str, float]] = []
            for cand in valid_candidates:
                features = build_candidate_feature_dict(candidate=cand, target_length=target_length, location=location)
                cand["qa_features"] = features
                features_batch.append(features)
                if qa_model_type == "heuristic":
                    cand["qa_score"] = float((0.7 * float(cand["coverage"])) + (0.3 * float(cand["similarity"])))
                elif qa_model_type == "linear":
                    if qa_model is None:
                        raise_error("TBM", location, "qa_model linear nao carregado", impact="1", examples=[str(qa_model_path)])
                    cand["qa_score"] = score_candidate_with_model(candidate_features=features, model=qa_model, location=location)
                elif qa_model_type == "qa_gnn":
                    # scored in batch below to preserve graph message passing across candidates
                    cand["qa_score"] = 0.0
                else:
                    raise_error("TBM", location, "qa_model_type desconhecido", impact="1", examples=[str(qa_model_type)])

            if qa_model_type == "qa_gnn":
                if qa_gnn_runtime is None:
                    raise_error("TBM", location, "qa_gnn_runtime nao carregado", impact="1", examples=[str(qa_model_path)])
                gnn_scores = score_candidate_feature_dicts_with_qa_gnn_runtime(
                    feature_dicts=features_batch,
                    runtime=qa_gnn_runtime,
                    location=location,
                )
                if len(gnn_scores) != len(valid_candidates):
                    raise_error(
                        "TBM",
                        location,
                        "quantidade de scores QA_GNN divergente dos candidatos",
                        impact=f"scores={len(gnn_scores)} candidatos={len(valid_candidates)}",
                        examples=[str(tid)],
                    )
                for cand, gscore in zip(valid_candidates, gnn_scores, strict=True):
                    cand["qa_score"] = float(gscore)

            qa_pool = sorted(
                valid_candidates,
                key=lambda c: (
                    -float(c["qa_score"]),
                    -float(c["coverage"]),
                    -float(c["similarity"]),
                    int(c["rank"]),
                ),
            )[: max(int(qa_top_pool), int(n_models))]
            selected = select_candidates_with_diversity(
                candidates=qa_pool,
                n_models=int(n_models),
                diversity_lambda=float(diversity_lambda),
                location=location,
            )
            wrote_target = False
            model_id = 0
            for cand in selected:
                model_id += 1
                selected_mapped_total += int(cand.get("mapped_count", 0))
                selected_match_total += int(cand.get("match_count", 0))
                selected_mismatch_total += int(cand.get("mismatch_count", 0))
                selected_chem_total += int(cand.get("chem_compatible_count", 0))
                selected_candidates_total += 1
                for resid, (base, xyz) in enumerate(zip(seq, cand["coords"], strict=True), start=1):
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
                            "template_uid": str(cand["uid"]),
                            "template_rank": int(cand["rank"]),
                            "similarity": float(cand["similarity"]),
                            "coverage": float(cand["coverage"]),
                            "gap_open_score": float(cand["gap_open_score"]),
                            "gap_extend_score": float(cand["gap_extend_score"]),
                            "variant_id": int(cand["variant_id"]),
                            "qa_score": float(cand.get("qa_score", 0.0)),
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
                "alvos sem candidatos de retrieval",
                impact=str(missing_candidates_count),
                examples=missing_candidates_examples,
            )
        if insufficient_coverage_count > 0:
            raise_error(
                "TBM",
                location,
                "alvos sem modelos suficientes apos filtro de cobertura",
                impact=str(insufficient_coverage_count),
                examples=insufficient_coverage_examples,
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
        "params": {
            "n_models": n_models,
            "min_coverage": min_coverage,
            "rerank_pool_size": rerank_pool_size,
            "gap_open_scores": [float(x) for x in gap_open_scores],
            "gap_extend_scores": [float(x) for x in gap_extend_scores],
            "max_variants_per_template": int(max_variants_per_template),
            "perturbation_scale": float(perturbation_scale),
            "mapping_mode": str(mapping_mode),
            "projection_mode": str(projection_mode),
            "qa_model_path": None if qa_model_path is None else _rel(qa_model_path, repo_root),
            "qa_model_type": str(qa_model_type),
            "qa_device": str(qa_device),
            "qa_model_weights_path": None
            if qa_gnn_runtime is None
            else _rel(qa_gnn_runtime.weights_path, repo_root),
            "qa_top_pool": int(qa_top_pool),
            "diversity_lambda": float(diversity_lambda),
        },
        "stats": {
            "n_rows": int(rows_written),
            "n_targets": int(len(target_written)),
            "chunk_size": int(chunk_size),
            "selected_candidates": int(selected_candidates_total),
            "selected_avg_mapped": 0.0 if selected_candidates_total == 0 else float(selected_mapped_total) / float(selected_candidates_total),
            "selected_avg_match": 0.0 if selected_candidates_total == 0 else float(selected_match_total) / float(selected_candidates_total),
            "selected_avg_mismatch": 0.0 if selected_candidates_total == 0 else float(selected_mismatch_total) / float(selected_candidates_total),
            "selected_avg_chem_compatible": 0.0
            if selected_candidates_total == 0
            else float(selected_chem_total) / float(selected_candidates_total),
        },
        "sha256": {
            "predictions_long.parquet": sha256_file(out_path),
            "qa_model.json": None if qa_model_sha256 is None else str(qa_model_sha256),
            "qa_model_weights.pt": None if qa_model_weights_sha256 is None else str(qa_model_weights_sha256),
        },
    }
    manifest_path = out_path.parent / "tbm_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TbmPredictionResult(predictions_path=out_path, manifest_path=manifest_path)
