from __future__ import annotations

import heapq
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash

from ..alignment import compute_coverage, map_target_to_template_alignment, project_target_coordinates
from ..bigdata import DEFAULT_MAX_ROWS_IN_MEMORY, DEFAULT_MEMORY_BUDGET_MB, assert_memory_budget, assert_row_budget
from ..compute_backend import ComputeBackend, resolve_compute_backend
from ..errors import raise_error
from ..qa_gnn_ranker import (
    is_qa_gnn_model_file,
    load_qa_gnn_runtime,
    score_candidate_feature_dicts_with_qa_gnn_runtime,
)
from ..qa_ranker import build_candidate_feature_dict, load_qa_model, score_candidate_with_model, select_candidates_with_diversity
from ..template_pt import load_template_features_for_target
from ..utils import sha256_file
from .config import RnaProConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _hash_kmer_features(seq: str, *, k: int, dim: int, seed: int) -> np.ndarray:
    s = (seq or "").strip().upper()
    arr = np.zeros(dim, dtype=np.float32)
    if len(s) == 0:
        return arr
    if len(s) < k:
        kmers = [s]
    else:
        kmers = [s[i : i + k] for i in range(0, len(s) - k + 1)]
    for km in kmers:
        h = xxhash.xxh64(km, seed=seed).intdigest() % dim
        arr[int(h)] += 1.0
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr /= norm
    return arr


def _read_model(path: Path, *, location: str) -> dict:
    if not path.exists():
        raise_error("RNAPRO_INFER", location, "model.json nao encontrado", impact="1", examples=[str(path)])
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise_error(
            "RNAPRO_INFER",
            location,
            "falha ao ler model.json",
            impact="1",
            examples=[f"{type(e).__name__}:{e}"],
        )
    raise AssertionError("unreachable")


def _read_targets(*, target_sequences_path: Path, location: str) -> pl.DataFrame:
    targets = pl.read_csv(target_sequences_path, infer_schema_length=1000)
    for col in ("target_id", "sequence", "temporal_cutoff"):
        if col not in targets.columns:
            raise_error("RNAPRO_INFER", location, "target_sequences sem coluna obrigatoria", impact="1", examples=[col])
    targets = targets.with_columns(
        pl.col("target_id").cast(pl.Utf8),
        pl.col("sequence").cast(pl.Utf8),
        pl.col("temporal_cutoff").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("temporal_cutoff"),
    )
    if int(targets.get_column("temporal_cutoff").null_count()) > 0:
        bad = targets.filter(pl.col("temporal_cutoff").is_null()).get_column("target_id").head(8).to_list()
        raise_error(
            "RNAPRO_INFER",
            location,
            "temporal_cutoff invalido em target_sequences",
            impact=str(int(targets.get_column("temporal_cutoff").null_count())),
            examples=bad,
        )
    return targets.sort("target_id")


def _assign_qa_scores(
    *,
    candidates: list[dict],
    target_length: int,
    qa_model_type: str,
    qa_model,
    qa_gnn_runtime,
    location: str,
    heuristic_mode: str,
) -> None:
    if heuristic_mode not in {"precomputed", "artifacts"}:
        raise_error("RNAPRO_INFER", location, "heuristic_mode invalido", impact="1", examples=[str(heuristic_mode)])
    features_batch: list[dict[str, float]] = []
    for cand in candidates:
        features = build_candidate_feature_dict(candidate=cand, target_length=target_length, location=location)
        cand["qa_features"] = features
        features_batch.append(features)
        if qa_model_type == "heuristic":
            if heuristic_mode == "precomputed":
                cand["qa_score"] = float((0.8 * float(cand["coverage"])) + 0.2)
            else:
                cand["qa_score"] = float((0.7 * float(cand["coverage"])) + (0.3 * float(cand["similarity"])))
        elif qa_model_type == "linear":
            if qa_model is None:
                raise_error("RNAPRO_INFER", location, "qa_model linear nao carregado", impact="1", examples=["qa_model=None"])
            cand["qa_score"] = score_candidate_with_model(candidate_features=features, model=qa_model, location=location)
        elif qa_model_type == "qa_gnn":
            # assigned in batch below to preserve graph message passing between candidates
            cand["qa_score"] = 0.0
        else:
            raise_error("RNAPRO_INFER", location, "qa_model_type desconhecido", impact="1", examples=[str(qa_model_type)])

    if qa_model_type == "qa_gnn":
        if qa_gnn_runtime is None:
            raise_error("RNAPRO_INFER", location, "qa_gnn_runtime nao carregado", impact="1", examples=["qa_gnn_runtime=None"])
        gnn_scores = score_candidate_feature_dicts_with_qa_gnn_runtime(
            feature_dicts=features_batch,
            runtime=qa_gnn_runtime,
            location=location,
        )
        if len(gnn_scores) != len(candidates):
            raise_error(
                "RNAPRO_INFER",
                location,
                "quantidade de scores QA_GNN divergente dos candidatos",
                impact=f"scores={len(gnn_scores)} candidatos={len(candidates)}",
                examples=[],
            )
        for cand, gscore in zip(candidates, gnn_scores, strict=True):
            cand["qa_score"] = float(gscore)


def infer_rnapro(
    *,
    repo_root: Path,
    model_dir: Path,
    target_sequences_path: Path,
    out_path: Path,
    n_models: int | None = None,
    min_coverage: float | None = None,
    rerank_pool_multiplier: int = 8,
    chunk_size: int = 200_000,
    use_template: str = "none",
    template_features_dir: Path | None = None,
    template_source: str = "tbm",
    mapping_mode: str = "hybrid",
    projection_mode: str = "template_warped",
    qa_model_path: Path | None = None,
    qa_device: str = "cuda",
    qa_top_pool: int = 40,
    diversity_lambda: float = 0.15,
    compute_backend: str = "auto",
    gpu_memory_budget_mb: int = 12_288,
    gpu_precision: str = "fp32",
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    max_rows_in_memory: int = DEFAULT_MAX_ROWS_IN_MEMORY,
) -> Path:
    """
    Inference for RNAPro proxy model.

    Modes:
    - use_template="none": legacy nearest-neighbor inference from trained RNAPro artifacts.
    - use_template="ca_precomputed": consume per-target template_features.pt artifacts.
    """
    location = "src/rna3d_local/rnapro/infer.py:infer_rnapro"
    assert_memory_budget(stage="RNAPRO_INFER", location=location, budget_mb=memory_budget_mb)

    if use_template not in {"none", "ca_precomputed"}:
        raise_error("RNAPRO_INFER", location, "use_template invalido", impact="1", examples=[str(use_template)])
    if template_source not in {"tbm", "mmseqs2", "external"}:
        raise_error("RNAPRO_INFER", location, "template_source invalido", impact="1", examples=[str(template_source)])
    if use_template == "ca_precomputed" and template_features_dir is None:
        raise_error("RNAPRO_INFER", location, "template_features_dir obrigatorio para ca_precomputed", impact="1", examples=["template_features_dir=None"])

    model_path = model_dir / "model.json"
    payload = _read_model(model_path, location=location)
    config = RnaProConfig.from_dict(payload.get("config", {}))
    try:
        config.validate()
    except ValueError as e:
        raise_error("RNAPRO_INFER", location, "config invalida no model.json", impact="1", examples=[str(e)])

    use_n_models = int(n_models) if n_models is not None else int(config.n_models)
    use_min_coverage = float(min_coverage) if min_coverage is not None else float(config.min_coverage)
    if use_n_models <= 0:
        raise_error("RNAPRO_INFER", location, "n_models invalido", impact="1", examples=[str(use_n_models)])
    if use_min_coverage <= 0 or use_min_coverage > 1:
        raise_error(
            "RNAPRO_INFER",
            location,
            "min_coverage invalido (0,1]",
            impact="1",
            examples=[str(use_min_coverage)],
        )
    if rerank_pool_multiplier <= 0:
        raise_error(
            "RNAPRO_INFER",
            location,
            "rerank_pool_multiplier deve ser > 0",
            impact="1",
            examples=[str(rerank_pool_multiplier)],
        )
    if chunk_size <= 0:
        raise_error("RNAPRO_INFER", location, "chunk_size invalido (deve ser > 0)", impact="1", examples=[str(chunk_size)])
    if qa_top_pool <= 0:
        raise_error("RNAPRO_INFER", location, "qa_top_pool invalido (deve ser > 0)", impact="1", examples=[str(qa_top_pool)])
    if diversity_lambda < 0.0:
        raise_error("RNAPRO_INFER", location, "diversity_lambda invalido (>=0)", impact="1", examples=[str(diversity_lambda)])
    backend = resolve_compute_backend(
        requested=str(compute_backend),
        precision=str(gpu_precision),
        gpu_memory_budget_mb=int(gpu_memory_budget_mb),
        stage="RNAPRO_INFER",
        location=location,
    )

    targets = _read_targets(target_sequences_path=target_sequences_path, location=location)
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

    if use_template == "ca_precomputed":
        assert template_features_dir is not None
        return _infer_from_precomputed_templates(
            repo_root=repo_root,
            targets=targets,
            target_sequences_path=target_sequences_path,
            out_path=out_path,
            n_models=use_n_models,
            min_coverage=use_min_coverage,
            chunk_size=chunk_size,
            template_features_dir=template_features_dir,
            template_source=template_source,
            qa_model_type=str(qa_model_type),
            qa_model=qa_model,
            qa_gnn_runtime=qa_gnn_runtime,
            qa_model_path=qa_model_path,
            qa_model_sha256=qa_model_sha256,
            qa_model_weights_sha256=qa_model_weights_sha256,
            qa_device=str(qa_device),
            qa_top_pool=int(qa_top_pool),
            diversity_lambda=float(diversity_lambda),
            backend=backend,
            memory_budget_mb=memory_budget_mb,
            max_rows_in_memory=max_rows_in_memory,
        )

    return _infer_from_model_artifacts(
        repo_root=repo_root,
        model_dir=model_dir,
        payload=payload,
        targets=targets,
        target_sequences_path=target_sequences_path,
        out_path=out_path,
        n_models=use_n_models,
        min_coverage=use_min_coverage,
        rerank_pool_multiplier=rerank_pool_multiplier,
        chunk_size=chunk_size,
        mapping_mode=str(mapping_mode),
        projection_mode=str(projection_mode),
        qa_model_type=str(qa_model_type),
        qa_model=qa_model,
        qa_gnn_runtime=qa_gnn_runtime,
        qa_model_path=qa_model_path,
        qa_model_sha256=qa_model_sha256,
        qa_model_weights_sha256=qa_model_weights_sha256,
        qa_device=str(qa_device),
        qa_top_pool=int(qa_top_pool),
        diversity_lambda=float(diversity_lambda),
        backend=backend,
        memory_budget_mb=memory_budget_mb,
        max_rows_in_memory=max_rows_in_memory,
    )


def _infer_from_precomputed_templates(
    *,
    repo_root: Path,
    targets: pl.DataFrame,
    target_sequences_path: Path,
    out_path: Path,
    n_models: int,
    min_coverage: float,
    chunk_size: int,
    template_features_dir: Path,
    template_source: str,
    qa_model_type: str,
    qa_model,
    qa_gnn_runtime,
    qa_model_path: Path | None,
    qa_model_sha256: str | None,
    qa_model_weights_sha256: str | None,
    qa_device: str,
    qa_top_pool: int,
    diversity_lambda: float,
    backend: ComputeBackend,
    memory_budget_mb: int,
    max_rows_in_memory: int,
) -> Path:
    location = "src/rna3d_local/rnapro/infer.py:_infer_from_precomputed_templates"
    for p in (target_sequences_path, template_features_dir):
        if not p.exists():
            raise_error("RNAPRO_INFER", location, "arquivo/diretorio obrigatorio ausente", impact="1", examples=[str(p)])

    expected_rows = int(
        targets.select(pl.col("sequence").str.len_chars().cast(pl.Int64).sum()).item()  # type: ignore[arg-type]
        * int(n_models)
    )
    assert_row_budget(
        stage="RNAPRO_INFER",
        location=location,
        rows=max(expected_rows, int(targets.height)),
        max_rows_in_memory=max_rows_in_memory,
        label="targets_or_expected_rows",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_out_path.exists():
        tmp_out_path.unlink()
    writer: pq.ParquetWriter | None = None
    buffer: list[dict] = []
    rows_written = 0
    targets_written: set[str] = set()
    finalized = False
    insufficient_coverage_count = 0
    insufficient_coverage_examples: list[str] = []
    selected_candidates_total = 0

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

    try:
        for tid, seq, _cutoff in targets.select("target_id", "sequence", "temporal_cutoff").iter_rows():
            tid = str(tid)
            seq = str(seq)
            payload = load_template_features_for_target(
                template_features_dir=template_features_dir,
                target_id=tid,
                stage="RNAPRO_INFER",
                location=location,
            )
            if payload["sequence"] != seq:
                raise_error(
                    "RNAPRO_INFER",
                    location,
                    "sequence divergente entre target_sequences e template_features.pt",
                    impact="1",
                    examples=[tid],
                )
            coords = payload["coordinates"]
            mask = payload["mask"]
            model_ids = payload["model_ids"]
            if len(model_ids) < int(n_models):
                raise_error(
                    "RNAPRO_INFER",
                    location,
                    "template_features com menos modelos que o solicitado",
                    impact=f"target={tid} requested={n_models} available={len(model_ids)}",
                    examples=[str(payload["path"])],
                )
            if coords.shape[1] != len(seq):
                raise_error(
                    "RNAPRO_INFER",
                    location,
                    "template_features com tamanho de residuo divergente do target",
                    impact=f"target={tid} seq_len={len(seq)} pt_resid={coords.shape[1]}",
                    examples=[str(payload["path"])],
                )

            selected_candidates: list[dict] = []
            for i in range(len(model_ids)):
                cov = float(mask[i].mean())
                if cov >= float(min_coverage):
                    is_complete = bool(mask[i].all())
                    coords_i = [
                        (float(coords[i, ridx, 0]), float(coords[i, ridx, 1]), float(coords[i, ridx, 2]))
                        for ridx in range(int(len(seq)))
                    ]
                    selected_candidates.append(
                        {
                            "idx": int(i),
                            "coverage": float(cov),
                            "similarity": 1.0,
                            "mapped_count": int(round(cov * len(seq))),
                            "match_count": int(round(cov * len(seq))),
                            "mismatch_count": 0,
                            "chem_compatible_count": 0,
                            "gap_open_score": 0.0,
                            "gap_extend_score": 0.0,
                            "coords": coords_i,
                            "is_complete": bool(is_complete),
                        }
                    )
            if len(selected_candidates) < int(n_models):
                insufficient_coverage_count += 1
                if len(insufficient_coverage_examples) < 8:
                    insufficient_coverage_examples.append(f"{tid}:{len(selected_candidates)}/{n_models}")
                continue

            complete_candidates = [cand for cand in selected_candidates if bool(cand.get("is_complete", False))]
            if len(complete_candidates) < int(n_models):
                insufficient_coverage_count += 1
                if len(insufficient_coverage_examples) < 8:
                    insufficient_coverage_examples.append(
                        f"{tid}:complete={len(complete_candidates)}/{n_models};coverage={len(selected_candidates)}"
                    )
                continue

            _assign_qa_scores(
                candidates=complete_candidates,
                target_length=len(seq),
                qa_model_type=str(qa_model_type),
                qa_model=qa_model,
                qa_gnn_runtime=qa_gnn_runtime,
                location=location,
                heuristic_mode="precomputed",
            )
            qa_pool = sorted(
                complete_candidates,
                key=lambda c: (-float(c["qa_score"]), -float(c["coverage"]), int(c["idx"])),
            )[: max(int(qa_top_pool), int(n_models))]
            selected = select_candidates_with_diversity(
                candidates=qa_pool,
                n_models=int(n_models),
                diversity_lambda=float(diversity_lambda),
                location=location,
            )
            model_id = 0
            for cand in selected:
                model_id += 1
                i = int(cand["idx"])
                cov = float(cand["coverage"])
                selected_candidates_total += 1
                for resid in range(1, len(seq) + 1):
                    idx = resid - 1
                    if not bool(mask[i, idx]):
                        raise_error(
                            "RNAPRO_INFER",
                            location,
                            "template precomputado com residuo faltante no modelo selecionado",
                            impact=f"target={tid} model={model_id} resid={resid}",
                            examples=[str(payload["path"])],
                        )
                    xyz = coords[i, idx]
                    buffer.append(
                        {
                            "branch": "rnapro",
                            "target_id": tid,
                            "ID": f"{tid}_{resid}",
                            "resid": resid,
                            "resname": seq[idx],
                            "model_id": model_id,
                            "x": float(xyz[0]),
                            "y": float(xyz[1]),
                            "z": float(xyz[2]),
                            "template_uid": f"{template_source}:{tid}:{model_id}",
                            "similarity": 1.0,
                            "coverage": float(cov),
                            "qa_score": float(cand.get("qa_score", 0.0)),
                        }
                    )
                if len(buffer) >= chunk_size:
                    _flush()
                    assert_memory_budget(stage="RNAPRO_INFER", location=location, budget_mb=memory_budget_mb)
            targets_written.add(tid)

        _flush()
        if writer is not None:
            writer.close()
            writer = None
        assert_memory_budget(stage="RNAPRO_INFER", location=location, budget_mb=memory_budget_mb)

        if insufficient_coverage_count > 0:
            raise_error(
                "RNAPRO_INFER",
                location,
                "alvos sem modelos completos suficientes apos filtros de cobertura/completude",
                impact=str(insufficient_coverage_count),
                examples=insufficient_coverage_examples,
            )
        if rows_written == 0:
            raise_error("RNAPRO_INFER", location, "nenhuma predicao gerada", impact="0", examples=[])
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
            "template_features_dir": _rel(template_features_dir, repo_root),
            "target_sequences": _rel(target_sequences_path, repo_root),
            "predictions_long": _rel(out_path, repo_root),
        },
        "params": {
            "n_models": int(n_models),
            "min_coverage": float(min_coverage),
            "use_template": "ca_precomputed",
            "template_source": template_source,
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
            "n_targets": int(len(targets_written)),
            "chunk_size": int(chunk_size),
            "selected_candidates": int(selected_candidates_total),
        },
        "compute": backend.to_manifest_dict(),
        "sha256": {
            "predictions_long.parquet": sha256_file(out_path),
            "qa_model.json": None if qa_model_sha256 is None else str(qa_model_sha256),
            "qa_model_weights.pt": None if qa_model_weights_sha256 is None else str(qa_model_weights_sha256),
        },
    }
    manifest_path = out_path.parent / "rnapro_infer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _infer_from_model_artifacts(
    *,
    repo_root: Path,
    model_dir: Path,
    payload: dict,
    targets: pl.DataFrame,
    target_sequences_path: Path,
    out_path: Path,
    n_models: int,
    min_coverage: float,
    rerank_pool_multiplier: int,
    chunk_size: int,
    mapping_mode: str,
    projection_mode: str,
    qa_model_type: str,
    qa_model,
    qa_gnn_runtime,
    qa_model_path: Path | None,
    qa_model_sha256: str | None,
    qa_model_weights_sha256: str | None,
    qa_device: str,
    qa_top_pool: int,
    diversity_lambda: float,
    backend: ComputeBackend,
    memory_budget_mb: int,
    max_rows_in_memory: int,
) -> Path:
    location = "src/rna3d_local/rnapro/infer.py:_infer_from_model_artifacts"
    try:
        features_path = repo_root / payload["paths"]["target_features"]
        seq_lookup_path = repo_root / payload["paths"]["sequence_lookup"]
        coords_path = repo_root / payload["paths"]["coordinates"]
    except Exception as e:  # noqa: BLE001
        raise_error(
            "RNAPRO_INFER",
            location,
            "model.json sem paths obrigatorios",
            impact="1",
            examples=[f"{type(e).__name__}:{e}"],
        )
        raise AssertionError("unreachable")

    for p in (features_path, seq_lookup_path, coords_path, target_sequences_path):
        if not p.exists():
            raise_error("RNAPRO_INFER", location, "arquivo obrigatorio ausente", impact="1", examples=[str(p)])

    config = RnaProConfig.from_dict(payload.get("config", {}))

    features_df = pl.read_parquet(features_path).sort("target_id")
    seq_df = pl.read_parquet(seq_lookup_path).sort("target_id")
    coords_df = pl.read_parquet(coords_path)
    assert_row_budget(
        stage="RNAPRO_INFER",
        location=location,
        rows=int(features_df.height + seq_df.height + coords_df.height),
        max_rows_in_memory=max_rows_in_memory,
        label="features+sequence_lookup+coordinates",
    )
    for col in ("target_id", "sequence", "release_date"):
        if col not in seq_df.columns:
            raise_error("RNAPRO_INFER", location, "sequence_lookup sem coluna obrigatoria", impact="1", examples=[col])
    release_dtype = seq_df.schema.get("release_date")
    if release_dtype != pl.Date:
        seq_df = seq_df.with_columns(
            pl.col("release_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("release_date")
        )
    if int(seq_df.get_column("release_date").null_count()) > 0:
        bad = seq_df.filter(pl.col("release_date").is_null()).get_column("target_id").head(8).to_list()
        raise_error(
            "RNAPRO_INFER",
            location,
            "release_date invalida no modelo",
            impact=str(int(seq_df.get_column("release_date").null_count())),
            examples=bad,
        )

    feat_cols = [c for c in features_df.columns if c.startswith("f_")]
    if len(feat_cols) != int(config.feature_dim):
        raise_error(
            "RNAPRO_INFER",
            location,
            "dimensao de features divergente do config",
            impact=f"expected={config.feature_dim} got={len(feat_cols)}",
            examples=feat_cols[:8],
        )
    train_ids = [str(v) for v in features_df.get_column("target_id").to_list()]
    train_matrix = features_df.select(feat_cols).to_numpy().astype(np.float32, copy=False)
    if train_matrix.shape[0] == 0:
        raise_error("RNAPRO_INFER", location, "matriz de treino vazia", impact="0", examples=[])
    torch = None
    train_matrix_t = None
    device = None
    gpu_dtype = None
    if backend.backend == "cuda":
        try:
            import torch as _torch  # noqa: PLC0415
        except Exception as e:  # noqa: BLE001
            raise_error("RNAPRO_INFER", location, "falha ao importar torch para backend CUDA", impact="1", examples=[f"{type(e).__name__}:{e}"])
        torch = _torch
        device = torch.device("cuda")
        gpu_dtype = torch.float16 if backend.precision == "fp16" else torch.float32
        train_matrix_t = torch.tensor(train_matrix, dtype=gpu_dtype, device=device)

    seq_map: dict[str, str] = {}
    release_map: dict[str, object] = {}
    for tid, seq, release_date in seq_df.select("target_id", "sequence", "release_date").iter_rows():
        tid = str(tid)
        seq_map[tid] = str(seq)
        release_map[tid] = release_date
    coords_map: dict[str, dict[int, tuple[float, float, float]]] = {}
    for tid, resid, x, y, z in coords_df.select("target_id", "resid", "x", "y", "z").iter_rows():
        tid = str(tid)
        coords_map.setdefault(tid, {})
        coords_map[tid][int(resid)] = (float(x), float(y), float(z))

    train_release_dates: list[object] = []
    for tid in train_ids:
        release_date = release_map.get(tid)
        if release_date is None:
            raise_error("RNAPRO_INFER", location, "target sem release_date no modelo", impact="1", examples=[tid])
        train_release_dates.append(release_date)

    assert_row_budget(
        stage="RNAPRO_INFER",
        location=location,
        rows=int(features_df.height + seq_df.height + coords_df.height + targets.height),
        max_rows_in_memory=max_rows_in_memory,
        label="model_data+targets",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_out_path.exists():
        tmp_out_path.unlink()
    writer: pq.ParquetWriter | None = None
    buffer: list[dict] = []
    rows_written = 0
    targets_written: set[str] = set()
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

    no_candidates_count = 0
    no_candidates_examples: list[str] = []
    insufficient_coverage_count = 0
    insufficient_coverage_examples: list[str] = []
    selected_mapped_total = 0
    selected_match_total = 0
    selected_mismatch_total = 0
    selected_chem_total = 0
    selected_candidates_total = 0
    try:
        for tid, seq, cutoff in targets.select("target_id", "sequence", "temporal_cutoff").iter_rows():
            tid = str(tid)
            seq = str(seq)
            feat = _hash_kmer_features(seq, k=config.kmer_size, dim=config.feature_dim, seed=config.seed)
            if float(np.linalg.norm(feat)) == 0.0:
                raise_error("RNAPRO_INFER", location, "feature nula para alvo", impact="1", examples=[tid])

            pool_k = max(int(n_models) * int(rerank_pool_multiplier), int(n_models))
            if backend.backend == "cuda":
                assert torch is not None and train_matrix_t is not None and device is not None and gpu_dtype is not None
                feat_t = torch.tensor(feat, dtype=gpu_dtype, device=device)
                sims_t = torch.mv(train_matrix_t, feat_t)
                valid_idx_np = np.asarray([int(i) for i, rel in enumerate(train_release_dates) if rel <= cutoff], dtype=np.int64)
                valid_candidates = int(valid_idx_np.size)
                if valid_candidates < int(n_models):
                    no_candidates_count += 1
                    if len(no_candidates_examples) < 8:
                        no_candidates_examples.append(f"{tid}:{valid_candidates}")
                    continue
                valid_idx_t = torch.tensor(valid_idx_np, dtype=torch.long, device=device)
                valid_sims_t = sims_t[valid_idx_t]
                topn = min(int(pool_k), int(valid_candidates))
                vals_t, ord_t = torch.topk(valid_sims_t, k=int(topn), largest=True, sorted=True)
                chosen_idx_t = valid_idx_t[ord_t]
                chosen_idx = chosen_idx_t.detach().cpu().numpy().tolist()
                chosen_vals = vals_t.detach().cpu().numpy().tolist()
                ranked = [(float(sim), str(train_ids[int(i)])) for sim, i in zip(chosen_vals, chosen_idx, strict=True)]
                ranked = sorted(ranked, key=lambda x: (-x[0], x[1]))
            else:
                sims = train_matrix @ feat
                top_heap: list[tuple[float, str]] = []
                valid_candidates = 0
                for idx, (sid, rel) in enumerate(zip(train_ids, train_release_dates, strict=True)):
                    if rel > cutoff:
                        continue
                    valid_candidates += 1
                    item = (float(sims[idx]), sid)
                    if len(top_heap) < pool_k:
                        heapq.heappush(top_heap, item)
                    elif item > top_heap[0]:
                        heapq.heapreplace(top_heap, item)

                if valid_candidates < int(n_models):
                    no_candidates_count += 1
                    if len(no_candidates_examples) < 8:
                        no_candidates_examples.append(f"{tid}:{valid_candidates}")
                    continue
                ranked = sorted(top_heap, key=lambda x: (-x[0], x[1]))
            target_len = len(seq)
            coverage_valid: list[dict] = []
            for sim, train_id in ranked:
                tpl_seq = seq_map.get(train_id)
                tpl_coords = coords_map.get(train_id)
                if tpl_seq is None or tpl_coords is None:
                    raise_error(
                        "RNAPRO_INFER",
                        location,
                        "modelo com template de treino incompleto",
                        impact="1",
                        examples=[train_id],
                    )
                mapping_info = map_target_to_template_alignment(
                    target_sequence=seq,
                    template_sequence=tpl_seq,
                    location=location,
                    mapping_mode=str(mapping_mode),
                )
                match_count = 0
                mismatch_count = 0
                chem_compatible_count = 0
                valid_mapping = {t: s for t, s in mapping_info.target_to_template.items() if int(s) in tpl_coords}
                for t in valid_mapping.keys():
                    ptype = mapping_info.pair_types.get(int(t), "mismatch")
                    if ptype == "match":
                        match_count += 1
                    elif ptype == "chem_compatible_mismatch":
                        mismatch_count += 1
                        chem_compatible_count += 1
                    else:
                        mismatch_count += 1
                cov = compute_coverage(mapped_positions=len(valid_mapping), target_length=target_len, location=location)
                if cov < float(min_coverage):
                    continue
                coords = project_target_coordinates(
                    target_length=target_len,
                    mapping=valid_mapping,
                    template_coordinates=tpl_coords,
                    location=location,
                    projection_mode=str(projection_mode),
                )
                coverage_valid.append(
                    {
                        "train_id": str(train_id),
                        "similarity": float(sim),
                        "coverage": float(cov),
                        "mapped_count": int(len(valid_mapping)),
                        "match_count": int(match_count),
                        "mismatch_count": int(mismatch_count),
                        "chem_compatible_count": int(chem_compatible_count),
                        "gap_open_score": 0.0,
                        "gap_extend_score": 0.0,
                        "coords": coords,
                    }
                )

            if len(coverage_valid) < int(n_models):
                insufficient_coverage_count += 1
                if len(insufficient_coverage_examples) < 8:
                    insufficient_coverage_examples.append(f"{tid}:{len(coverage_valid)}/{n_models}")
                continue

            _assign_qa_scores(
                candidates=coverage_valid,
                target_length=target_len,
                qa_model_type=str(qa_model_type),
                qa_model=qa_model,
                qa_gnn_runtime=qa_gnn_runtime,
                location=location,
                heuristic_mode="artifacts",
            )

            qa_pool = sorted(
                coverage_valid,
                key=lambda c: (-float(c["qa_score"]), -float(c["coverage"]), -float(c["similarity"]), c["train_id"]),
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
                            "branch": "rnapro",
                            "target_id": tid,
                            "ID": f"{tid}_{resid}",
                            "resid": resid,
                            "resname": base,
                            "model_id": model_id,
                            "x": float(xyz[0]),
                            "y": float(xyz[1]),
                            "z": float(xyz[2]),
                            "template_uid": f"rnapro_train:{cand['train_id']}",
                            "similarity": float(cand["similarity"]),
                            "coverage": float(cand["coverage"]),
                            "qa_score": float(cand.get("qa_score", 0.0)),
                        }
                    )
                wrote_target = True
                if len(buffer) >= int(chunk_size):
                    _flush()
                    assert_memory_budget(stage="RNAPRO_INFER", location=location, budget_mb=memory_budget_mb)
            if wrote_target:
                targets_written.add(tid)

        _flush()
        if writer is not None:
            writer.close()
            writer = None
        assert_memory_budget(stage="RNAPRO_INFER", location=location, budget_mb=memory_budget_mb)

        if no_candidates_count > 0:
            raise_error(
                "RNAPRO_INFER",
                location,
                "alvos sem candidatos suficientes apos filtro temporal",
                impact=str(no_candidates_count),
                examples=no_candidates_examples,
            )
        if insufficient_coverage_count > 0:
            raise_error(
                "RNAPRO_INFER",
                location,
                "alvos sem modelos suficientes apos filtro de cobertura",
                impact=str(insufficient_coverage_count),
                examples=insufficient_coverage_examples,
            )
        if rows_written == 0:
            raise_error("RNAPRO_INFER", location, "nenhuma predicao gerada", impact="0", examples=[])

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
            "model_dir": _rel(model_dir, repo_root),
            "target_sequences": _rel(target_sequences_path, repo_root),
            "predictions_long": _rel(out_path, repo_root),
        },
        "params": {
            "n_models": int(n_models),
            "min_coverage": float(min_coverage),
            "rerank_pool_multiplier": int(rerank_pool_multiplier),
            "use_template": "none",
            "template_source": "tbm",
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
            "n_targets": int(len(targets_written)),
            "chunk_size": int(chunk_size),
            "selected_candidates": int(selected_candidates_total),
            "selected_avg_mapped": 0.0 if selected_candidates_total == 0 else float(selected_mapped_total) / float(selected_candidates_total),
            "selected_avg_match": 0.0 if selected_candidates_total == 0 else float(selected_match_total) / float(selected_candidates_total),
            "selected_avg_mismatch": 0.0 if selected_candidates_total == 0 else float(selected_mismatch_total) / float(selected_candidates_total),
            "selected_avg_chem_compatible": 0.0
            if selected_candidates_total == 0
            else float(selected_chem_total) / float(selected_candidates_total),
        },
        "compute": backend.to_manifest_dict(),
        "sha256": {
            "predictions_long.parquet": sha256_file(out_path),
            "qa_model.json": None if qa_model_sha256 is None else str(qa_model_sha256),
            "qa_model_weights.pt": None if qa_model_weights_sha256 is None else str(qa_model_weights_sha256),
        },
    }
    manifest_path = out_path.parent / "rnapro_infer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
