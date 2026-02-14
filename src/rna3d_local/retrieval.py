from __future__ import annotations

import json
import heapq
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
import xxhash

from .alignment import normalized_global_alignment_similarity
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
from .retrieval_cache import (
    prepare_retrieval_cache,
    restore_retrieval_candidates_from_cache,
    write_retrieval_cache,
)
from .utils import sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _kmer_set(seq: str, k: int) -> set[str]:
    s = (seq or "").strip().upper()
    if len(s) < k:
        return {s}
    return {s[i : i + k] for i in range(0, len(s) - k + 1)}


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
    if norm > 0.0:
        arr /= norm
    return arr


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 0.0
    return float(len(a & b)) / float(len(u))


def _length_compatibility(*, target_len: int, template_len: int) -> float:
    if target_len <= 0 or template_len <= 0:
        return 0.0
    den = float(max(target_len, template_len))
    if den <= 0.0:
        return 0.0
    return max(0.0, 1.0 - (abs(float(target_len - template_len)) / den))


@dataclass(frozen=True)
class RetrievalResult:
    candidates_path: Path
    manifest_path: Path


def retrieve_template_candidates(
    *,
    repo_root: Path,
    template_index_path: Path,
    target_sequences_path: Path,
    out_path: Path,
    top_k: int = 20,
    kmer_size: int = 3,
    length_weight: float = 0.15,
    refine_pool_size: int = 64,
    refine_alignment_weight: float = 0.25,
    refine_open_gap_score: float = -5.0,
    refine_extend_gap_score: float = -1.0,
    chunk_size: int = 200_000,
    compute_backend: str = "auto",
    gpu_memory_budget_mb: int = 12_288,
    gpu_precision: str = "fp32",
    gpu_hash_dim: int = 4096,
    cache_dir: Path | None = None,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    max_rows_in_memory: int = DEFAULT_MAX_ROWS_IN_MEMORY,
) -> RetrievalResult:
    """
    Retrieve top-k template candidates per target with strict temporal filtering.
    """
    location = "src/rna3d_local/retrieval.py:retrieve_template_candidates"
    assert_memory_budget(stage="RETRIEVAL", location=location, budget_mb=memory_budget_mb)
    for p in (template_index_path, target_sequences_path):
        if not p.exists():
            raise_error("RETRIEVAL", location, "arquivo obrigatorio ausente", impact="1", examples=[str(p)])
    if top_k <= 0:
        raise_error("RETRIEVAL", location, "top_k deve ser > 0", impact="1", examples=[str(top_k)])
    if kmer_size <= 0:
        raise_error("RETRIEVAL", location, "kmer_size deve ser > 0", impact="1", examples=[str(kmer_size)])
    if length_weight < 0.0 or length_weight > 1.0:
        raise_error("RETRIEVAL", location, "length_weight invalido [0,1]", impact="1", examples=[str(length_weight)])
    if refine_pool_size <= 0:
        raise_error("RETRIEVAL", location, "refine_pool_size deve ser > 0", impact="1", examples=[str(refine_pool_size)])
    if refine_alignment_weight < 0.0 or refine_alignment_weight > 1.0:
        raise_error(
            "RETRIEVAL",
            location,
            "refine_alignment_weight invalido [0,1]",
            impact="1",
            examples=[str(refine_alignment_weight)],
        )
    if chunk_size <= 0:
        raise_error("RETRIEVAL", location, "chunk_size deve ser > 0", impact="1", examples=[str(chunk_size)])
    if int(gpu_hash_dim) <= 0:
        raise_error("RETRIEVAL", location, "gpu_hash_dim deve ser > 0", impact="1", examples=[str(gpu_hash_dim)])
    backend = resolve_compute_backend(
        requested=str(compute_backend),
        precision=str(gpu_precision),
        gpu_memory_budget_mb=int(gpu_memory_budget_mb),
        stage="RETRIEVAL",
        location=location,
    )

    cache = prepare_retrieval_cache(
        cache_dir=cache_dir,
        template_index_path=template_index_path,
        target_sequences_path=target_sequences_path,
        top_k=top_k,
        kmer_size=kmer_size,
        length_weight=length_weight,
        refine_pool_size=refine_pool_size,
        refine_alignment_weight=refine_alignment_weight,
        refine_open_gap_score=float(refine_open_gap_score),
        refine_extend_gap_score=float(refine_extend_gap_score),
        compute_backend=str(backend.backend),
        gpu_precision=str(backend.precision),
        gpu_hash_dim=int(gpu_hash_dim),
        location=location,
    )
    cache_hit = False
    if restore_retrieval_candidates_from_cache(cache=cache, out_path=out_path):
        cache_hit = True
        assert cache.cache_dir is not None
        assert cache.cache_key is not None
        assert cache.cache_candidates_path is not None
        assert cache.cache_meta_path is not None
        manifest = {
            "created_utc": _utc_now(),
            "paths": {
                "template_index": _rel(template_index_path, repo_root),
                "target_sequences": _rel(target_sequences_path, repo_root),
                "candidates": _rel(out_path, repo_root),
            },
            "params": {
                "top_k": top_k,
                "kmer_size": kmer_size,
                "length_weight": length_weight,
                "refine_pool_size": refine_pool_size,
                "refine_alignment_weight": refine_alignment_weight,
                "refine_open_gap_score": float(refine_open_gap_score),
                "refine_extend_gap_score": float(refine_extend_gap_score),
                "compute_backend": str(backend.backend),
                "compute_backend_requested": str(backend.requested),
                "gpu_precision": str(backend.precision),
                "gpu_hash_dim": int(gpu_hash_dim),
                "gpu_memory_budget_mb": int(backend.gpu_memory_budget_mb),
            },
            "stats": {
                "n_rows": int(cache.cache_rows),
                "n_targets": int(cache.cache_targets),
                "chunk_size": int(chunk_size),
            },
            "compute": backend.to_manifest_dict(),
            "cache": {
                "enabled": True,
                "cache_dir": _rel(cache.cache_dir, repo_root),
                "cache_key": str(cache.cache_key),
                "cache_hit": True,
                "cache_candidates": _rel(cache.cache_candidates_path, repo_root),
                "cache_meta": _rel(cache.cache_meta_path, repo_root),
            },
            "sha256": {"candidates.parquet": sha256_file(out_path)},
        }
        manifest_path = out_path.parent / "retrieval_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return RetrievalResult(candidates_path=out_path, manifest_path=manifest_path)

    idx = collect_streaming(
        lf=scan_table(
            config=TableReadConfig(
                path=template_index_path,
                stage="RETRIEVAL",
                location=location,
                columns=("template_uid", "template_id", "source", "sequence", "release_date"),
            )
        ),
        stage="RETRIEVAL",
        location=location,
    )
    idx = idx.with_columns(
        pl.col("template_uid").cast(pl.Utf8),
        pl.col("template_id").cast(pl.Utf8),
        pl.col("source").cast(pl.Utf8),
        pl.col("sequence").cast(pl.Utf8),
        pl.col("release_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("release_date"),
    )
    if int(idx.get_column("release_date").null_count()) > 0:
        raise_error(
            "RETRIEVAL",
            location,
            "template_index com release_date invalida",
            impact=str(int(idx.get_column("release_date").null_count())),
            examples=[],
        )
    assert_row_budget(
        stage="RETRIEVAL",
        location=location,
        rows=int(idx.height),
        max_rows_in_memory=max_rows_in_memory,
        label="template_index",
    )

    tgt = collect_streaming(
        lf=scan_table(
            config=TableReadConfig(
                path=target_sequences_path,
                stage="RETRIEVAL",
                location=location,
                columns=("target_id", "sequence", "temporal_cutoff"),
            )
        ),
        stage="RETRIEVAL",
        location=location,
    )
    tgt = tgt.with_columns(
        pl.col("target_id").cast(pl.Utf8),
        pl.col("sequence").cast(pl.Utf8),
        pl.col("temporal_cutoff").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("temporal_cutoff"),
    )
    if int(tgt.get_column("temporal_cutoff").null_count()) > 0:
        bad = tgt.filter(pl.col("temporal_cutoff").is_null()).get_column("target_id").head(8).to_list()
        raise_error(
            "RETRIEVAL",
            location,
            "target_sequences com temporal_cutoff invalido",
            impact=str(int(tgt.get_column("temporal_cutoff").null_count())),
            examples=bad,
        )
    assert_row_budget(
        stage="RETRIEVAL",
        location=location,
        rows=int(idx.height + tgt.height),
        max_rows_in_memory=max_rows_in_memory,
        label="template_index+targets",
    )

    idx_rows: list[tuple[str, str, str, object, int, set[str], str]] = []
    for uid, tid, source, seq, release_date in idx.select(
        "template_uid",
        "template_id",
        "source",
        "sequence",
        "release_date",
    ).iter_rows():
        seq_s = str(seq)
        idx_rows.append((str(uid), str(tid), str(source), release_date, len(seq_s), _kmer_set(seq_s, kmer_size), seq_s))

    torch = None
    device = None
    template_features_t = None
    template_lengths_t = None
    release_dates = [row[3] for row in idx_rows]
    if backend.backend == "cuda":
        try:
            import torch as _torch  # noqa: PLC0415
        except Exception as e:  # noqa: BLE001
            raise_error("RETRIEVAL", location, "falha ao importar torch para backend CUDA", impact="1", examples=[f"{type(e).__name__}:{e}"])
        torch = _torch
        device = torch.device("cuda")
        dtype = torch.float16 if backend.precision == "fp16" else torch.float32
        feat_matrix = np.zeros((len(idx_rows), int(gpu_hash_dim)), dtype=np.float32)
        lengths = np.zeros((len(idx_rows),), dtype=np.float32)
        for i, (_uid, _tid, _source, _rd, tpl_len, _tpl_kmers, tpl_seq) in enumerate(idx_rows):
            feat_matrix[i, :] = _hash_kmer_features(str(tpl_seq), k=int(kmer_size), dim=int(gpu_hash_dim), seed=17)
            lengths[i] = float(tpl_len)
        template_features_t = torch.tensor(feat_matrix, dtype=dtype, device=device)
        template_lengths_t = torch.tensor(lengths, dtype=dtype, device=device)
        assert_memory_budget(stage="RETRIEVAL", location=location, budget_mb=memory_budget_mb)

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
    try:
        for tid, seq, cutoff in tgt.select("target_id", "sequence", "temporal_cutoff").sort("target_id").iter_rows():
            tid = str(tid)
            seq = str(seq)
            if cutoff is None:
                raise_error("RETRIEVAL", location, "temporal_cutoff nulo apos cast", impact="1", examples=[tid])

            target_len = len(seq)
            pool_k = max(int(top_k), int(refine_pool_size))
            if backend.backend == "cuda":
                assert torch is not None and device is not None and template_features_t is not None and template_lengths_t is not None
                target_feat = _hash_kmer_features(str(seq), k=int(kmer_size), dim=int(gpu_hash_dim), seed=17)
                dtype = torch.float16 if backend.precision == "fp16" else torch.float32
                target_feat_t = torch.tensor(target_feat, dtype=dtype, device=device)
                kmer_sim_t = torch.mv(template_features_t, target_feat_t)
                target_len_t = torch.tensor(float(target_len), dtype=dtype, device=device)
                den = torch.maximum(template_lengths_t, target_len_t)
                len_sim_t = 1.0 - (torch.abs(template_lengths_t - target_len_t) / torch.where(den <= 0, torch.ones_like(den), den))
                len_sim_t = torch.clamp(len_sim_t, min=0.0, max=1.0)
                coarse_t = ((1.0 - float(length_weight)) * kmer_sim_t) + (float(length_weight) * len_sim_t)
                valid_mask_np = np.asarray([(rd is not None and rd <= cutoff) for rd in release_dates], dtype=np.bool_)
                candidate_count = int(valid_mask_np.sum())
                if candidate_count == 0:
                    no_candidates_count += 1
                    if len(no_candidates_examples) < 8:
                        no_candidates_examples.append(tid)
                    continue
                valid_idx_np = np.flatnonzero(valid_mask_np)
                valid_idx_t = torch.tensor(valid_idx_np, dtype=torch.long, device=device)
                valid_scores_t = coarse_t[valid_idx_t]
                topn = min(int(pool_k), candidate_count)
                vals_t, ord_t = torch.topk(valid_scores_t, k=int(topn), largest=True, sorted=True)
                top_idx_t = valid_idx_t[ord_t]
                top_idx_np = top_idx_t.detach().cpu().numpy()
                top_vals_np = vals_t.detach().cpu().numpy()
                top_kmer_np = kmer_sim_t[top_idx_t].detach().cpu().numpy()
                top_len_np = len_sim_t[top_idx_t].detach().cpu().numpy()
                pool_ranked = []
                for i, tpl_idx in enumerate(top_idx_np.tolist()):
                    template_uid, template_id, source, release_date, _tpl_len, _tpl_kmers, tpl_seq = idx_rows[int(tpl_idx)]
                    pool_ranked.append(
                        (
                            float(top_vals_np[i]),
                            float(top_kmer_np[i]),
                            float(top_len_np[i]),
                            template_uid,
                            template_id,
                            source,
                            release_date,
                            tpl_seq,
                        )
                    )
                pool_ranked = sorted(pool_ranked, key=lambda x: (-x[0], -x[1], -x[2], x[3]))
            else:
                target_kmers = _kmer_set(seq, kmer_size)
                top_heap: list[tuple[float, float, float, str, str, str, object, str]] = []
                candidate_count = 0
                for template_uid, template_id, source, release_date, tpl_len, tpl_kmers, tpl_seq in idx_rows:
                    if release_date is None or release_date > cutoff:
                        continue
                    candidate_count += 1
                    jacc = _jaccard(target_kmers, tpl_kmers)
                    len_sim = _length_compatibility(target_len=target_len, template_len=tpl_len)
                    blended = ((1.0 - length_weight) * jacc) + (length_weight * len_sim)
                    item = (blended, jacc, len_sim, template_uid, template_id, source, release_date, tpl_seq)
                    if len(top_heap) < pool_k:
                        heapq.heappush(top_heap, item)
                    elif item > top_heap[0]:
                        heapq.heapreplace(top_heap, item)
                if candidate_count == 0:
                    no_candidates_count += 1
                    if len(no_candidates_examples) < 8:
                        no_candidates_examples.append(tid)
                    continue
                pool_ranked = sorted(top_heap, key=lambda x: (-x[0], -x[1], -x[2], x[3]))

            refined_ranked: list[tuple[float, float, float, float, float, str, str, str, object]] = []
            for coarse_sim, jacc, len_sim, template_uid, template_id, source, release_date, tpl_seq in pool_ranked:
                align_sim = (
                    normalized_global_alignment_similarity(
                        target_sequence=seq,
                        template_sequence=tpl_seq,
                        location=location,
                        open_gap_score=float(refine_open_gap_score),
                        extend_gap_score=float(refine_extend_gap_score),
                    )
                    if refine_alignment_weight > 0.0
                    else 0.0
                )
                final_sim = ((1.0 - refine_alignment_weight) * coarse_sim) + (refine_alignment_weight * align_sim)
                refined_ranked.append(
                    (
                        float(final_sim),
                        float(coarse_sim),
                        float(align_sim),
                        float(jacc),
                        float(len_sim),
                        template_uid,
                        template_id,
                        source,
                        release_date,
                    )
                )

            top_ranked = sorted(
                refined_ranked,
                key=lambda x: (-x[0], -x[2], -x[1], -x[3], -x[4], x[5]),
            )[:top_k]
            for rank, (sim, coarse_sim, align_sim, jacc, len_sim, template_uid, template_id, source, release_date) in enumerate(
                top_ranked, start=1
            ):
                buffer.append(
                    {
                        "target_id": tid,
                        "template_uid": template_uid,
                        "template_id": template_id,
                        "source": source,
                        "rank": rank,
                        "similarity": float(sim),
                        "coarse_similarity": float(coarse_sim),
                        "alignment_similarity": float(align_sim),
                        "kmer_similarity": float(jacc),
                        "length_compatibility": float(len_sim),
                        "template_release_date": release_date,
                        "target_temporal_cutoff": cutoff,
                    }
                )
            targets_written.add(tid)
            if len(buffer) >= chunk_size:
                _flush()
                assert_memory_budget(stage="RETRIEVAL", location=location, budget_mb=memory_budget_mb)

        _flush()
        if writer is not None:
            writer.close()
            writer = None
        assert_memory_budget(stage="RETRIEVAL", location=location, budget_mb=memory_budget_mb)

        if no_candidates_count > 0:
            raise_error(
                "RETRIEVAL",
                location,
                "sem candidatos apos filtro temporal",
                impact=str(no_candidates_count),
                examples=no_candidates_examples,
            )
        if rows_written == 0:
            raise_error("RETRIEVAL", location, "nenhum candidato gerado", impact="0", examples=[])
        if out_path.exists():
            out_path.unlink()
        tmp_out_path.replace(out_path)
        finalized = True
    finally:
        if writer is not None:
            writer.close()
        if not finalized and tmp_out_path.exists():
            tmp_out_path.unlink()

    write_retrieval_cache(
        cache=cache,
        out_path=out_path,
        created_utc=_utc_now(),
        template_index_path=template_index_path,
        target_sequences_path=target_sequences_path,
        top_k=int(top_k),
        kmer_size=int(kmer_size),
        length_weight=float(length_weight),
        refine_pool_size=int(refine_pool_size),
        refine_alignment_weight=float(refine_alignment_weight),
        refine_open_gap_score=float(refine_open_gap_score),
        refine_extend_gap_score=float(refine_extend_gap_score),
        compute_backend=str(backend.backend),
        gpu_precision=str(backend.precision),
        gpu_hash_dim=int(gpu_hash_dim),
        rows_written=int(rows_written),
        targets_written=int(len(targets_written)),
    )

    manifest = {
        "created_utc": _utc_now(),
        "paths": {
            "template_index": _rel(template_index_path, repo_root),
            "target_sequences": _rel(target_sequences_path, repo_root),
            "candidates": _rel(out_path, repo_root),
        },
        "params": {
            "top_k": top_k,
            "kmer_size": kmer_size,
            "length_weight": length_weight,
            "refine_pool_size": refine_pool_size,
            "refine_alignment_weight": refine_alignment_weight,
            "refine_open_gap_score": float(refine_open_gap_score),
            "refine_extend_gap_score": float(refine_extend_gap_score),
            "compute_backend": str(backend.backend),
            "compute_backend_requested": str(backend.requested),
            "gpu_precision": str(backend.precision),
            "gpu_hash_dim": int(gpu_hash_dim),
            "gpu_memory_budget_mb": int(backend.gpu_memory_budget_mb),
        },
        "stats": {"n_rows": int(rows_written), "n_targets": int(len(targets_written)), "chunk_size": int(chunk_size)},
        "compute": backend.to_manifest_dict(),
        "cache": {
            "enabled": bool(cache.enabled),
            "cache_dir": None if cache.cache_dir is None else _rel(cache.cache_dir, repo_root),
            "cache_key": None if cache.cache_key is None else str(cache.cache_key),
            "cache_hit": bool(cache_hit),
            "cache_candidates": None if cache.cache_candidates_path is None else _rel(cache.cache_candidates_path, repo_root),
            "cache_meta": None if cache.cache_meta_path is None else _rel(cache.cache_meta_path, repo_root),
        },
        "sha256": {"candidates.parquet": sha256_file(out_path)},
    }
    manifest_path = out_path.parent / "retrieval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RetrievalResult(candidates_path=out_path, manifest_path=manifest_path)
