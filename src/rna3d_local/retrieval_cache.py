from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import xxhash

from .bigdata import TableReadConfig, collect_streaming, scan_table
from .errors import raise_error
from .utils import sha256_file


@dataclass(frozen=True)
class RetrievalCacheContext:
    enabled: bool
    cache_dir: Path | None
    cache_key: str | None
    cache_candidates_path: Path | None
    cache_meta_path: Path | None
    cache_rows: int
    cache_targets: int


def _build_cache_key(
    *,
    template_index_path: Path,
    target_sequences_path: Path,
    top_k: int,
    kmer_size: int,
    length_weight: float,
    refine_pool_size: int,
    refine_alignment_weight: float,
    refine_open_gap_score: float,
    refine_extend_gap_score: float,
    compute_backend: str,
    gpu_precision: str,
    gpu_hash_dim: int,
) -> str:
    payload = {
        "template_index_sha256": sha256_file(template_index_path),
        "target_sequences_sha256": sha256_file(target_sequences_path),
        "top_k": int(top_k),
        "kmer_size": int(kmer_size),
        "length_weight": float(length_weight),
        "refine_pool_size": int(refine_pool_size),
        "refine_alignment_weight": float(refine_alignment_weight),
        "refine_open_gap_score": float(refine_open_gap_score),
        "refine_extend_gap_score": float(refine_extend_gap_score),
        "compute_backend": str(compute_backend),
        "gpu_precision": str(gpu_precision),
        "gpu_hash_dim": int(gpu_hash_dim),
    }
    return xxhash.xxh64(json.dumps(payload, sort_keys=True)).hexdigest()


def _validate_cached_candidates(*, cache_candidates_path: Path, location: str) -> tuple[int, int]:
    cols = (
        "target_id",
        "template_uid",
        "template_id",
        "source",
        "rank",
        "similarity",
        "coarse_similarity",
        "alignment_similarity",
        "kmer_similarity",
        "length_compatibility",
        "template_release_date",
        "target_temporal_cutoff",
    )
    df = collect_streaming(
        lf=scan_table(
            config=TableReadConfig(
                path=cache_candidates_path,
                stage="RETRIEVAL",
                location=location,
                columns=cols,
            )
        ),
        stage="RETRIEVAL",
        location=location,
    )
    if df.height <= 0:
        raise_error(
            "RETRIEVAL",
            location,
            "cache de retrieval vazio",
            impact="0",
            examples=[str(cache_candidates_path)],
        )
    n_targets = int(df.select("target_id").n_unique())
    return int(df.height), int(n_targets)


def prepare_retrieval_cache(
    *,
    cache_dir: Path | None,
    template_index_path: Path,
    target_sequences_path: Path,
    top_k: int,
    kmer_size: int,
    length_weight: float,
    refine_pool_size: int,
    refine_alignment_weight: float,
    refine_open_gap_score: float,
    refine_extend_gap_score: float,
    compute_backend: str,
    gpu_precision: str,
    gpu_hash_dim: int,
    location: str,
) -> RetrievalCacheContext:
    if cache_dir is None:
        return RetrievalCacheContext(
            enabled=False,
            cache_dir=None,
            cache_key=None,
            cache_candidates_path=None,
            cache_meta_path=None,
            cache_rows=0,
            cache_targets=0,
        )

    cache_key = _build_cache_key(
        template_index_path=template_index_path,
        target_sequences_path=target_sequences_path,
        top_k=top_k,
        kmer_size=kmer_size,
        length_weight=length_weight,
        refine_pool_size=refine_pool_size,
        refine_alignment_weight=refine_alignment_weight,
        refine_open_gap_score=float(refine_open_gap_score),
        refine_extend_gap_score=float(refine_extend_gap_score),
        compute_backend=compute_backend,
        gpu_precision=gpu_precision,
        gpu_hash_dim=int(gpu_hash_dim),
    )
    cache_dir_resolved = cache_dir.resolve()
    cache_dir_resolved.mkdir(parents=True, exist_ok=True)
    cache_candidates_path = cache_dir_resolved / f"{cache_key}.candidates.parquet"
    cache_meta_path = cache_dir_resolved / f"{cache_key}.meta.json"

    cache_candidates_exists = cache_candidates_path.exists()
    cache_meta_exists = cache_meta_path.exists()
    if cache_candidates_exists != cache_meta_exists:
        raise_error(
            "RETRIEVAL",
            location,
            "cache de retrieval inconsistente (meta/candidates incompletos)",
            impact=f"candidates={int(cache_candidates_exists)} meta={int(cache_meta_exists)}",
            examples=[str(cache_candidates_path), str(cache_meta_path)],
        )

    cache_rows = 0
    cache_targets = 0
    if cache_candidates_exists and cache_meta_exists:
        cache_rows, cache_targets = _validate_cached_candidates(
            cache_candidates_path=cache_candidates_path,
            location=location,
        )
        try:
            cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        except Exception as error:  # noqa: BLE001
            raise_error(
                "RETRIEVAL",
                location,
                "cache meta invalido para retrieval",
                impact="1",
                examples=[f"{type(error).__name__}:{error}", str(cache_meta_path)],
            )
        cached_key = str(cache_meta.get("cache_key", ""))
        if cached_key != str(cache_key):
            raise_error(
                "RETRIEVAL",
                location,
                "cache meta com chave divergente",
                impact="1",
                examples=[f"expected={cache_key}", f"actual={cached_key}"],
            )

    return RetrievalCacheContext(
        enabled=True,
        cache_dir=cache_dir_resolved,
        cache_key=cache_key,
        cache_candidates_path=cache_candidates_path,
        cache_meta_path=cache_meta_path,
        cache_rows=int(cache_rows),
        cache_targets=int(cache_targets),
    )


def restore_retrieval_candidates_from_cache(
    *,
    cache: RetrievalCacheContext,
    out_path: Path,
) -> bool:
    if not cache.enabled:
        return False
    assert cache.cache_candidates_path is not None
    assert cache.cache_meta_path is not None
    if not (cache.cache_candidates_path.exists() and cache.cache_meta_path.exists()):
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_out_path.exists():
        tmp_out_path.unlink()
    shutil.copy2(cache.cache_candidates_path, tmp_out_path)
    if out_path.exists():
        out_path.unlink()
    tmp_out_path.replace(out_path)
    return True


def write_retrieval_cache(
    *,
    cache: RetrievalCacheContext,
    out_path: Path,
    created_utc: str,
    template_index_path: Path,
    target_sequences_path: Path,
    top_k: int,
    kmer_size: int,
    length_weight: float,
    refine_pool_size: int,
    refine_alignment_weight: float,
    refine_open_gap_score: float,
    refine_extend_gap_score: float,
    compute_backend: str,
    gpu_precision: str,
    gpu_hash_dim: int,
    rows_written: int,
    targets_written: int,
) -> None:
    if not cache.enabled:
        return
    assert cache.cache_candidates_path is not None
    assert cache.cache_meta_path is not None
    assert cache.cache_key is not None

    cache_tmp_candidates = cache.cache_candidates_path.with_suffix(cache.cache_candidates_path.suffix + ".tmp")
    cache_tmp_meta = cache.cache_meta_path.with_suffix(cache.cache_meta_path.suffix + ".tmp")
    if cache_tmp_candidates.exists():
        cache_tmp_candidates.unlink()
    if cache_tmp_meta.exists():
        cache_tmp_meta.unlink()

    shutil.copy2(out_path, cache_tmp_candidates)
    cache_meta_payload = {
        "created_utc": str(created_utc),
        "cache_key": str(cache.cache_key),
        "params": {
            "top_k": int(top_k),
            "kmer_size": int(kmer_size),
            "length_weight": float(length_weight),
            "refine_pool_size": int(refine_pool_size),
            "refine_alignment_weight": float(refine_alignment_weight),
            "refine_open_gap_score": float(refine_open_gap_score),
            "refine_extend_gap_score": float(refine_extend_gap_score),
            "compute_backend": str(compute_backend),
            "gpu_precision": str(gpu_precision),
            "gpu_hash_dim": int(gpu_hash_dim),
        },
        "sha256": {
            "template_index": sha256_file(template_index_path),
            "target_sequences": sha256_file(target_sequences_path),
            "candidates": sha256_file(out_path),
        },
        "stats": {
            "n_rows": int(rows_written),
            "n_targets": int(targets_written),
        },
    }
    cache_tmp_meta.write_text(json.dumps(cache_meta_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if cache.cache_candidates_path.exists():
        cache.cache_candidates_path.unlink()
    if cache.cache_meta_path.exists():
        cache.cache_meta_path.unlink()
    cache_tmp_candidates.replace(cache.cache_candidates_path)
    cache_tmp_meta.replace(cache.cache_meta_path)


__all__ = [
    "RetrievalCacheContext",
    "prepare_retrieval_cache",
    "restore_retrieval_candidates_from_cache",
    "write_retrieval_cache",
]
