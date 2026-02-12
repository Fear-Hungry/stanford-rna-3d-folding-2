from __future__ import annotations

import json
import heapq
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

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
from .errors import raise_error
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

            target_kmers = _kmer_set(seq, kmer_size)
            target_len = len(seq)
            pool_k = max(int(top_k), int(refine_pool_size))
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
        },
        "stats": {"n_rows": int(rows_written), "n_targets": int(len(targets_written)), "chunk_size": int(chunk_size)},
        "sha256": {"candidates.parquet": sha256_file(out_path)},
    }
    manifest_path = out_path.parent / "retrieval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RetrievalResult(candidates_path=out_path, manifest_path=manifest_path)
