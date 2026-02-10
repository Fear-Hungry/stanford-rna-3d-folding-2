from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

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
) -> RetrievalResult:
    """
    Retrieve top-k template candidates per target with strict temporal filtering.
    """
    location = "src/rna3d_local/retrieval.py:retrieve_template_candidates"
    for p in (template_index_path, target_sequences_path):
        if not p.exists():
            raise_error("RETRIEVAL", location, "arquivo obrigatorio ausente", impact="1", examples=[str(p)])
    if top_k <= 0:
        raise_error("RETRIEVAL", location, "top_k deve ser > 0", impact="1", examples=[str(top_k)])
    if kmer_size <= 0:
        raise_error("RETRIEVAL", location, "kmer_size deve ser > 0", impact="1", examples=[str(kmer_size)])

    idx = pl.read_parquet(template_index_path)
    for col in ("template_uid", "template_id", "source", "sequence", "release_date"):
        if col not in idx.columns:
            raise_error("RETRIEVAL", location, "template_index sem coluna obrigatoria", impact="1", examples=[col])
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

    tgt = pl.read_csv(target_sequences_path, infer_schema_length=1000)
    for col in ("target_id", "sequence", "temporal_cutoff"):
        if col not in tgt.columns:
            raise_error("RETRIEVAL", location, "target_sequences sem coluna obrigatoria", impact="1", examples=[col])
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

    idx_rows = idx.select("template_uid", "template_id", "source", "sequence", "release_date").to_dicts()
    rows: list[dict] = []
    no_candidates: list[str] = []
    for target in tgt.select("target_id", "sequence", "temporal_cutoff").to_dicts():
        tid = str(target["target_id"])
        seq = str(target["sequence"])
        cutoff = target["temporal_cutoff"]
        if cutoff is None:
            raise_error("RETRIEVAL", location, "temporal_cutoff nulo apos cast", impact="1", examples=[tid])

        target_kmers = _kmer_set(seq, kmer_size)
        scored: list[tuple[float, str, str, str, object]] = []
        for tpl in idx_rows:
            if tpl["release_date"] is None or tpl["release_date"] > cutoff:
                continue
            sim = _jaccard(target_kmers, _kmer_set(str(tpl["sequence"]), kmer_size))
            scored.append(
                (
                    sim,
                    str(tpl["template_uid"]),
                    str(tpl["template_id"]),
                    str(tpl["source"]),
                    tpl["release_date"],
                )
            )
        if not scored:
            no_candidates.append(tid)
            continue
        scored.sort(key=lambda x: (-x[0], x[1]))
        for rank, (sim, template_uid, template_id, source, release_date) in enumerate(scored[:top_k], start=1):
            rows.append(
                {
                    "target_id": tid,
                    "template_uid": template_uid,
                    "template_id": template_id,
                    "source": source,
                    "rank": rank,
                    "similarity": float(sim),
                    "template_release_date": release_date,
                    "target_temporal_cutoff": cutoff,
                }
            )

    if no_candidates:
        raise_error(
            "RETRIEVAL",
            location,
            "sem candidatos apos filtro temporal",
            impact=str(len(no_candidates)),
            examples=no_candidates[:8],
        )
    if not rows:
        raise_error("RETRIEVAL", location, "nenhum candidato gerado", impact="0", examples=[])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pl.DataFrame(rows).sort(["target_id", "rank"])
    out_df.write_parquet(out_path)

    manifest = {
        "created_utc": _utc_now(),
        "paths": {
            "template_index": _rel(template_index_path, repo_root),
            "target_sequences": _rel(target_sequences_path, repo_root),
            "candidates": _rel(out_path, repo_root),
        },
        "params": {"top_k": top_k, "kmer_size": kmer_size},
        "stats": {"n_rows": int(out_df.height), "n_targets": int(out_df.get_column("target_id").n_unique())},
        "sha256": {"candidates.parquet": sha256_file(out_path)},
    }
    manifest_path = out_path.parent / "retrieval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RetrievalResult(candidates_path=out_path, manifest_path=manifest_path)
