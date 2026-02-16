from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .contracts import require_columns
from .encoder import encode_sequences
from .errors import raise_error
from .io_tables import read_table, write_table
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class EmbeddingIndexResult:
    embeddings_path: Path
    index_path: Path | None
    manifest_path: Path


def _build_faiss_ivfpq(
    embeddings: np.ndarray,
    index_path: Path,
    *,
    stage: str,
    location: str,
) -> None:
    try:
        import faiss  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "faiss indisponivel para ann_engine=faiss_ivfpq", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    rows, dim = embeddings.shape
    if rows < 2:
        raise_error(stage, location, "embeddings insuficientes para treinar IVF-PQ", impact=str(rows), examples=[str(rows)])
    nlist = max(1, min(256, int(rows ** 0.5)))
    m = 8
    while dim % m != 0 and m > 1:
        m -= 1
    if m <= 0:
        raise_error(stage, location, "dimensao invalida para PQ", impact="1", examples=[str(dim)])
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
    index.train(embeddings)
    index.add(embeddings)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def build_embedding_index(
    *,
    repo_root: Path,
    template_index_path: Path,
    out_dir: Path,
    embedding_dim: int,
    encoder: str,
    model_path: Path | None,
    ann_engine: str = "faiss_ivfpq",
) -> EmbeddingIndexResult:
    stage = "EMBEDDING_INDEX"
    location = "src/rna3d_local/embedding_index.py:build_embedding_index"
    table = read_table(template_index_path, stage=stage, location=location)
    require_columns(table, ["template_uid", "sequence"], stage=stage, location=location, label="template_index")
    sequences = table.get_column("sequence").cast(pl.Utf8).to_list()
    uids = table.get_column("template_uid").cast(pl.Utf8).to_list()
    vectors = encode_sequences(
        [str(seq) for seq in sequences],
        encoder=encoder,
        embedding_dim=embedding_dim,
        model_path=model_path,
        stage=stage,
        location=location,
    )
    if vectors.shape[0] != len(uids):
        raise_error(stage, location, "numero de embeddings divergente", impact="1", examples=[f"vectors={vectors.shape[0]}", f"uids={len(uids)}"])

    embedding_rows = [{"template_uid": uid, "embedding": vec.tolist()} for uid, vec in zip(uids, vectors)]
    embeddings_df = pl.DataFrame(embedding_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = out_dir / "template_embeddings.parquet"
    write_table(embeddings_df, embeddings_path)

    mode = ann_engine.strip().lower()
    index_path: Path | None = None
    if mode == "faiss_ivfpq":
        index_path = out_dir / "faiss_ivfpq.index"
        _build_faiss_ivfpq(vectors.astype(np.float32), index_path, stage=stage, location=location)
    elif mode == "none":
        index_path = None
    else:
        raise_error(stage, location, "ann_engine invalido", impact="1", examples=[ann_engine])

    manifest_path = out_dir / "embedding_manifest.json"
    manifest = {
        "created_utc": utc_now_iso(),
        "encoder": encoder,
        "embedding_dim": int(embedding_dim),
        "ann_engine": mode,
        "paths": {
            "template_index": rel_or_abs(template_index_path, repo_root),
            "template_embeddings": rel_or_abs(embeddings_path, repo_root),
            "ann_index": None if index_path is None else rel_or_abs(index_path, repo_root),
        },
        "stats": {"n_templates": len(uids)},
        "sha256": {
            "template_embeddings.parquet": sha256_file(embeddings_path),
            "ann_index": None if index_path is None else sha256_file(index_path),
        },
    }
    write_json(manifest_path, manifest)
    return EmbeddingIndexResult(embeddings_path=embeddings_path, index_path=index_path, manifest_path=manifest_path)
