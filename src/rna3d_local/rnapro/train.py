from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import xxhash

from ..errors import raise_error
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
    arr = np.zeros(dim, dtype=np.float64)
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


def train_rnapro(
    *,
    repo_root: Path,
    train_sequences_path: Path,
    train_labels_path: Path,
    out_dir: Path,
    config: RnaProConfig,
) -> Path:
    """
    Train a lightweight RNAPro proxy model:
    - builds deterministic sequence features
    - persists training coordinates for nearest-neighbor inference
    """
    location = "src/rna3d_local/rnapro/train.py:train_rnapro"
    for p in (train_sequences_path, train_labels_path):
        if not p.exists():
            raise_error("RNAPRO_TRAIN", location, "arquivo obrigatorio ausente", impact="1", examples=[str(p)])
    try:
        config.validate()
    except ValueError as e:
        raise_error("RNAPRO_TRAIN", location, "config invalida", impact="1", examples=[str(e)])

    seq_df = pl.read_csv(train_sequences_path, infer_schema_length=1000)
    required_seq = ["target_id", "sequence", "temporal_cutoff"]
    miss = [c for c in required_seq if c not in seq_df.columns]
    if miss:
        raise_error("RNAPRO_TRAIN", location, "train_sequences sem coluna obrigatoria", impact=str(len(miss)), examples=miss)
    seq_df = seq_df.select(
        pl.col("target_id").cast(pl.Utf8),
        pl.col("sequence").cast(pl.Utf8),
        pl.col("temporal_cutoff").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("release_date"),
    )
    if int(seq_df.get_column("release_date").null_count()) > 0:
        bad = seq_df.filter(pl.col("release_date").is_null()).get_column("target_id").head(8).to_list()
        raise_error(
            "RNAPRO_TRAIN",
            location,
            "temporal_cutoff invalido em train_sequences",
            impact=str(int(seq_df.get_column("release_date").null_count())),
            examples=bad,
        )

    dup_targets = (
        seq_df.group_by("target_id")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
        .get_column("target_id")
        .to_list()
    )
    if dup_targets:
        raise_error(
            "RNAPRO_TRAIN",
            location,
            "target_id duplicado em train_sequences",
            impact=str(len(dup_targets)),
            examples=dup_targets[:8],
        )

    scan = pl.scan_csv(train_labels_path, infer_schema_length=1000)
    label_cols = scan.collect_schema().names()
    required_labels = ["ID", "resid", "resname", "x_1", "y_1", "z_1"]
    miss_labels = [c for c in required_labels if c not in label_cols]
    if miss_labels:
        raise_error(
            "RNAPRO_TRAIN",
            location,
            "train_labels sem coluna obrigatoria",
            impact=str(len(miss_labels)),
            examples=miss_labels[:8],
        )

    train_ids = seq_df.get_column("target_id").to_list()
    labels_df = (
        scan.with_columns(pl.col("ID").cast(pl.Utf8).str.extract(r"^(.*)_\d+$", 1).alias("target_id"))
        .filter(pl.col("target_id").is_in(train_ids))
        .select(
            pl.col("target_id").cast(pl.Utf8),
            pl.col("resid").cast(pl.Int32),
            pl.col("resname").cast(pl.Utf8),
            pl.col("x_1").cast(pl.Float64).alias("x"),
            pl.col("y_1").cast(pl.Float64).alias("y"),
            pl.col("z_1").cast(pl.Float64).alias("z"),
        )
        .collect(engine="streaming")
    )
    if labels_df.height == 0:
        raise_error("RNAPRO_TRAIN", location, "nenhuma coordenada de treino encontrada", impact="0", examples=[])

    labeled_targets = set(labels_df.get_column("target_id").unique().to_list())
    missing_targets = [tid for tid in train_ids if tid not in labeled_targets]
    if missing_targets:
        raise_error(
            "RNAPRO_TRAIN",
            location,
            "targets sem labels no treino",
            impact=str(len(missing_targets)),
            examples=missing_targets[:8],
        )

    seq_sorted = seq_df.sort("target_id")
    feats: list[dict] = []
    feat_cols = [f"f_{i}" for i in range(config.feature_dim)]
    for row in seq_sorted.select("target_id", "sequence").to_dicts():
        vec = _hash_kmer_features(
            str(row["sequence"]),
            k=config.kmer_size,
            dim=config.feature_dim,
            seed=config.seed,
        )
        out = {"target_id": str(row["target_id"])}
        out.update({c: float(v) for c, v in zip(feat_cols, vec)})
        feats.append(out)

    features_df = pl.DataFrame(feats).sort("target_id")
    coords_df = labels_df.join(seq_df.select("target_id", "release_date"), on="target_id", how="inner").sort(
        ["target_id", "resid"]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "target_features.parquet"
    seq_lookup_path = out_dir / "sequence_lookup.parquet"
    coords_path = out_dir / "coordinates.parquet"
    features_df.write_parquet(features_path)
    seq_sorted.write_parquet(seq_lookup_path)
    coords_df.write_parquet(coords_path)

    model = {
        "created_utc": _utc_now(),
        "model_type": "rnapro_proxy_knn",
        "config": config.to_dict(),
        "stats": {
            "n_targets": int(seq_sorted.height),
            "n_coordinate_rows": int(coords_df.height),
            "feature_dim": int(config.feature_dim),
        },
        "paths": {
            "target_features": _rel(features_path, repo_root),
            "sequence_lookup": _rel(seq_lookup_path, repo_root),
            "coordinates": _rel(coords_path, repo_root),
        },
        "sha256": {
            "target_features.parquet": sha256_file(features_path),
            "sequence_lookup.parquet": sha256_file(seq_lookup_path),
            "coordinates.parquet": sha256_file(coords_path),
        },
    }
    model_path = out_dir / "model.json"
    model_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return model_path
