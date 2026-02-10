from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import xxhash

from ..alignment import compute_coverage, map_target_to_template_positions, project_target_coordinates
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


def infer_rnapro(
    *,
    repo_root: Path,
    model_dir: Path,
    target_sequences_path: Path,
    out_path: Path,
    n_models: int | None = None,
    min_coverage: float | None = None,
) -> Path:
    """
    Inference for RNAPro proxy model using nearest-neighbor feature retrieval + alignment projection.
    """
    location = "src/rna3d_local/rnapro/infer.py:infer_rnapro"
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

    features_df = pl.read_parquet(features_path).sort("target_id")
    seq_df = pl.read_parquet(seq_lookup_path).sort("target_id")
    coords_df = pl.read_parquet(coords_path)
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
    train_ids = features_df.get_column("target_id").to_list()
    train_matrix = features_df.select(feat_cols).to_numpy()
    if train_matrix.shape[0] == 0:
        raise_error("RNAPRO_INFER", location, "matriz de treino vazia", impact="0", examples=[])

    seq_map = {str(r["target_id"]): str(r["sequence"]) for r in seq_df.select("target_id", "sequence").to_dicts()}
    release_map = {str(r["target_id"]): r["release_date"] for r in seq_df.select("target_id", "release_date").to_dicts()}
    coords_map: dict[str, dict[int, tuple[float, float, float]]] = {}
    for r in coords_df.select("target_id", "resid", "x", "y", "z").to_dicts():
        tid = str(r["target_id"])
        coords_map.setdefault(tid, {})
        coords_map[tid][int(r["resid"])] = (float(r["x"]), float(r["y"]), float(r["z"]))

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

    rows: list[dict] = []
    no_candidates: list[str] = []
    low_coverage: list[str] = []
    for target in targets.select("target_id", "sequence", "temporal_cutoff").to_dicts():
        tid = str(target["target_id"])
        seq = str(target["sequence"])
        cutoff = target["temporal_cutoff"]
        feat = _hash_kmer_features(seq, k=config.kmer_size, dim=config.feature_dim, seed=config.seed)
        feat_norm = float(np.linalg.norm(feat))
        if feat_norm == 0:
            raise_error("RNAPRO_INFER", location, "feature nula para alvo", impact="1", examples=[tid])
        sims = train_matrix @ feat
        scored: list[tuple[float, str]] = []
        for idx, sid in enumerate(train_ids):
            rel = release_map.get(str(sid))
            if rel is None or rel > cutoff:
                continue
            scored.append((float(sims[idx]), str(sid)))
        if len(scored) < use_n_models:
            no_candidates.append(f"{tid}:{len(scored)}")
            continue
        scored.sort(key=lambda x: (-x[0], x[1]))
        target_len = len(seq)
        for model_id, (sim, train_id) in enumerate(scored[:use_n_models], start=1):
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
            mapping = map_target_to_template_positions(
                target_sequence=seq,
                template_sequence=tpl_seq,
                location=location,
            )
            valid_mapping = {t: s for t, s in mapping.items() if s in tpl_coords}
            cov = compute_coverage(mapped_positions=len(valid_mapping), target_length=target_len, location=location)
            if cov < use_min_coverage:
                low_coverage.append(f"{tid}:{train_id}:{cov:.4f}")
                continue
            coords = project_target_coordinates(
                target_length=target_len,
                mapping=valid_mapping,
                template_coordinates=tpl_coords,
                location=location,
            )
            for resid, (base, xyz) in enumerate(zip(seq, coords, strict=True), start=1):
                rows.append(
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
                        "template_uid": f"rnapro_train:{train_id}",
                        "similarity": float(sim),
                        "coverage": float(cov),
                    }
                )

    if no_candidates:
        raise_error(
            "RNAPRO_INFER",
            location,
            "alvos sem candidatos suficientes apos filtro temporal",
            impact=str(len(no_candidates)),
            examples=no_candidates[:8],
        )
    if low_coverage:
        raise_error(
            "RNAPRO_INFER",
            location,
            "cobertura abaixo do minimo",
            impact=str(len(low_coverage)),
            examples=low_coverage[:8],
        )
    if not rows:
        raise_error("RNAPRO_INFER", location, "nenhuma predicao gerada", impact="0", examples=[])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pl.DataFrame(rows).sort(["target_id", "model_id", "resid"])
    out_df.write_parquet(out_path)

    manifest = {
        "created_utc": _utc_now(),
        "paths": {
            "model_dir": _rel(model_dir, repo_root),
            "target_sequences": _rel(target_sequences_path, repo_root),
            "predictions_long": _rel(out_path, repo_root),
        },
        "params": {"n_models": use_n_models, "min_coverage": use_min_coverage},
        "stats": {"n_rows": int(out_df.height), "n_targets": int(out_df.get_column("target_id").n_unique())},
        "sha256": {"predictions_long.parquet": sha256_file(out_path)},
    }
    manifest_path = out_path.parent / "rnapro_infer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
