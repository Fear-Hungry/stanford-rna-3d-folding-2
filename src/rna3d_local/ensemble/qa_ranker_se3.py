from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ..contracts import require_columns
from ..errors import raise_error
from ..io_tables import read_table, write_table
from ..utils import rel_or_abs, sha256_file, utc_now_iso, write_json
from .diversity import average_similarity, build_sample_vectors


@dataclass(frozen=True)
class RankSe3Result:
    ranked_path: Path
    manifest_path: Path


def _load_qa_config(path: Path | None, *, stage: str, location: str) -> dict[str, float]:
    if path is None:
        return {"compactness": 0.6, "smoothness": 0.4}
    if not path.exists():
        raise_error(stage, location, "qa_config ausente", impact="1", examples=[str(path)])
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ["compactness", "smoothness"]:
        if key not in payload:
            raise_error(stage, location, "qa_config sem campo obrigatorio", impact="1", examples=[key])
    return {"compactness": float(payload["compactness"]), "smoothness": float(payload["smoothness"])}


def _sample_quality(sample_df: pl.DataFrame, *, weights: dict[str, float]) -> float:
    coords = sample_df.select("x", "y", "z").to_numpy().astype(np.float64)
    center = coords.mean(axis=0, keepdims=True)
    radius = np.sqrt(((coords - center) ** 2).sum(axis=1))
    compactness = float(1.0 / (1.0 + float(radius.mean())))
    if coords.shape[0] > 1:
        steps = np.sqrt(((coords[1:] - coords[:-1]) ** 2).sum(axis=1))
        smoothness = float(1.0 / (1.0 + float(np.std(steps))))
    else:
        smoothness = 1.0
    return (weights["compactness"] * compactness) + (weights["smoothness"] * smoothness)


def rank_se3_ensemble(
    *,
    repo_root: Path,
    candidates_path: Path,
    out_path: Path,
    qa_config_path: Path | None,
    diversity_lambda: float,
) -> RankSe3Result:
    stage = "RANK_SE3"
    location = "src/rna3d_local/ensemble/qa_ranker_se3.py:rank_se3_ensemble"
    if diversity_lambda < 0:
        raise_error(stage, location, "diversity_lambda invalido (<0)", impact="1", examples=[str(diversity_lambda)])
    candidates = read_table(candidates_path, stage=stage, location=location)
    require_columns(
        candidates,
        ["target_id", "sample_id", "resid", "resname", "x", "y", "z"],
        stage=stage,
        location=location,
        label="se3_candidates",
    )
    qa_weights = _load_qa_config(qa_config_path, stage=stage, location=location)

    rows: list[dict[str, object]] = []
    for target_id, target_df in candidates.group_by("target_id", maintain_order=True):
        tid = str(target_id[0]) if isinstance(target_id, tuple) else str(target_id)
        vectors = build_sample_vectors(target_df, stage=stage, location=location)
        sample_scores: list[tuple[str, float, float, float]] = []
        for sample_id, sample_df in target_df.group_by("sample_id", maintain_order=True):
            sid = str(sample_id[0]) if isinstance(sample_id, tuple) else str(sample_id)
            qa_score = _sample_quality(sample_df.sort("resid"), weights=qa_weights)
            diversity_penalty = average_similarity(sid, vectors)
            final_score = float(qa_score) - (float(diversity_lambda) * float(diversity_penalty))
            sample_scores.append((sid, qa_score, diversity_penalty, final_score))
        if not sample_scores:
            raise_error(stage, location, "target sem samples para ranking", impact="1", examples=[tid])

        score_df = pl.DataFrame(
            {
                "target_id": [tid for _ in sample_scores],
                "sample_id": [item[0] for item in sample_scores],
                "qa_score": [float(item[1]) for item in sample_scores],
                "diversity_penalty": [float(item[2]) for item in sample_scores],
                "final_score": [float(item[3]) for item in sample_scores],
            }
        ).sort("final_score", descending=True)
        rank_df = score_df.with_columns(pl.int_range(1, pl.len() + 1).alias("rank"))
        joined = target_df.join(rank_df, on=["target_id", "sample_id"], how="left")
        missing = joined.filter(pl.col("rank").is_null())
        if missing.height > 0:
            raise_error(stage, location, "falha de join no ranking", impact=str(int(missing.height)), examples=[tid])
        rows.extend(joined.to_dicts())

    ranked = pl.DataFrame(rows).sort(["target_id", "rank", "resid"])
    write_table(ranked, out_path)
    manifest_path = out_path.parent / "rank_se3_manifest.json"
    write_json(
        manifest_path,
        {
            "created_utc": utc_now_iso(),
            "params": {"diversity_lambda": float(diversity_lambda), "qa_weights": qa_weights},
            "paths": {
                "candidates": rel_or_abs(candidates_path, repo_root),
                "qa_config": None if qa_config_path is None else rel_or_abs(qa_config_path, repo_root),
                "ranked": rel_or_abs(out_path, repo_root),
            },
            "stats": {"n_rows": int(ranked.height), "n_targets": int(ranked.get_column("target_id").n_unique())},
            "sha256": {"ranked.parquet": sha256_file(out_path)},
        },
    )
    return RankSe3Result(ranked_path=out_path, manifest_path=manifest_path)
