from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..contracts import require_columns
from ..errors import raise_error
from ..io_tables import read_table, write_table
from ..utils import rel_or_abs, sha256_file, utc_now_iso, write_json
from .diversity import SampleCandidate, build_sample_vectors, estimate_clash_ratio, prune_low_quality_half, select_cluster_medoids


@dataclass(frozen=True)
class SelectTop5Se3Result:
    predictions_path: Path
    manifest_path: Path


def select_top5_se3(
    *,
    repo_root: Path,
    ranked_path: Path,
    out_path: Path,
    n_models: int,
    diversity_lambda: float,
) -> SelectTop5Se3Result:
    stage = "SELECT_TOP5_SE3"
    location = "src/rna3d_local/ensemble/select_top5.py:select_top5_se3"
    if n_models <= 0:
        raise_error(stage, location, "n_models deve ser > 0", impact="1", examples=[str(n_models)])
    ranked = read_table(ranked_path, stage=stage, location=location)
    require_columns(
        ranked,
        ["target_id", "sample_id", "resid", "resname", "x", "y", "z", "qa_score", "final_score"],
        stage=stage,
        location=location,
        label="ranked_se3",
    )
    qa_missing_global = ranked.filter(pl.col("qa_score").is_null() | pl.col("final_score").is_null())
    if qa_missing_global.height > 0:
        examples = (
            qa_missing_global.with_columns(
                (pl.col("target_id") + pl.lit(":") + pl.col("sample_id") + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k")
            )
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(
            stage,
            location,
            "qa_score/final_score ausente no ranked_se3",
            impact=str(int(qa_missing_global.height)),
            examples=[str(item) for item in examples],
        )

    out_parts: list[pl.DataFrame] = []
    selected_count: list[dict[str, object]] = []
    for target_id, target_df in ranked.group_by("target_id", maintain_order=True):
        tid = str(target_id[0]) if isinstance(target_id, tuple) else str(target_id)
        summary = (
            target_df.select("target_id", "sample_id", "qa_score", "final_score")
            .group_by(["target_id", "sample_id"])
            .agg(
                pl.col("qa_score").max().alias("qa_score"),
                pl.col("final_score").max().alias("score"),
            )
            .sort("score", descending=True)
        )
        if summary.height < n_models:
            raise_error(
                stage,
                location,
                "samples insuficientes para top5 se3",
                impact=str(max(0, n_models - int(summary.height))),
                examples=[tid],
            )
        vectors = build_sample_vectors(target_df, stage=stage, location=location)
        candidates: list[SampleCandidate] = []
        for row in summary.iter_rows(named=True):
            sample_id = str(row["sample_id"])
            score = float(row["score"])
            sample_part = target_df.filter(pl.col("sample_id") == sample_id).sort("resid")
            clash_ratio = float(estimate_clash_ratio(sample_part, min_distance=2.1, covalent_skip=1))
            adjusted = float(score) - (0.40 * clash_ratio)
            candidates.append(
                SampleCandidate(
                    sample_id=sample_id,
                    score=float(score),
                    clash_ratio=float(clash_ratio),
                    adjusted_score=float(adjusted),
                )
            )
        kept = prune_low_quality_half(
            candidates=candidates,
            keep_fraction=0.50,
            min_keep=int(n_models),
            stage=stage,
            location=location,
        )
        kept_ids = [item.sample_id for item in kept]
        kept_scores = [(item.sample_id, float(item.adjusted_score)) for item in kept]
        vectors_kept = {sample_id: vectors[sample_id] for sample_id in kept_ids if sample_id in vectors}
        if len(vectors_kept) != len(kept_ids):
            missing = [sample_id for sample_id in kept_ids if sample_id not in vectors_kept]
            raise_error(
                stage,
                location,
                "vetor estrutural ausente apos pre-filtro",
                impact=str(int(len(missing))),
                examples=[str(item) for item in missing[:8]],
            )
        selected_ids, cluster_count = select_cluster_medoids(
            sample_scores=kept_scores,
            vectors=vectors_kept,
            n_select=int(n_models),
            lambda_diversity=float(diversity_lambda),
            stage=stage,
            location=location,
        )
        if len(selected_ids) != int(n_models):
            raise_error(
                stage,
                location,
                "falha na selecao diversa top5",
                impact=str(int(n_models) - len(selected_ids)),
                examples=[tid],
            )
        for model_id, sample_id in enumerate(selected_ids, start=1):
            part = (
                target_df.filter(pl.col("sample_id") == sample_id)
                .sort("resid")
                .with_columns(
                    pl.lit(int(model_id)).alias("model_id"),
                    pl.lit("generative_se3").alias("source"),
                    pl.col("qa_score").cast(pl.Float64).alias("confidence"),
                )
                .select("target_id", "model_id", "resid", "resname", "x", "y", "z", "source", "confidence", "sample_id")
            )
            out_parts.append(part)
        selected_count.append(
            {
                "target_id": tid,
                "n_candidates": int(len(candidates)),
                "n_after_prune": int(len(kept)),
                "cluster_count": int(cluster_count),
                "selected_ids": [str(item) for item in selected_ids],
            }
        )

    out = pl.concat(out_parts, how="vertical_relaxed").sort(["target_id", "model_id", "resid"])
    write_table(out, out_path)
    manifest_path = out_path.parent / "select_top5_se3_manifest.json"
    write_json(
        manifest_path,
        {
            "created_utc": utc_now_iso(),
            "params": {"n_models": int(n_models), "diversity_lambda": float(diversity_lambda)},
            "paths": {
                "ranked": rel_or_abs(ranked_path, repo_root),
                "predictions": rel_or_abs(out_path, repo_root),
            },
            "stats": {
                "n_rows": int(out.height),
                "n_targets": int(out.get_column("target_id").n_unique()),
                "selection": selected_count,
            },
            "sha256": {"predictions.parquet": sha256_file(out_path)},
        },
    )
    return SelectTop5Se3Result(predictions_path=out_path, manifest_path=manifest_path)
