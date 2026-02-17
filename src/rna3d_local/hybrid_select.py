from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .contracts import require_columns
from .errors import raise_error
from .io_tables import read_table, write_table
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json
from .ensemble.diversity import SampleCandidate, build_sample_vectors, estimate_clash_ratio, prune_low_quality_half, select_cluster_medoids


@dataclass(frozen=True)
class HybridTop5Result:
    predictions_path: Path
    manifest_path: Path


def select_top5_hybrid(
    *,
    repo_root: Path,
    candidates_path: Path,
    out_path: Path,
    n_models: int = 5,
    diversity_lambda: float = 0.35,
) -> HybridTop5Result:
    stage = "SELECT_TOP5_HYBRID"
    location = "src/rna3d_local/hybrid_select.py:select_top5_hybrid"
    if n_models <= 0:
        raise_error(stage, location, "n_models deve ser > 0", impact="1", examples=[str(n_models)])
    if float(diversity_lambda) < 0.0:
        raise_error(stage, location, "diversity_lambda invalido (>=0)", impact="1", examples=[str(diversity_lambda)])

    candidates = read_table(candidates_path, stage=stage, location=location)
    require_columns(
        candidates,
        ["target_id", "model_id", "resid", "resname", "x", "y", "z", "source", "confidence"],
        stage=stage,
        location=location,
        label="candidates",
    )
    key_dup = candidates.group_by(["target_id", "source", "model_id", "resid"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if key_dup.height > 0:
        examples = (
            key_dup.with_columns(
                (
                    pl.col("target_id")
                    + pl.lit(":")
                    + pl.col("source")
                    + pl.lit(":")
                    + pl.col("model_id").cast(pl.Utf8)
                    + pl.lit(":")
                    + pl.col("resid").cast(pl.Utf8)
                ).alias("k")
            )
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(stage, location, "candidates com chave duplicada", impact=str(key_dup.height), examples=[str(x) for x in examples])

    cand = candidates.with_columns(
        (
            pl.col("source").cast(pl.Utf8)
            + pl.lit(":")
            + pl.col("model_id").cast(pl.Int32).cast(pl.Utf8)
        ).alias("sample_id")
    )
    out_parts: list[pl.DataFrame] = []
    selected_stats: list[dict[str, object]] = []
    for target_id, target_df in cand.group_by("target_id", maintain_order=True):
        tid = str(target_id[0]) if isinstance(target_id, tuple) else str(target_id)
        summary = (
            target_df.group_by(["target_id", "sample_id", "source", "model_id"])
            .agg(
                pl.col("confidence").mean().alias("mean_confidence"),
                pl.col("confidence").null_count().alias("n_conf_null"),
                pl.len().alias("n_rows"),
            )
            .sort(["mean_confidence", "source", "model_id"], descending=[True, False, False])
        )
        if summary.height < int(n_models):
            raise_error(
                stage,
                location,
                "candidatos insuficientes para top5 hybrid",
                impact=str(int(n_models) - int(summary.height)),
                examples=[tid],
            )

        vectors = build_sample_vectors(target_df)
        candidates_scored: list[SampleCandidate] = []
        for row in summary.iter_rows(named=True):
            sample_id = str(row["sample_id"])
            mean_conf = row["mean_confidence"]
            score = float(mean_conf) if mean_conf is not None else 0.0
            sample_part = target_df.filter(pl.col("sample_id") == sample_id).sort("resid")
            clash_ratio = float(estimate_clash_ratio(sample_part, min_distance=2.1, covalent_skip=1))
            adjusted = float(score) - (0.40 * clash_ratio)
            candidates_scored.append(
                SampleCandidate(
                    sample_id=sample_id,
                    score=float(score),
                    clash_ratio=float(clash_ratio),
                    adjusted_score=float(adjusted),
                )
            )
        kept = prune_low_quality_half(
            candidates=candidates_scored,
            keep_fraction=0.50,
            min_keep=int(n_models),
            stage=stage,
            location=location,
        )
        kept_ids = [item.sample_id for item in kept]
        vectors_kept = {sample_id: vectors[sample_id] for sample_id in kept_ids if sample_id in vectors}
        if len(vectors_kept) != len(kept_ids):
            missing = [sample_id for sample_id in kept_ids if sample_id not in vectors_kept]
            raise_error(
                stage,
                location,
                "vetor estrutural ausente apos pre-filtro (hybrid)",
                impact=str(int(len(missing))),
                examples=[str(item) for item in missing[:8]],
            )
        kept_scores = [(item.sample_id, float(item.adjusted_score)) for item in kept]
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
                "falha na selecao diversa top5 (hybrid)",
                impact=str(int(n_models) - len(selected_ids)),
                examples=[tid],
            )
        selected_map = {sid: int(idx) for idx, sid in enumerate(selected_ids, start=1)}
        selected_df = (
            target_df.filter(pl.col("sample_id").is_in(selected_ids))
            .with_columns(pl.col("sample_id").replace(selected_map).cast(pl.Int32).alias("model_id"))
            .select(
                pl.col("target_id").cast(pl.Utf8),
                pl.col("model_id").cast(pl.Int32),
                pl.col("resid").cast(pl.Int32),
                pl.col("resname").cast(pl.Utf8),
                pl.col("x").cast(pl.Float64),
                pl.col("y").cast(pl.Float64),
                pl.col("z").cast(pl.Float64),
                pl.col("source").cast(pl.Utf8),
                pl.col("confidence").cast(pl.Float64),
            )
            .sort(["target_id", "model_id", "resid"])
        )
        out_parts.append(selected_df)
        selected_stats.append(
            {
                "target_id": tid,
                "n_candidates": int(len(candidates_scored)),
                "n_after_prune": int(len(kept)),
                "cluster_count": int(cluster_count),
                "selected_ids": [str(item) for item in selected_ids],
            }
        )

    out = pl.concat(out_parts, how="vertical_relaxed").sort(["target_id", "model_id", "resid"])
    write_table(out, out_path)
    manifest_path = out_path.parent / "select_top5_hybrid_manifest.json"
    write_json(
        manifest_path,
        {
            "created_utc": utc_now_iso(),
            "params": {"n_models": int(n_models), "diversity_lambda": float(diversity_lambda)},
            "paths": {
                "candidates": rel_or_abs(candidates_path, repo_root),
                "predictions": rel_or_abs(out_path, repo_root),
            },
            "stats": {
                "n_rows": int(out.height),
                "n_targets": int(out.get_column("target_id").n_unique()),
                "selection": selected_stats,
            },
            "sha256": {"predictions.parquet": sha256_file(out_path)},
        },
    )
    return HybridTop5Result(predictions_path=out_path, manifest_path=manifest_path)
