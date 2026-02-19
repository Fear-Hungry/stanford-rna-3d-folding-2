from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


_REQUIRED_RANKED_COLUMNS = ["target_id", "sample_id", "resid", "resname", "x", "y", "z", "qa_score", "final_score"]
_CLASH_PENALTY = 0.40
_KEEP_FRACTION = 0.50
_MIN_DISTANCE = 2.1
_COVALENT_SKIP = 1
_TOP5_BASE_COLUMNS = ["target_id", "model_id", "resid", "resname", "x", "y", "z", "source", "confidence", "sample_id"]
_TOP5_OPTIONAL_COLUMNS = ["chain_index", "residue_index_1d"]


def _normalize_group_key(group_key: object) -> str:
    return str(group_key[0]) if isinstance(group_key, tuple) else str(group_key)


def _validate_select_params(*, n_models: int, diversity_lambda: float, stage: str, location: str) -> None:
    if int(n_models) <= 0:
        raise_error(stage, location, "n_models deve ser > 0", impact="1", examples=[str(n_models)])
    if float(diversity_lambda) < 0.0:
        raise_error(stage, location, "diversity_lambda invalido (>=0)", impact="1", examples=[str(diversity_lambda)])


def _validate_ranked_table(ranked: pl.DataFrame, *, stage: str, location: str) -> None:
    require_columns(
        ranked,
        _REQUIRED_RANKED_COLUMNS,
        stage=stage,
        location=location,
        label="ranked_se3",
    )
    if int(ranked.height) <= 0:
        raise_error(stage, location, "ranked_se3 vazio para selecao top5", impact="1", examples=[])
    qa_missing = ranked.filter(pl.col("qa_score").is_null() | pl.col("final_score").is_null())
    if qa_missing.height > 0:
        examples = (
            qa_missing.with_columns(
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
            impact=str(int(qa_missing.height)),
            examples=[str(item) for item in examples],
        )


def _build_sample_parts(target_df: pl.DataFrame, *, stage: str, location: str) -> dict[str, pl.DataFrame]:
    parts: dict[str, pl.DataFrame] = {}
    for raw_sample_id, sample_part in target_df.group_by("sample_id", maintain_order=True):
        sample_id = _normalize_group_key(raw_sample_id)
        if sample_id in parts:
            raise_error(stage, location, "sample_id duplicado apos agrupamento", impact="1", examples=[sample_id])
        parts[sample_id] = sample_part.sort("resid")
    if not parts:
        raise_error(stage, location, "target sem candidatos para top5", impact="1", examples=[])
    return parts


def _build_score_summary(sample_parts: dict[str, pl.DataFrame]) -> list[tuple[str, float, float]]:
    summary: list[tuple[str, float, float]] = []
    for sample_id, sample_part in sample_parts.items():
        qa_score = float(sample_part.get_column("qa_score").max())
        final_score = float(sample_part.get_column("final_score").max())
        summary.append((sample_id, qa_score, final_score))
    summary.sort(key=lambda item: float(item[2]), reverse=True)
    return summary


def _build_candidates(summary: list[tuple[str, float, float]], sample_parts: dict[str, pl.DataFrame]) -> list[SampleCandidate]:
    candidates: list[SampleCandidate] = []
    for sample_id, _qa_score, final_score in summary:
        sample_part = sample_parts[sample_id]
        clash_ratio = float(estimate_clash_ratio(sample_part, min_distance=_MIN_DISTANCE, covalent_skip=_COVALENT_SKIP))
        adjusted = float(final_score) - (_CLASH_PENALTY * clash_ratio)
        candidates.append(
            SampleCandidate(
                sample_id=sample_id,
                score=float(final_score),
                clash_ratio=clash_ratio,
                adjusted_score=float(adjusted),
            )
        )
    return candidates


def _build_vectors_for_kept(
    *,
    kept_ids: list[str],
    vectors: dict[str, Any],
    stage: str,
    location: str,
) -> dict[str, Any]:
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
    return vectors_kept


def _materialize_selected_parts(*, sample_parts: dict[str, pl.DataFrame], selected_ids: list[str]) -> list[pl.DataFrame]:
    out_parts: list[pl.DataFrame] = []
    for model_id, sample_id in enumerate(selected_ids, start=1):
        sample_part = sample_parts[sample_id]
        select_columns = list(_TOP5_BASE_COLUMNS)
        for col in _TOP5_OPTIONAL_COLUMNS:
            if col in sample_part.columns:
                select_columns.append(col)
        out_parts.append(
            sample_part.with_columns(
                pl.lit(int(model_id)).alias("model_id"),
                pl.lit("generative_se3").alias("source"),
                pl.col("qa_score").cast(pl.Float64).alias("confidence"),
            ).select(select_columns)
        )
    return out_parts


def _select_target_models(
    *,
    target_id: str,
    target_df: pl.DataFrame,
    n_models: int,
    diversity_lambda: float,
    stage: str,
    location: str,
) -> tuple[list[pl.DataFrame], dict[str, object]]:
    sample_parts = _build_sample_parts(target_df, stage=stage, location=location)
    summary = _build_score_summary(sample_parts)
    if len(summary) < int(n_models):
        raise_error(
            stage,
            location,
            "samples insuficientes para top5 se3",
            impact=str(max(0, int(n_models) - int(len(summary)))),
            examples=[target_id],
        )

    vectors = build_sample_vectors(target_df, stage=stage, location=location)
    candidates = _build_candidates(summary, sample_parts)
    kept = prune_low_quality_half(
        candidates=candidates,
        keep_fraction=_KEEP_FRACTION,
        min_keep=int(n_models),
        stage=stage,
        location=location,
    )
    kept_ids = [item.sample_id for item in kept]
    kept_scores = [(item.sample_id, float(item.adjusted_score)) for item in kept]
    vectors_kept = _build_vectors_for_kept(kept_ids=kept_ids, vectors=vectors, stage=stage, location=location)
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
            examples=[target_id],
        )

    target_parts = _materialize_selected_parts(sample_parts=sample_parts, selected_ids=selected_ids)
    target_stats: dict[str, object] = {
        "target_id": target_id,
        "n_candidates": int(len(candidates)),
        "n_after_prune": int(len(kept)),
        "cluster_count": int(cluster_count),
        "selected_ids": [str(item) for item in selected_ids],
    }
    return target_parts, target_stats


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
    _validate_select_params(n_models=n_models, diversity_lambda=diversity_lambda, stage=stage, location=location)

    ranked = read_table(ranked_path, stage=stage, location=location)
    _validate_ranked_table(ranked, stage=stage, location=location)

    out_parts: list[pl.DataFrame] = []
    selected_count: list[dict[str, object]] = []
    for raw_target_id, target_df in ranked.group_by("target_id", maintain_order=True):
        target_id = _normalize_group_key(raw_target_id)
        target_parts, target_stats = _select_target_models(
            target_id=target_id,
            target_df=target_df,
            n_models=int(n_models),
            diversity_lambda=float(diversity_lambda),
            stage=stage,
            location=location,
        )
        out_parts.extend(target_parts)
        selected_count.append(target_stats)

    if not out_parts:
        raise_error(stage, location, "nenhuma linha selecionada para top5 se3", impact="1", examples=[])
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
