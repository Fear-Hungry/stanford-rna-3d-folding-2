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


_RNA_RG_COEFF = 5.5
_RNA_RG_EXPONENT = 0.33
_RG_REL_TOL = 0.35
_RG_EPS = 1e-6


def _expected_rg_for_rna(length: int) -> float:
    if int(length) <= 0:
        raise_error(
            "RANK_SE3",
            "src/rna3d_local/ensemble/qa_ranker_se3.py:_expected_rg_for_rna",
            "comprimento invalido para estimar Rg",
            impact="1",
            examples=[str(length)],
        )
    return float(_RNA_RG_COEFF * (float(length) ** float(_RNA_RG_EXPONENT)))


def _load_qa_config(path: Path | None, *, stage: str, location: str) -> dict[str, float]:
    if path is None:
        return {"compactness": 0.6, "smoothness": 0.4, "chem_exposure_consistency": 0.0}
    if not path.exists():
        raise_error(stage, location, "qa_config ausente", impact="1", examples=[str(path)])
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ["compactness", "smoothness"]:
        if key not in payload:
            raise_error(stage, location, "qa_config sem campo obrigatorio", impact="1", examples=[key])
    out = {
        "compactness": float(payload["compactness"]),
        "smoothness": float(payload["smoothness"]),
        "chem_exposure_consistency": float(payload.get("chem_exposure_consistency", 0.0)),
    }
    bad = [key for key, value in out.items() if float(value) < 0.0]
    if bad:
        raise_error(stage, location, "qa_config com peso negativo", impact=str(len(bad)), examples=bad[:8])
    if float(sum(float(value) for value in out.values())) <= 0.0:
        raise_error(stage, location, "qa_config com soma de pesos <= 0", impact="1", examples=[str(out)])
    return out


def _sample_quality(
    sample_df: pl.DataFrame,
    *,
    weights: dict[str, float],
    expected_exposure: np.ndarray | None,
) -> tuple[float, float, float, float]:
    coords = sample_df.select("x", "y", "z").to_numpy().astype(np.float64)
    n_res = int(coords.shape[0])
    if n_res <= 0:
        raise_error(
            "RANK_SE3",
            "src/rna3d_local/ensemble/qa_ranker_se3.py:_sample_quality",
            "sample sem coordenadas para QA",
            impact="1",
            examples=[],
        )
    center = coords.mean(axis=0, keepdims=True)
    squared_radius = ((coords - center) ** 2).sum(axis=1)
    radius = np.sqrt(squared_radius)
    rg_pred = float(np.sqrt(float(np.mean(squared_radius))))
    rg_expected = _expected_rg_for_rna(n_res)
    rg_rel_error = float(abs(rg_pred - rg_expected) / max(rg_expected, float(_RG_EPS)))
    compactness = float(1.0 / (1.0 + ((rg_rel_error / float(_RG_REL_TOL)) ** 2)))
    compactness = float(np.clip(compactness, 0.0, 1.0))
    if coords.shape[0] > 1:
        steps = np.sqrt(((coords[1:] - coords[:-1]) ** 2).sum(axis=1))
        smoothness = float(1.0 / (1.0 + float(np.std(steps))))
    else:
        smoothness = 1.0
    chem_weight = float(weights.get("chem_exposure_consistency", 0.0))
    if chem_weight > 0.0:
        if expected_exposure is None:
            raise_error(
                "RANK_SE3",
                "src/rna3d_local/ensemble/qa_ranker_se3.py:_sample_quality",
                "peso quimico > 0 sem expected_exposure",
                impact="1",
                examples=[],
            )
        if int(expected_exposure.shape[0]) != int(coords.shape[0]):
            raise_error(
                "RANK_SE3",
                "src/rna3d_local/ensemble/qa_ranker_se3.py:_sample_quality",
                "expected_exposure com comprimento diferente das coordenadas",
                impact="1",
                examples=[f"expected={int(expected_exposure.shape[0])}", f"coords={int(coords.shape[0])}"],
            )
        r_min = float(radius.min()) if radius.size > 0 else 0.0
        r_max = float(radius.max()) if radius.size > 0 else 0.0
        if r_max <= r_min:
            predicted_exposure = np.full_like(radius, 0.5, dtype=np.float64)
        else:
            predicted_exposure = (radius - r_min) / (r_max - r_min)
        mismatch = np.abs(predicted_exposure - expected_exposure.astype(np.float64))
        chem_consistency = float(np.clip(1.0 - float(np.mean(mismatch)), 0.0, 1.0))
    else:
        chem_consistency = 0.0
    qa_score = (
        (weights["compactness"] * compactness)
        + (weights["smoothness"] * smoothness)
        + (chem_weight * chem_consistency)
    )
    return float(qa_score), compactness, smoothness, chem_consistency


def _load_chemical_expectations(
    *,
    chemical_features_path: Path,
    stage: str,
    location: str,
) -> dict[str, pl.DataFrame]:
    chemical = read_table(chemical_features_path, stage=stage, location=location)
    require_columns(
        chemical,
        ["target_id", "resid", "p_open", "p_paired"],
        stage=stage,
        location=location,
        label="chemical_features",
    )
    chem_cast = chemical.select(
        pl.col("target_id").cast(pl.Utf8),
        pl.col("resid").cast(pl.Int32),
        pl.col("p_open").cast(pl.Float64),
        pl.col("p_paired").cast(pl.Float64),
    )
    bad_range = chem_cast.filter(
        pl.col("p_open").is_null()
        | pl.col("p_paired").is_null()
        | (pl.col("p_open") < 0.0)
        | (pl.col("p_open") > 1.0)
        | (pl.col("p_paired") < 0.0)
        | (pl.col("p_paired") > 1.0)
    )
    if bad_range.height > 0:
        examples = (
            bad_range.with_columns((pl.col("target_id") + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k"))
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(
            stage,
            location,
            "chemical_features com p_open/p_paired fora de [0,1] ou nulos",
            impact=str(int(bad_range.height)),
            examples=[str(item) for item in examples],
        )
    dup = chem_cast.group_by(["target_id", "resid"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if dup.height > 0:
        examples = (
            dup.with_columns((pl.col("target_id") + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k"))
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(stage, location, "chemical_features com chave duplicada", impact=str(int(dup.height)), examples=[str(item) for item in examples])
    chem_expect = chem_cast.with_columns(
        (((pl.col("p_open") + (1.0 - pl.col("p_paired"))) / 2.0).clip(0.0, 1.0)).alias("expected_exposure")
    ).select("target_id", "resid", "expected_exposure")
    out: dict[str, pl.DataFrame] = {}
    for target_id, part in chem_expect.group_by("target_id", maintain_order=True):
        tid = str(target_id[0]) if isinstance(target_id, tuple) else str(target_id)
        out[tid] = part.sort("resid")
    return out


def rank_se3_ensemble(
    *,
    repo_root: Path,
    candidates_path: Path,
    out_path: Path,
    qa_config_path: Path | None,
    diversity_lambda: float,
    chemical_features_path: Path | None = None,
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
    use_chem_score = float(qa_weights.get("chem_exposure_consistency", 0.0)) > 0.0
    chem_expect_by_target: dict[str, pl.DataFrame] = {}
    if chemical_features_path is not None:
        chem_expect_by_target = _load_chemical_expectations(
            chemical_features_path=chemical_features_path,
            stage=stage,
            location=location,
        )
    if use_chem_score and chemical_features_path is None:
        raise_error(
            stage,
            location,
            "qa_config exige sinal quimico, mas --chemical-features nao foi fornecido",
            impact="1",
            examples=["chem_exposure_consistency>0"],
        )

    rows: list[dict[str, object]] = []
    for target_id, target_df in candidates.group_by("target_id", maintain_order=True):
        tid = str(target_id[0]) if isinstance(target_id, tuple) else str(target_id)
        expected_by_resid: np.ndarray | None = None
        if chemical_features_path is not None:
            chem_target = chem_expect_by_target.get(tid)
            if chem_target is None:
                raise_error(stage, location, "chemical_features sem target para ranking", impact="1", examples=[tid])
            expected_by_resid = chem_target.get_column("expected_exposure").to_numpy().astype(np.float64)
        vectors = build_sample_vectors(target_df, stage=stage, location=location)
        sample_scores: list[tuple[str, float, float, float, float, float, float]] = []
        for sample_id, sample_df in target_df.group_by("sample_id", maintain_order=True):
            sid = str(sample_id[0]) if isinstance(sample_id, tuple) else str(sample_id)
            sample_sorted = sample_df.sort("resid")
            if expected_by_resid is not None:
                sample_resid = sample_sorted.get_column("resid").cast(pl.Int32).to_numpy()
                chem_target = chem_expect_by_target[tid]
                chem_resid = chem_target.get_column("resid").cast(pl.Int32).to_numpy()
                if int(sample_resid.shape[0]) != int(chem_resid.shape[0]) or not bool(np.array_equal(sample_resid, chem_resid)):
                    raise_error(
                        stage,
                        location,
                        "mismatch de residuo entre candidates e chemical_features no ranking",
                        impact="1",
                        examples=[f"{tid}:{sid}", f"sample_n={int(sample_resid.shape[0])}", f"chem_n={int(chem_resid.shape[0])}"],
                    )
            qa_score, qa_compactness, qa_smoothness, qa_chem = _sample_quality(
                sample_sorted,
                weights=qa_weights,
                expected_exposure=expected_by_resid,
            )
            diversity_penalty = average_similarity(sid, vectors)
            final_score = float(qa_score) - (float(diversity_lambda) * float(diversity_penalty))
            sample_scores.append((sid, qa_score, diversity_penalty, final_score, qa_compactness, qa_smoothness, qa_chem))
        if not sample_scores:
            raise_error(stage, location, "target sem samples para ranking", impact="1", examples=[tid])

        score_df = pl.DataFrame(
            {
                "target_id": [tid for _ in sample_scores],
                "sample_id": [item[0] for item in sample_scores],
                "qa_score": [float(item[1]) for item in sample_scores],
                "diversity_penalty": [float(item[2]) for item in sample_scores],
                "final_score": [float(item[3]) for item in sample_scores],
                "qa_compactness": [float(item[4]) for item in sample_scores],
                "qa_smoothness": [float(item[5]) for item in sample_scores],
                "qa_chem_exposure_consistency": [float(item[6]) for item in sample_scores],
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
                "chemical_features": None if chemical_features_path is None else rel_or_abs(chemical_features_path, repo_root),
                "ranked": rel_or_abs(out_path, repo_root),
            },
            "stats": {"n_rows": int(ranked.height), "n_targets": int(ranked.get_column("target_id").n_unique())},
            "sha256": {"ranked.parquet": sha256_file(out_path)},
        },
    )
    return RankSe3Result(ranked_path=out_path, manifest_path=manifest_path)
