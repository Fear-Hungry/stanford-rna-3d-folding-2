from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .contracts import require_columns
from .errors import raise_error
from .io_tables import read_table, write_table
from .predictor_common import load_targets_with_contract
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class HybridCandidatesResult:
    candidates_path: Path
    routing_path: Path
    manifest_path: Path


def _aggressive_foundation_offload(*, model: object | None, stage: str, location: str, context: str) -> None:
    if model is not None:
        to_fn = getattr(model, "to", None)
        if callable(to_fn):
            try:
                to_fn("cpu")
            except Exception as exc:  # noqa: BLE001
                raise_error(stage, location, "falha ao mover modelo para cpu no offload", impact="1", examples=[f"{context}:{type(exc).__name__}:{exc}"])
        del model
    gc.collect()
    try:
        import torch  # type: ignore
    except Exception:
        return
    if bool(torch.cuda.is_available()):
        try:
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            raise_error(stage, location, "falha ao liberar cache CUDA no offload", impact="1", examples=[f"{context}:{type(exc).__name__}:{exc}"])


def _normalize_predictions_table(df: pl.DataFrame, *, source: str, stage: str, location: str) -> pl.DataFrame:
    require_columns(df, ["target_id", "model_id", "resid", "resname", "x", "y", "z"], stage=stage, location=location, label=source)
    out = df.select(
        pl.col("target_id").cast(pl.Utf8),
        pl.col("model_id").cast(pl.Int32),
        pl.col("resid").cast(pl.Int32),
        pl.col("resname").cast(pl.Utf8),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
        (pl.col("confidence").cast(pl.Float64) if "confidence" in df.columns else pl.lit(None).cast(pl.Float64)).alias("confidence"),
        (pl.col("source").cast(pl.Utf8) if "source" in df.columns else pl.lit(source)).alias("source"),
    )
    dup = out.group_by(["target_id", "model_id", "resid"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if dup.height > 0:
        examples = (
            dup.with_columns(
                (pl.col("target_id") + pl.lit(":") + pl.col("model_id").cast(pl.Utf8) + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k")
            )
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(stage, location, f"{source} com chave duplicada", impact=str(dup.height), examples=[str(x) for x in examples])
    return out


def _template_score_by_target(retrieval: pl.DataFrame, *, stage: str, location: str) -> dict[str, float]:
    require_columns(retrieval, ["target_id"], stage=stage, location=location, label="retrieval")
    score_col = None
    for candidate in ("final_score", "similarity", "cosine_score"):
        if candidate in retrieval.columns:
            score_col = candidate
            break
    if score_col is None:
        raise_error(stage, location, "retrieval sem coluna de score", impact="1", examples=retrieval.columns[:8])
    agg = (
        retrieval.select(pl.col("target_id").cast(pl.Utf8), pl.col(score_col).cast(pl.Float64).alias("score"))
        .group_by("target_id")
        .agg(pl.col("score").max().alias("max_score"))
    )
    return {str(row["target_id"]): float(row["max_score"]) for row in agg.iter_rows(named=True)}


def _target_subset(df: pl.DataFrame, *, target_id: str) -> pl.DataFrame:
    return df.filter(pl.col("target_id") == target_id).select("target_id", "model_id", "resid", "resname", "x", "y", "z", "confidence", "source")


def _length_bucket(*, length: int, short_max_len: int, medium_max_len: int) -> str:
    if int(length) <= int(short_max_len):
        return "short"
    if int(length) <= int(medium_max_len):
        return "medium"
    return "long"


def build_hybrid_candidates(
    *,
    repo_root: Path,
    targets_path: Path,
    retrieval_path: Path,
    tbm_path: Path,
    out_path: Path,
    routing_path: Path,
    template_score_threshold: float,
    short_max_len: int = 350,
    medium_max_len: int = 600,
    ultra_long_seq_threshold: int | None = None,
    rnapro_path: Path | None,
    chai1_path: Path | None,
    boltz1_path: Path | None,
    se3_path: Path | None = None,
    se3_flash_path: Path | None = None,
    se3_mamba_path: Path | None = None,
) -> HybridCandidatesResult:
    stage = "HYBRID_ROUTER"
    location = "src/rna3d_local/hybrid_router.py:build_hybrid_candidates"
    if template_score_threshold < 0.0:
        raise_error(stage, location, "template_score_threshold invalido (>=0)", impact="1", examples=[str(template_score_threshold)])
    effective_short_max = int(short_max_len)
    if effective_short_max <= 0:
        raise_error(stage, location, "short_max_len deve ser > 0", impact="1", examples=[str(short_max_len)])
    effective_medium_max = int(medium_max_len)
    if ultra_long_seq_threshold is not None:
        if int(ultra_long_seq_threshold) <= 0:
            raise_error(stage, location, "ultra_long_seq_threshold deve ser > 0", impact="1", examples=[str(ultra_long_seq_threshold)])
        effective_medium_max = int(ultra_long_seq_threshold)
    if effective_medium_max <= effective_short_max:
        raise_error(
            stage,
            location,
            "medium_max_len deve ser maior que short_max_len",
            impact="1",
            examples=[f"short={effective_short_max}", f"medium={effective_medium_max}"],
        )

    targets = load_targets_with_contract(targets_path=targets_path, stage=stage, location=location)
    retrieval = read_table(retrieval_path, stage=stage, location=location)
    target_scores = _template_score_by_target(retrieval, stage=stage, location=location)

    tbm = _normalize_predictions_table(read_table(tbm_path, stage=stage, location=location), source="tbm", stage=stage, location=location)
    rnapro = None if rnapro_path is None else _normalize_predictions_table(read_table(rnapro_path, stage=stage, location=location), source="rnapro", stage=stage, location=location)
    chai1 = None if chai1_path is None else _normalize_predictions_table(read_table(chai1_path, stage=stage, location=location), source="chai1", stage=stage, location=location)
    boltz1 = None if boltz1_path is None else _normalize_predictions_table(read_table(boltz1_path, stage=stage, location=location), source="boltz1", stage=stage, location=location)
    effective_se3_flash_path = se3_flash_path if se3_flash_path is not None else se3_path
    effective_se3_mamba_path = se3_mamba_path if se3_mamba_path is not None else se3_path
    se3_flash = (
        None
        if effective_se3_flash_path is None
        else _normalize_predictions_table(
            read_table(effective_se3_flash_path, stage=stage, location=location),
            source="generative_se3_flash",
            stage=stage,
            location=location,
        )
    )
    se3_mamba = (
        None
        if effective_se3_mamba_path is None
        else _normalize_predictions_table(
            read_table(effective_se3_mamba_path, stage=stage, location=location),
            source="generative_se3_mamba",
            stage=stage,
            location=location,
        )
    )
    candidate_parts: list[pl.DataFrame] = []
    routing_rows: list[dict[str, object]] = []
    for row in targets.select("target_id", "sequence", "ligand_SMILES").iter_rows(named=True):
        tid = str(row["target_id"])
        sequence = str(row["sequence"])
        target_length = int(len(sequence))
        has_ligand = len(str(row["ligand_SMILES"]).strip()) > 0
        score = float(target_scores.get(tid, 0.0))
        template_strong = score >= float(template_score_threshold)
        length_bucket = _length_bucket(length=target_length, short_max_len=effective_short_max, medium_max_len=effective_medium_max)
        fallback_used = False
        fallback_source: str | None = None

        if length_bucket == "short":
            missing_sources: list[str] = []
            if chai1 is None:
                missing_sources.append("chai1")
            if boltz1 is None:
                missing_sources.append("boltz1")
            if rnapro is None:
                missing_sources.append("rnapro")
            if missing_sources:
                raise_error(
                    stage,
                    location,
                    "bucket short exige foundation trio completo",
                    impact=str(int(len(missing_sources))),
                    examples=[f"{tid}:L={target_length}", *missing_sources],
                )
            chai_t = _target_subset(chai1, target_id=tid)  # type: ignore[arg-type]
            boltz_t = _target_subset(boltz1, target_id=tid)  # type: ignore[arg-type]
            rnapro_t = _target_subset(rnapro, target_id=tid)  # type: ignore[arg-type]
            if chai_t.height == 0 or boltz_t.height == 0 or rnapro_t.height == 0:
                raise_error(
                    stage,
                    location,
                    "cobertura incompleta no bucket short (foundation trio)",
                    impact="1",
                    examples=[f"{tid}:chai={chai_t.height}:boltz={boltz_t.height}:rnapro={rnapro_t.height}"],
                )
            primary_df = pl.concat([chai_t, boltz_t, rnapro_t], how="vertical_relaxed").sort(["source", "model_id", "resid"])
            primary_source = "foundation_trio"
            rule = f"len<={effective_short_max}->foundation_trio"
            _aggressive_foundation_offload(model=None, stage=stage, location=location, context=f"{tid}:foundation_trio")
        elif length_bucket == "medium":
            if se3_flash is None:
                raise_error(
                    stage,
                    location,
                    "bucket medium exige se3_flash",
                    impact="1",
                    examples=[f"{tid}:L={target_length}", "forneca --se3-flash ou --se3 legado"],
                )
            primary_df = _target_subset(se3_flash, target_id=tid)
            if primary_df.height == 0:
                raise_error(
                    stage,
                    location,
                    "bucket medium sem cobertura em se3_flash",
                    impact="1",
                    examples=[f"{tid}:L={target_length}"],
                )
            primary_source = "se3_flash"
            rule = f"{effective_short_max}<len<={effective_medium_max}->se3_flash"
        else:
            rule = f"len>{effective_medium_max}->tbm+se3_mamba"
            mamba_t = pl.DataFrame()
            if se3_mamba is not None:
                mamba_t = _target_subset(se3_mamba, target_id=tid)
            tbm_t = _target_subset(tbm, target_id=tid).with_columns(
                pl.coalesce([pl.col("confidence"), pl.lit(float(score))]).alias("confidence")
            )
            parts: list[pl.DataFrame] = []
            if mamba_t.height > 0:
                parts.append(mamba_t)
            if tbm_t.height > 0:
                parts.append(tbm_t)
            if not parts:
                raise_error(
                    stage,
                    location,
                    "bucket long sem cobertura em se3_mamba e tbm",
                    impact="1",
                    examples=[f"{tid}:L={target_length}", f"mamba={mamba_t.height}", f"tbm={tbm_t.height}"],
                )
            if len(parts) == 2:
                primary_df = pl.concat(parts, how="vertical_relaxed").sort(["source", "model_id", "resid"])
                primary_source = "tbm+se3_mamba"
            else:
                fallback_used = True
                fallback_source = "se3_mamba" if mamba_t.height > 0 else "tbm"
                primary_source = str(fallback_source)
                primary_df = parts[0].sort(["source", "model_id", "resid"])
                rule = f"{rule}_fallback->{fallback_source}"
                print(
                    f"[{stage}] [{location}] bucket long degradado para fonte unica | impacto=1 | exemplos={tid}:L={target_length}:{fallback_source}",
                    file=sys.stderr,
                )

        if primary_df.height == 0:
            raise_error(stage, location, "fonte primaria sem cobertura do alvo", impact="1", examples=[f"{tid}:{primary_source}"])
        primary_df = primary_df.with_columns(pl.lit(rule).alias("route_rule"))
        candidate_parts.append(primary_df)

        routing_rows.append(
            {
                "target_id": tid,
                "target_length": int(target_length),
                "length_bucket": length_bucket,
                "short_max_len": int(effective_short_max),
                "medium_max_len": int(effective_medium_max),
                "template_score": score,
                "template_strong": bool(template_strong),
                "has_ligand": bool(has_ligand),
                "route_rule": rule,
                "primary_source": primary_source,
                "fallback_used": bool(fallback_used),
                "fallback_source": fallback_source,
            }
        )

    if not candidate_parts:
        raise_error(stage, location, "nenhum candidato gerado pelo roteador", impact="0", examples=[])

    candidates = pl.concat(candidate_parts, how="vertical_relaxed").sort(["target_id", "source", "model_id", "resid"])
    write_table(candidates, out_path)
    routing = pl.DataFrame(routing_rows).sort("target_id")
    write_table(routing, routing_path)
    manifest_path = out_path.parent / "hybrid_router_manifest.json"
    write_json(
        manifest_path,
        {
            "created_utc": utc_now_iso(),
            "params": {
                "template_score_threshold": float(template_score_threshold),
                "short_max_len": int(effective_short_max),
                "medium_max_len": int(effective_medium_max),
                "legacy_ultra_long_seq_threshold": None if ultra_long_seq_threshold is None else int(ultra_long_seq_threshold),
                "legacy_se3_alias_used": bool(se3_path is not None),
            },
            "paths": {
                "targets": rel_or_abs(targets_path, repo_root),
                "retrieval": rel_or_abs(retrieval_path, repo_root),
                "tbm": rel_or_abs(tbm_path, repo_root),
                "rnapro": None if rnapro_path is None else rel_or_abs(rnapro_path, repo_root),
                "chai1": None if chai1_path is None else rel_or_abs(chai1_path, repo_root),
                "boltz1": None if boltz1_path is None else rel_or_abs(boltz1_path, repo_root),
                "se3_legacy": None if se3_path is None else rel_or_abs(se3_path, repo_root),
                "se3_flash": None if effective_se3_flash_path is None else rel_or_abs(effective_se3_flash_path, repo_root),
                "se3_mamba": None if effective_se3_mamba_path is None else rel_or_abs(effective_se3_mamba_path, repo_root),
                "candidates": rel_or_abs(out_path, repo_root),
                "routing": rel_or_abs(routing_path, repo_root),
            },
            "stats": {
                "n_candidate_rows": int(candidates.height),
                "n_routing_rows": int(routing.height),
                "n_short_targets": int(routing.filter(pl.col("length_bucket") == "short").height),
                "n_medium_targets": int(routing.filter(pl.col("length_bucket") == "medium").height),
                "n_long_targets": int(routing.filter(pl.col("length_bucket") == "long").height),
                "n_long_dual_stack": int(routing.filter(pl.col("primary_source") == "tbm+se3_mamba").height),
                "n_long_fallback_tbm": int(routing.filter(pl.col("fallback_source") == "tbm").height),
                "n_long_fallback_mamba": int(routing.filter(pl.col("fallback_source") == "se3_mamba").height),
            },
            "sha256": {
                "candidates.parquet": sha256_file(out_path),
                "routing.parquet": sha256_file(routing_path),
            },
        },
    )
    return HybridCandidatesResult(candidates_path=out_path, routing_path=routing_path, manifest_path=manifest_path)
