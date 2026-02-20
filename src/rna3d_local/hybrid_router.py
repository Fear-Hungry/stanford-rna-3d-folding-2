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
        (pl.col("chain_index").cast(pl.Int32) if "chain_index" in df.columns else pl.lit(None).cast(pl.Int32)).alias("chain_index"),
        (pl.col("residue_index_1d").cast(pl.Int32) if "residue_index_1d" in df.columns else pl.lit(None).cast(pl.Int32)).alias("residue_index_1d"),
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
    return df.filter(pl.col("target_id") == target_id).select(
        "target_id",
        "model_id",
        "resid",
        "resname",
        "x",
        "y",
        "z",
        "confidence",
        "source",
        "chain_index",
        "residue_index_1d",
    )


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

    def _recovery_sources_for_target(*, tid: str, target_score: float) -> list[tuple[str, pl.DataFrame]]:
        out: list[tuple[str, pl.DataFrame]] = []
        tbm_t = _target_subset(tbm, target_id=tid).with_columns(
            pl.coalesce([pl.col("confidence"), pl.lit(float(target_score))]).alias("confidence")
        )
        out.append(("tbm", tbm_t))
        if se3_mamba is not None:
            out.append(("se3_mamba", _target_subset(se3_mamba, target_id=tid)))
        if se3_flash is not None:
            out.append(("se3_flash", _target_subset(se3_flash, target_id=tid)))
        if rnapro is not None:
            out.append(("rnapro", _target_subset(rnapro, target_id=tid)))
        if chai1 is not None:
            out.append(("chai1", _target_subset(chai1, target_id=tid)))
        if boltz1 is not None:
            out.append(("boltz1", _target_subset(boltz1, target_id=tid)))
        return out

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

        # Always prioritize TBM when retrieval confidence indicates strong template homology.
        tbm_t = _target_subset(tbm, target_id=tid).with_columns(
            pl.coalesce([pl.col("confidence"), pl.lit(float(score))]).alias("confidence")
        )
        if template_strong and tbm_t.height > 0:
            primary_df = tbm_t.sort(["source", "model_id", "resid"])
            primary_source = "tbm"
            rule = f"template_strong->tbm(len_bucket={length_bucket})"
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
            continue

        if length_bucket == "short":
            foundation_parts: list[pl.DataFrame] = []
            present_sources = 0
            for source_name, source_df in (("chai1", chai1), ("boltz1", boltz1), ("rnapro", rnapro)):
                if source_df is None:
                    continue
                present_sources += 1
                part = _target_subset(source_df, target_id=tid)
                if part.height > 0:
                    foundation_parts.append(part)
            if foundation_parts:
                primary_df = pl.concat(foundation_parts, how="vertical_relaxed").sort(["source", "model_id", "resid"])
            else:
                primary_df = pl.DataFrame(schema=tbm.schema)
            if present_sources == 3 and len(foundation_parts) == 3:
                primary_source = "foundation_trio"
                rule = f"len<={effective_short_max}->foundation_trio"
            elif foundation_parts:
                primary_source = "foundation_partial"
                fallback_used = True
                fallback_source = "foundation_partial"
                rule = f"len<={effective_short_max}->foundation_partial"
                print(
                    f"[{stage}] [{location}] bucket short degradado para cobertura parcial de foundation | "
                    f"impacto=1 | exemplos={tid}:L={target_length}:parts={len(foundation_parts)}",
                    file=sys.stderr,
                )
            else:
                primary_source = "foundation_missing"
                fallback_used = True
                fallback_source = "coverage_recovery_pending"
                rule = f"len<={effective_short_max}->foundation_missing"
                print(
                    f"[{stage}] [{location}] bucket short sem cobertura foundation para alvo; iniciando recovery | "
                    f"impacto=1 | exemplos={tid}:L={target_length}",
                    file=sys.stderr,
                )
            _aggressive_foundation_offload(model=None, stage=stage, location=location, context=f"{tid}:foundation_trio")
        elif length_bucket == "medium":
            if se3_flash is None:
                primary_df = pl.DataFrame(schema=tbm.schema)
                primary_source = "se3_flash_missing"
                fallback_used = True
                fallback_source = "coverage_recovery_pending"
                rule = f"{effective_short_max}<len<={effective_medium_max}->se3_flash_missing"
                print(
                    f"[{stage}] [{location}] bucket medium sem se3_flash; iniciando recovery | "
                    f"impacto=1 | exemplos={tid}:L={target_length}",
                    file=sys.stderr,
                )
            else:
                primary_df = _target_subset(se3_flash, target_id=tid)
                primary_source = "se3_flash"
                rule = f"{effective_short_max}<len<={effective_medium_max}->se3_flash"
        else:
            rule = f"len>{effective_medium_max}->tbm+se3_mamba"
            mamba_t = pl.DataFrame()
            if se3_mamba is not None:
                mamba_t = _target_subset(se3_mamba, target_id=tid)
            parts: list[pl.DataFrame] = []
            if mamba_t.height > 0:
                parts.append(mamba_t)
            if tbm_t.height > 0:
                parts.append(tbm_t)
            if not parts:
                primary_df = pl.DataFrame(schema=tbm.schema)
                primary_source = "tbm+se3_mamba_missing"
                fallback_used = True
                fallback_source = "coverage_recovery_pending"
                rule = f"{rule}_missing"
                print(
                    f"[{stage}] [{location}] bucket long sem cobertura em se3_mamba e tbm; iniciando recovery | "
                    f"impacto=1 | exemplos={tid}:L={target_length}:mamba={mamba_t.height}:tbm={tbm_t.height}",
                    file=sys.stderr,
                )
            elif len(parts) == 2:
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
            recovered_df: pl.DataFrame | None = None
            recovered_source: str | None = None
            for source_name, source_df in _recovery_sources_for_target(tid=tid, target_score=score):
                if source_df.height > 0:
                    recovered_df = source_df.sort(["source", "model_id", "resid"])
                    recovered_source = source_name
                    break
            if recovered_df is None or recovered_source is None:
                raise_error(
                    stage,
                    location,
                    "alvo sem cobertura em todas as fontes do roteador",
                    impact="1",
                    examples=[f"{tid}:len={target_length}", f"bucket={length_bucket}"],
                )
            fallback_used = True
            fallback_source = str(recovered_source)
            primary_df = recovered_df
            primary_source = str(recovered_source)
            rule = f"{rule}_coverage_recovery->{recovered_source}"
            print(
                f"[{stage}] [{location}] recovery de cobertura aplicado | "
                f"impacto=1 | exemplos={tid}:len={target_length}:{recovered_source}",
                file=sys.stderr,
            )
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
        raise_error(
            stage,
            location,
            "nenhum candidato gerado pelo roteador",
            impact="1",
            examples=["global"],
        )
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
