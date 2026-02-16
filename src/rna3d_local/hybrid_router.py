from __future__ import annotations

import gc
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


def _default_confidence_for_source(source: str) -> float:
    mapping = {
        "tbm": 0.82,
        "rnapro": 0.72,
        "chai1": 0.78,
        "boltz1": 0.74,
        "chai1_boltz1_ensemble": 0.79,
        "generative_se3": 0.80,
        "se3": 0.80,
    }
    return float(mapping.get(source, 0.70))


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
        (pl.col("confidence").cast(pl.Float64) if "confidence" in df.columns else pl.lit(_default_confidence_for_source(source))).alias("confidence"),
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


def _build_chai_boltz_ensemble_for_target(*, target_id: str, chai: pl.DataFrame, boltz: pl.DataFrame, stage: str, location: str) -> pl.DataFrame:
    chai_t = chai.filter(pl.col("target_id") == target_id).select("target_id", "model_id", "resid", "resname", "x", "y", "z", "confidence")
    boltz_t = boltz.filter(pl.col("target_id") == target_id).select("target_id", "model_id", "resid", "resname", "x", "y", "z", "confidence")
    if chai_t.height == 0 or boltz_t.height == 0:
        raise_error(
            stage,
            location,
            "ensemble chai+boltz sem cobertura completa do alvo",
            impact="1",
            examples=[f"{target_id}:chai={chai_t.height}:boltz={boltz_t.height}"],
        )
    join = chai_t.join(
        boltz_t,
        on=["target_id", "model_id", "resid"],
        how="inner",
        suffix="_boltz",
    )
    if join.height != chai_t.height or join.height != boltz_t.height:
        raise_error(
            stage,
            location,
            "ensemble chai+boltz com mismatch de chaves",
            impact=str(abs(chai_t.height - join.height) + abs(boltz_t.height - join.height)),
            examples=[target_id],
        )
    out = join.select(
        pl.col("target_id"),
        pl.col("model_id"),
        pl.col("resid"),
        pl.col("resname"),
        ((pl.col("x") + pl.col("x_boltz")) / 2.0).alias("x"),
        ((pl.col("y") + pl.col("y_boltz")) / 2.0).alias("y"),
        ((pl.col("z") + pl.col("z_boltz")) / 2.0).alias("z"),
        ((pl.col("confidence") + pl.col("confidence_boltz")) / 2.0).alias("confidence"),
    ).with_columns(pl.lit("chai1_boltz1_ensemble").alias("source"))
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


def build_hybrid_candidates(
    *,
    repo_root: Path,
    targets_path: Path,
    retrieval_path: Path,
    tbm_path: Path,
    out_path: Path,
    routing_path: Path,
    template_score_threshold: float,
    ultra_long_seq_threshold: int = 1500,
    rnapro_path: Path | None,
    chai1_path: Path | None,
    boltz1_path: Path | None,
    se3_path: Path | None = None,
) -> HybridCandidatesResult:
    stage = "HYBRID_ROUTER"
    location = "src/rna3d_local/hybrid_router.py:build_hybrid_candidates"
    if template_score_threshold < 0.0:
        raise_error(stage, location, "template_score_threshold invalido (>=0)", impact="1", examples=[str(template_score_threshold)])
    if int(ultra_long_seq_threshold) <= 0:
        raise_error(stage, location, "ultra_long_seq_threshold deve ser > 0", impact="1", examples=[str(ultra_long_seq_threshold)])

    targets = load_targets_with_contract(targets_path=targets_path, stage=stage, location=location)
    retrieval = read_table(retrieval_path, stage=stage, location=location)
    target_scores = _template_score_by_target(retrieval, stage=stage, location=location)

    tbm = _normalize_predictions_table(read_table(tbm_path, stage=stage, location=location), source="tbm", stage=stage, location=location)
    rnapro = None if rnapro_path is None else _normalize_predictions_table(read_table(rnapro_path, stage=stage, location=location), source="rnapro", stage=stage, location=location)
    chai1 = None if chai1_path is None else _normalize_predictions_table(read_table(chai1_path, stage=stage, location=location), source="chai1", stage=stage, location=location)
    boltz1 = None if boltz1_path is None else _normalize_predictions_table(read_table(boltz1_path, stage=stage, location=location), source="boltz1", stage=stage, location=location)
    se3 = None if se3_path is None else _normalize_predictions_table(read_table(se3_path, stage=stage, location=location), source="generative_se3", stage=stage, location=location)
    tbm_targets = set(tbm.get_column("target_id").unique().to_list())
    se3_targets = set() if se3 is None else set(se3.get_column("target_id").unique().to_list())

    candidate_parts: list[pl.DataFrame] = []
    routing_rows: list[dict[str, object]] = []
    for row in targets.select("target_id", "sequence", "ligand_SMILES").iter_rows(named=True):
        tid = str(row["target_id"])
        sequence = str(row["sequence"])
        target_length = int(len(sequence))
        ultralong = bool(target_length > int(ultra_long_seq_threshold))
        has_ligand = len(str(row["ligand_SMILES"]).strip()) > 0
        score = float(target_scores.get(tid, 0.0))
        template_strong = score >= float(template_score_threshold)
        if ultralong:
            rule = "ultralong->generative_se3"
            primary_source = "generative_se3"
            if se3 is None:
                raise_error(stage, location, "alvo ultralongo exige se3_path", impact="1", examples=[f"{tid}:L={target_length}"])
            primary_df = se3.filter(pl.col("target_id") == tid)
            if primary_df.height == 0:
                raise_error(
                    stage,
                    location,
                    "alvo ultralongo sem cobertura em se3",
                    impact="1",
                    examples=[f"{tid}:L={target_length}", "forneca --se3 com cobertura completa"],
                )
            _aggressive_foundation_offload(model=None, stage=stage, location=location, context=f"{tid}:ultralong_fallback")
        elif template_strong and (tid in tbm_targets):
            rule = "template->tbm"
            primary_source = "tbm"
            primary_df = tbm.filter(pl.col("target_id") == tid)
        elif tid in se3_targets:
            if template_strong:
                rule = "template_missing->generative_se3"
            elif has_ligand:
                rule = "ligand_or_weak_template->generative_se3"
            else:
                rule = "orphan_or_weak_template->generative_se3"
            primary_source = "generative_se3"
            if se3 is None:
                raise_error(stage, location, "rota generative_se3 exige se3_path", impact="1", examples=[tid])
            primary_df = se3.filter(pl.col("target_id") == tid)
        elif template_strong and (tid not in tbm_targets):
            if has_ligand:
                rule = "template_missing->boltz1"
                primary_source = "boltz1"
                if boltz1 is None:
                    raise_error(stage, location, "rota template_missing com ligante exige boltz1_path", impact="1", examples=[tid])
                primary_df = boltz1.filter(pl.col("target_id") == tid)
            else:
                rule = "template_missing->chai1+boltz1"
                primary_source = "chai1_boltz1_ensemble"
                if chai1 is None or boltz1 is None:
                    raise_error(stage, location, "rota template_missing sem ligante exige chai1_path e boltz1_path", impact="1", examples=[tid])
                primary_df = _build_chai_boltz_ensemble_for_target(target_id=tid, chai=chai1, boltz=boltz1, stage=stage, location=location)
        elif has_ligand:
            rule = "ligand->boltz1"
            primary_source = "boltz1"
            if boltz1 is None:
                raise_error(stage, location, "rota ligante exige boltz1_path", impact="1", examples=[tid])
            primary_df = boltz1.filter(pl.col("target_id") == tid)
        else:
            rule = "orphan->chai1+boltz1"
            primary_source = "chai1_boltz1_ensemble"
            if chai1 is None or boltz1 is None:
                raise_error(stage, location, "rota orfao exige chai1_path e boltz1_path", impact="1", examples=[tid])
            primary_df = _build_chai_boltz_ensemble_for_target(target_id=tid, chai=chai1, boltz=boltz1, stage=stage, location=location)

        if primary_source in {"boltz1", "chai1", "chai1_boltz1_ensemble", "rnapro"}:
            _aggressive_foundation_offload(model=None, stage=stage, location=location, context=f"{tid}:{primary_source}")

        if primary_df.height == 0:
            raise_error(stage, location, "fonte primaria sem cobertura do alvo", impact="1", examples=[f"{tid}:{primary_source}"])
        primary_df = primary_df.with_columns(pl.lit(rule).alias("route_rule"))
        candidate_parts.append(primary_df)

        if (not ultralong) and (rnapro is not None):
            rn = rnapro.filter(pl.col("target_id") == tid)
            if rn.height > 0:
                rn = rn.with_columns(pl.lit(rule).alias("route_rule"))
                candidate_parts.append(rn)
                _aggressive_foundation_offload(model=None, stage=stage, location=location, context=f"{tid}:supplemental_rnapro")

        routing_rows.append(
            {
                "target_id": tid,
                "target_length": int(target_length),
                "ultralong_fallback": bool(ultralong),
                "template_score": score,
                "template_strong": bool(template_strong),
                "has_ligand": bool(has_ligand),
                "route_rule": rule,
                "primary_source": primary_source,
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
                "ultra_long_seq_threshold": int(ultra_long_seq_threshold),
            },
            "paths": {
                "targets": rel_or_abs(targets_path, repo_root),
                "retrieval": rel_or_abs(retrieval_path, repo_root),
                "tbm": rel_or_abs(tbm_path, repo_root),
                "rnapro": None if rnapro_path is None else rel_or_abs(rnapro_path, repo_root),
                "chai1": None if chai1_path is None else rel_or_abs(chai1_path, repo_root),
                "boltz1": None if boltz1_path is None else rel_or_abs(boltz1_path, repo_root),
                "se3": None if se3_path is None else rel_or_abs(se3_path, repo_root),
                "candidates": rel_or_abs(out_path, repo_root),
                "routing": rel_or_abs(routing_path, repo_root),
            },
            "stats": {
                "n_candidate_rows": int(candidates.height),
                "n_routing_rows": int(routing.height),
            },
            "sha256": {
                "candidates.parquet": sha256_file(out_path),
                "routing.parquet": sha256_file(routing_path),
            },
        },
    )
    return HybridCandidatesResult(candidates_path=out_path, routing_path=routing_path, manifest_path=manifest_path)
