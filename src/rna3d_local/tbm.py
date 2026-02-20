from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .errors import raise_error
from .se3.sequence_parser import parse_sequence_with_chains
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class TbmResult:
    predictions_path: Path
    manifest_path: Path


def _scan_table(path: Path, *, stage: str, location: str, label: str) -> pl.LazyFrame:
    if not path.exists():
        raise_error(stage, location, f"{label} ausente", impact="1", examples=[str(path)])
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            return pl.scan_parquet(str(path))
        if suffix == ".csv":
            return pl.scan_csv(str(path))
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, f"falha ao abrir {label} em modo lazy", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    raise_error(stage, location, f"formato nao suportado para {label}", impact="1", examples=[str(path)])


def _require_columns_lazy(lf: pl.LazyFrame, required: list[str], *, stage: str, location: str, label: str) -> None:
    schema = lf.collect_schema()
    missing = [c for c in required if c not in schema.names()]
    if missing:
        raise_error(stage, location, f"{label} sem coluna obrigatoria", impact=str(len(missing)), examples=missing[:8])


def _rank_column(schema_names: list[str], *, stage: str, location: str) -> tuple[str, list[str], list[bool]]:
    if "rerank_rank" in schema_names:
        return "rerank_rank", ["target_id", "rerank_rank"], [False, False]
    if "rank" in schema_names:
        return "rank", ["target_id", "rank"], [False, False]
    if "final_score" in schema_names:
        return "final_score", ["target_id", "final_score"], [False, True]
    raise_error(stage, location, "retrieval sem coluna de ranking", impact="1", examples=schema_names[:8])
    raise AssertionError("unreachable")


def _build_target_parse_frames(
    *,
    targets: pl.DataFrame,
    chain_separator: str,
    stage: str,
    location: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    target_rows: list[dict[str, object]] = []
    residue_rows: list[dict[str, object]] = []
    for row in targets.select("target_id", "sequence").iter_rows(named=True):
        target_id = str(row["target_id"])
        parsed = parse_sequence_with_chains(
            sequence=str(row["sequence"]),
            chain_separator=chain_separator,
            stage=stage,
            location=location,
            target_id=target_id,
        )
        target_rows.append({"target_id": target_id, "target_len": int(len(parsed.residues))})
        for resid, base in enumerate(parsed.residues, start=1):
            target_rows_idx = int(resid - 1)
            residue_rows.append(
                {
                    "target_id": target_id,
                    "resid": int(resid),
                    "resname": str(base),
                    "chain_index": int(parsed.chain_index[target_rows_idx]),
                    "residue_index_1d": int(parsed.residue_position_index_1d[target_rows_idx]),
                }
            )
    if not target_rows:
        return (
            pl.DataFrame(schema={"target_id": pl.Utf8, "target_len": pl.Int32}),
            pl.DataFrame(
                schema={
                    "target_id": pl.Utf8,
                    "resid": pl.Int32,
                    "resname": pl.Utf8,
                    "chain_index": pl.Int32,
                    "residue_index_1d": pl.Int32,
                }
            ),
        )
    return (
        pl.DataFrame(target_rows).with_columns(pl.col("target_id").cast(pl.Utf8), pl.col("target_len").cast(pl.Int32)),
        pl.DataFrame(residue_rows).with_columns(
            pl.col("target_id").cast(pl.Utf8),
            pl.col("resid").cast(pl.Int32),
            pl.col("resname").cast(pl.Utf8),
            pl.col("chain_index").cast(pl.Int32),
            pl.col("residue_index_1d").cast(pl.Int32),
        ),
    )


def predict_tbm(
    *,
    repo_root: Path,
    retrieval_path: Path,
    templates_path: Path,
    targets_path: Path,
    out_path: Path,
    n_models: int,
    chain_separator: str = "|",
    allow_missing_targets: bool = False,
    min_template_coverage: float = 1.0,
) -> TbmResult:
    stage = "PREDICT_TBM"
    location = "src/rna3d_local/tbm.py:predict_tbm"
    if int(n_models) <= 0:
        raise_error(stage, location, "n_models deve ser > 0", impact="1", examples=[str(n_models)])
    if len(str(chain_separator)) != 1:
        raise_error(stage, location, "chain_separator deve ter 1 caractere", impact="1", examples=[str(chain_separator)])
    if (not (0.0 < float(min_template_coverage) <= 1.0)):
        raise_error(stage, location, "min_template_coverage deve estar em (0,1]", impact="1", examples=[str(min_template_coverage)])

    retrieval_lf = _scan_table(retrieval_path, stage=stage, location=location, label="retrieval")
    _require_columns_lazy(retrieval_lf, ["target_id", "template_uid"], stage=stage, location=location, label="retrieval")
    rank_col, sort_by, sort_desc = _rank_column(retrieval_lf.collect_schema().names(), stage=stage, location=location)

    ranked = (
        retrieval_lf.select(
            pl.col("target_id").cast(pl.Utf8),
            pl.col("template_uid").cast(pl.Utf8),
            pl.col(rank_col),
        )
        .sort(sort_by, descending=sort_desc)
        .unique(subset=["target_id", "template_uid"], keep="first")
    )
    candidate_uids = ranked.select(pl.col("template_uid")).unique()

    templates_lf = _scan_table(templates_path, stage=stage, location=location, label="templates")
    _require_columns_lazy(templates_lf, ["template_uid", "resid", "resname", "x", "y", "z"], stage=stage, location=location, label="templates")
    templates_lf = templates_lf.select(
        pl.col("template_uid").cast(pl.Utf8),
        pl.col("resid").cast(pl.Int32),
        pl.col("resname").cast(pl.Utf8).alias("template_resname"),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
    )
    templates_lf = templates_lf.join(candidate_uids, on="template_uid", how="semi")

    # Validate duplicate coordinates (same template_uid+resid repeated).
    dup = templates_lf.group_by(["template_uid", "resid"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    dup_count = int(dup.select(pl.len()).collect().item())
    if dup_count > 0:
        examples = (
            dup.select((pl.col("template_uid") + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k"))
            .head(8)
            .collect()
            .get_column("k")
            .to_list()
        )
        raise_error(stage, location, "resid duplicado no template", impact=str(dup_count), examples=[str(x) for x in examples])

    template_min = templates_lf.group_by("template_uid").agg(pl.col("resid").min().alias("min_resid"))
    templates_with_norm = templates_lf.join(template_min, on="template_uid", how="left").with_columns(
        (pl.col("resid") - pl.col("min_resid") + 1).cast(pl.Int32).alias("resid_norm")
    )

    targets_lf = _scan_table(targets_path, stage=stage, location=location, label="targets")
    _require_columns_lazy(targets_lf, ["target_id", "sequence", "temporal_cutoff"], stage=stage, location=location, label="targets")
    targets_lf = targets_lf.select(
        pl.col("target_id").cast(pl.Utf8),
        pl.col("sequence").cast(pl.Utf8),
        pl.col("temporal_cutoff").cast(pl.Utf8),
    ).with_columns(pl.col("temporal_cutoff").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("temporal_cutoff"))
    bad_dates_lf = targets_lf.filter(pl.col("temporal_cutoff").is_null()).select("target_id")
    bad_dates_n = int(bad_dates_lf.select(pl.len()).collect().item())
    if bad_dates_n > 0:
        examples = bad_dates_lf.head(8).collect().get_column("target_id").to_list()
        raise_error(stage, location, "targets com data invalida", impact=str(int(bad_dates_n)), examples=[str(x) for x in examples])

    targets_collected = targets_lf.select("target_id", "sequence").collect()
    targets_len_df, resids_df = _build_target_parse_frames(
        targets=targets_collected,
        chain_separator=str(chain_separator),
        stage=stage,
        location=location,
    )
    targets_len = targets_len_df.lazy()

    ranked_with_len = ranked.join(targets_len.select("target_id", "target_len"), on="target_id", how="inner")
    prefix_rows = (
        ranked_with_len.join(templates_with_norm.select("template_uid", "resid_norm"), on="template_uid", how="inner")
        .filter(pl.col("resid_norm") <= pl.col("target_len"))
        .select("target_id", "template_uid", pl.col("resid_norm"), pl.col(rank_col), pl.col("target_len"))
    )
    prefix_counts = (
        prefix_rows.group_by(["target_id", "template_uid"]).agg(
            pl.col("resid_norm").n_unique().alias("n_prefix"),
            pl.first(rank_col).alias(rank_col),
            pl.first("target_len").alias("target_len"),
        )
    ).with_columns(
        (
            pl.col("n_prefix").cast(pl.Float64) / pl.col("target_len").cast(pl.Float64)
        ).alias("coverage_ratio")
    )

    # Filter templates by configurable minimal coverage ratio.
    valid_candidates = (
        prefix_counts.filter(pl.col("coverage_ratio") >= float(min_template_coverage))
        .sort(sort_by, descending=sort_desc)
        .with_columns(pl.cum_count("template_uid").over("target_id").alias("model_id"))
        .filter(pl.col("model_id") <= int(n_models))
        .select("target_id", "template_uid", "model_id")
    )

    target_ids = targets_len.select("target_id").unique()
    valid_counts = valid_candidates.group_by("target_id").agg(pl.len().alias("n_valid"))
    missing_targets = target_ids.join(valid_counts, on="target_id", how="left").filter(pl.col("n_valid").is_null() | (pl.col("n_valid") <= 0))
    missing_n = int(missing_targets.select(pl.len()).collect().item())
    missing_examples = missing_targets.select("target_id").head(8).collect().get_column("target_id").to_list() if missing_n > 0 else []
    if missing_n > 0:
        if not bool(allow_missing_targets):
            raise_error(
                stage,
                location,
                "alvos sem templates validos para TBM (cobertura insuficiente para export estrito)",
                impact=str(missing_n),
                examples=[str(x) for x in missing_examples],
            )
        print(
            f"[{stage}] [{location}] alvos sem templates validos para TBM; mantendo saida parcial para fallback do roteador | "
            f"impacto={missing_n} | exemplos={','.join(str(x) for x in missing_examples) if missing_examples else '-'}",
            file=sys.stderr,
        )

    padded = valid_counts.filter(pl.col("n_valid") < int(n_models)).with_columns(
        (pl.col("target_id") + pl.lit(":validos=") + pl.col("n_valid").cast(pl.Utf8) + pl.lit(" pad=") + (pl.lit(int(n_models)) - pl.col("n_valid")).cast(pl.Utf8)).alias(
            "msg"
        )
    )
    padded_count = int(padded.select(pl.len()).collect().item())
    padded_examples = padded.select("msg").head(8).collect().get_column("msg").to_list() if padded_count > 0 else []

    first_uid = valid_candidates.filter(pl.col("model_id") == 1).select(pl.col("target_id"), pl.col("template_uid").alias("first_template_uid"))

    model_ids_lf = pl.LazyFrame({"model_id": list(range(1, int(n_models) + 1, 1))})
    all_models = target_ids.join(model_ids_lf, how="cross")
    chosen = (
        all_models.join(valid_candidates, on=["target_id", "model_id"], how="left")
        .join(first_uid, on="target_id", how="left")
        .with_columns(pl.coalesce([pl.col("template_uid"), pl.col("first_template_uid")]).alias("template_uid"))
        .select("target_id", "model_id", "template_uid")
    )
    if bool(allow_missing_targets):
        chosen = chosen.filter(pl.col("template_uid").is_not_null())

    resids = resids_df.lazy()

    expanded = resids.join(chosen, on="target_id", how="inner")
    templates_norm_lf = (
        templates_with_norm.select(
            pl.col("template_uid"),
            pl.col("resid_norm").alias("resid"),
            pl.col("template_resname"),
            pl.col("x"),
            pl.col("y"),
            pl.col("z"),
        )
    )
    out_lf_raw = (
        expanded.join(templates_norm_lf, on=["template_uid", "resid"], how="left")
        .select(
            pl.col("target_id"),
            pl.col("model_id").cast(pl.Int32),
            pl.col("resid").cast(pl.Int32),
            pl.col("resname").cast(pl.Utf8),
            pl.col("chain_index").cast(pl.Int32),
            pl.col("residue_index_1d").cast(pl.Int32),
            pl.col("x"),
            pl.col("y"),
            pl.col("z"),
            pl.col("template_uid").cast(pl.Utf8),
            pl.col("template_resname").cast(pl.Utf8),
        )
        .sort(["target_id", "model_id", "resid"])
    )

    missing_coords = out_lf_raw.filter(pl.col("x").is_null() | pl.col("y").is_null() | pl.col("z").is_null())
    missing_coords_n = int(missing_coords.select(pl.len()).collect().item())
    if missing_coords_n > 0:
        examples = (
            missing_coords.select(
                (pl.col("target_id") + pl.lit(":") + pl.col("model_id").cast(pl.Utf8) + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k")
            )
            .head(8)
            .collect()
            .get_column("k")
            .to_list()
        )
        raise_error(
            stage,
            location,
            "TBM gerou coordenadas faltantes apos join (template_uid+resid ausente)",
            impact=str(missing_coords_n),
            examples=[str(x) for x in examples],
        )
    out_lf = out_lf_raw

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_lf.sink_parquet(str(out_path), engine="streaming")

    manifest_path = out_path.parent / "tbm_manifest.json"
    n_rows = int(pl.scan_parquet(str(out_path)).select(pl.len()).collect().item())
    n_targets_with_tbm = int(pl.scan_parquet(str(out_path)).select(pl.col("target_id").n_unique()).collect().item())
    manifest = {
        "created_utc": utc_now_iso(),
        "paths": {
            "retrieval": rel_or_abs(retrieval_path, repo_root),
            "templates": rel_or_abs(templates_path, repo_root),
            "targets": rel_or_abs(targets_path, repo_root),
            "predictions": rel_or_abs(out_path, repo_root),
        },
        "params": {"n_models": int(n_models), "min_template_coverage": float(min_template_coverage)},
        "stats": {
            "n_rows": int(n_rows),
            "n_targets_with_tbm": int(n_targets_with_tbm),
            "n_targets_without_template": int(missing_n),
            "examples_targets_without_template": [str(x) for x in missing_examples],
            "n_missing_coords": int(missing_coords_n),
            "n_targets_padded": int(padded_count),
            "examples_targets_padded": [str(x) for x in padded_examples],
        },
        "policy": {"allow_missing_targets": bool(allow_missing_targets)},
        "sha256": {"predictions.parquet": sha256_file(out_path)},
    }
    write_json(manifest_path, manifest)
    return TbmResult(predictions_path=out_path, manifest_path=manifest_path)
