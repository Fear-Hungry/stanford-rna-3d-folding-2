from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .contracts import require_columns
from .errors import raise_error
from .io_tables import read_table, write_table
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class PairingsResult:
    pairings_path: Path
    manifest_path: Path


def derive_pairings_from_chemical(
    *,
    repo_root: Path,
    chemical_features_path: Path,
    out_path: Path,
) -> PairingsResult:
    stage = "PAIRINGS"
    location = "src/rna3d_local/pairings.py:derive_pairings_from_chemical"
    chem = read_table(chemical_features_path, stage=stage, location=location)
    require_columns(chem, ["target_id", "resid", "p_paired"], stage=stage, location=location, label="chemical_features")

    view = chem.select(
        pl.col("target_id").cast(pl.Utf8),
        pl.col("resid").cast(pl.Int32),
        pl.col("p_paired").cast(pl.Float64, strict=False).alias("pair_prob"),
    ).sort(["target_id", "resid"])
    bad = view.filter(
        pl.col("target_id").is_null()
        | (pl.col("target_id").str.strip_chars() == "")
        | pl.col("resid").is_null()
        | pl.col("pair_prob").is_null()
    )
    if bad.height > 0:
        examples = bad.select("target_id", "resid").head(8).rows()
        raise_error(stage, location, "valores nulos/invalidos ao derivar pairings", impact=str(int(bad.height)), examples=[str(item) for item in examples])

    dup = view.group_by(["target_id", "resid"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if dup.height > 0:
        examples = (
            dup.with_columns((pl.col("target_id") + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k"))
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(stage, location, "chaves duplicadas ao derivar pairings", impact=str(int(dup.height)), examples=[str(item) for item in examples])

    out_of_range = view.filter((pl.col("pair_prob") < 0.0) | (pl.col("pair_prob") > 1.0))
    if out_of_range.height > 0:
        examples = out_of_range.select("target_id", "resid", "pair_prob").head(8).rows()
        raise_error(stage, location, "pair_prob fora de [0,1] ao derivar pairings", impact=str(int(out_of_range.height)), examples=[str(item) for item in examples])

    write_table(view, out_path)
    manifest_path = out_path.parent / "pairings_manifest.json"
    write_json(
        manifest_path,
        {
            "created_utc": utc_now_iso(),
            "paths": {
                "chemical_features": rel_or_abs(chemical_features_path, repo_root),
                "pairings": rel_or_abs(out_path, repo_root),
            },
            "stats": {
                "n_rows": int(view.height),
                "n_targets": int(view.get_column("target_id").n_unique()),
            },
            "sha256": {"pairings.parquet": sha256_file(out_path)},
        },
    )
    return PairingsResult(pairings_path=out_path, manifest_path=manifest_path)

