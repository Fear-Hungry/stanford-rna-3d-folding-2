from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from .contracts import require_columns
from .errors import raise_error
from .io_tables import read_table


def load_targets_with_contract(*, targets_path: Path, stage: str, location: str) -> pl.DataFrame:
    targets = read_table(targets_path, stage=stage, location=location)
    require_columns(targets, ["target_id", "sequence"], stage=stage, location=location, label="targets")
    if "ligand_SMILES" not in targets.columns:
        targets = targets.with_columns(pl.lit("").alias("ligand_SMILES"))
    targets = targets.select(
        pl.col("target_id").cast(pl.Utf8),
        pl.col("sequence").cast(pl.Utf8),
        pl.col("ligand_SMILES").cast(pl.Utf8),
    )
    bad = targets.filter((pl.col("target_id").str.len_chars() == 0) | (pl.col("sequence").str.len_chars() == 0))
    if bad.height > 0:
        examples = bad.select("target_id").head(8).get_column("target_id").to_list()
        raise_error(stage, location, "targets com target_id/sequence vazios", impact=str(bad.height), examples=[str(x) for x in examples])
    dup = targets.group_by("target_id").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if dup.height > 0:
        examples = dup.select("target_id").head(8).get_column("target_id").to_list()
        raise_error(stage, location, "target_id duplicado em targets", impact=str(dup.height), examples=[str(x) for x in examples])
    return targets


def ensure_model_artifacts(*, model_dir: Path, required_files: list[str], stage: str, location: str) -> None:
    if not model_dir.exists():
        raise_error(stage, location, "diretorio de modelo ausente", impact="1", examples=[str(model_dir)])
    missing: list[str] = []
    for rel in required_files:
        path = model_dir / rel
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise_error(stage, location, "arquivos obrigatorios de modelo ausentes", impact=str(len(missing)), examples=missing[:8])


def deterministic_offset(*, key: str, axis: str, scale: float) -> float:
    digest = hashlib.sha256(f"{key}|{axis}".encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], "big", signed=False) / float(2**64 - 1)
    return ((raw * 2.0) - 1.0) * scale


def build_synthetic_long_predictions(
    *,
    targets: pl.DataFrame,
    n_models: int,
    source: str,
    base_scale: float,
    confidence_base: float,
    ligand_bonus: float,
) -> pl.DataFrame:
    if n_models <= 0:
        raise ValueError("n_models must be > 0")
    rows: list[dict[str, object]] = []
    for target_id, sequence, ligand_smiles in targets.select("target_id", "sequence", "ligand_SMILES").iter_rows():
        tid = str(target_id)
        seq = str(sequence)
        has_ligand = len(str(ligand_smiles).strip()) > 0
        confidence = float(confidence_base + (ligand_bonus if has_ligand else 0.0))
        for model_id in range(1, n_models + 1):
            model_scale = base_scale * float(model_id)
            for resid, base in enumerate(seq, start=1):
                key = f"{source}|{tid}|{model_id}|{resid}|{base}"
                rows.append(
                    {
                        "target_id": tid,
                        "model_id": model_id,
                        "resid": resid,
                        "resname": str(base),
                        "x": deterministic_offset(key=key, axis="x", scale=model_scale),
                        "y": deterministic_offset(key=key, axis="y", scale=model_scale),
                        "z": deterministic_offset(key=key, axis="z", scale=model_scale),
                        "source": source,
                        "confidence": confidence,
                    }
                )
    return pl.DataFrame(rows).sort(["target_id", "model_id", "resid"])
