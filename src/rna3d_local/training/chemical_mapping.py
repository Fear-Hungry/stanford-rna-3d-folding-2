from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import torch

from ..contracts import require_columns
from ..errors import raise_error
from ..se3.sequence_parser import parse_sequence_with_chains


@dataclass(frozen=True)
class ChemicalExposureTarget:
    target_id: str
    sequence: str
    exposure: torch.Tensor
    source: str


def _assert_unique_keys(df: pl.DataFrame, *, label: str, stage: str, location: str) -> None:
    dup = df.group_by(["target_id", "resid"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if dup.height > 0:
        examples = (
            dup.with_columns((pl.col("target_id") + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k"))
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(stage, location, f"{label} com chaves duplicadas", impact=str(int(dup.height)), examples=[str(item) for item in examples])


def _minmax_norm(values: torch.Tensor) -> torch.Tensor:
    if int(values.numel()) == 0:
        return values
    vmin = float(values.min().item())
    vmax = float(values.max().item())
    if vmax <= vmin:
        return torch.full_like(values, 0.5)
    return (values - vmin) / (vmax - vmin)


def compute_chemical_exposure_mapping(
    *,
    targets: pl.DataFrame,
    chemical_features: pl.DataFrame,
    pdb_labels: pl.DataFrame | None,
    chain_separator: str,
    stage: str,
    location: str,
) -> dict[str, ChemicalExposureTarget]:
    require_columns(targets, ["target_id", "sequence"], stage=stage, location=location, label="targets")
    require_columns(
        chemical_features,
        ["target_id", "resid", "reactivity_dms", "reactivity_2a3"],
        stage=stage,
        location=location,
        label="chemical_features",
    )
    _assert_unique_keys(chemical_features, label="chemical_features", stage=stage, location=location)
    labels = None
    if pdb_labels is not None:
        require_columns(pdb_labels, ["target_id", "resid", "x", "y", "z"], stage=stage, location=location, label="pdb_labels")
        _assert_unique_keys(pdb_labels, label="pdb_labels", stage=stage, location=location)
        labels = pdb_labels.select(
            pl.col("target_id").cast(pl.Utf8),
            pl.col("resid").cast(pl.Int32),
            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
            pl.col("z").cast(pl.Float64),
        )
    chem_cast = chemical_features.select(
        pl.col("target_id").cast(pl.Utf8),
        pl.col("resid").cast(pl.Int32),
        pl.col("reactivity_dms").cast(pl.Float64),
        pl.col("reactivity_2a3").cast(pl.Float64),
    )
    outputs: dict[str, ChemicalExposureTarget] = {}
    for target_id, sequence in targets.select("target_id", "sequence").iter_rows():
        tid = str(target_id)
        parsed = parse_sequence_with_chains(
            sequence=str(sequence),
            chain_separator=chain_separator,
            stage=stage,
            location=location,
            target_id=tid,
        )
        seq = "".join(parsed.residues)
        length = len(seq)
        chem_rows = chem_cast.filter(pl.col("target_id") == tid).sort("resid")
        if chem_rows.height != length:
            raise_error(
                stage,
                location,
                "chemical_features sem cobertura completa para target",
                impact=str(abs(int(length - chem_rows.height))),
                examples=[f"{tid}:expected={length}:got={int(chem_rows.height)}"],
            )
        expected = torch.arange(1, length + 1, dtype=torch.int32)
        got = torch.tensor(chem_rows.get_column("resid").to_numpy(), dtype=torch.int32)
        if not bool(torch.equal(expected, got)):
            raise_error(
                stage,
                location,
                "resid em chemical_features fora de ordem/continuidade",
                impact="1",
                examples=[tid],
            )
        dms = torch.tensor(chem_rows.get_column("reactivity_dms").to_numpy(), dtype=torch.float32)
        a3 = torch.tensor(chem_rows.get_column("reactivity_2a3").to_numpy(), dtype=torch.float32)
        chem_exposure = (_minmax_norm(dms) + _minmax_norm(a3)) / 2.0
        source = "quickstart_only"
        if labels is not None:
            label_rows = labels.filter(pl.col("target_id") == tid).sort("resid")
            if label_rows.height != length:
                raise_error(
                    stage,
                    location,
                    "pdb_labels sem cobertura completa para target",
                    impact=str(abs(int(length - label_rows.height))),
                    examples=[f"{tid}:expected={length}:got={int(label_rows.height)}"],
                )
            coords = torch.tensor(label_rows.select("x", "y", "z").to_numpy(), dtype=torch.float32)
            if bool(torch.isnan(coords).any()):
                raise_error(stage, location, "pdb_labels contem coordenadas nulas", impact="1", examples=[tid])
            centroid = coords.mean(dim=0, keepdim=True)
            dist = torch.linalg.norm(coords - centroid, dim=1)
            geom_exposure = _minmax_norm(dist)
            chem_exposure = torch.clamp((chem_exposure + geom_exposure) / 2.0, min=0.0, max=1.0)
            source = "quickstart_pdb_cross"
        outputs[tid] = ChemicalExposureTarget(target_id=tid, sequence=seq, exposure=chem_exposure, source=source)
    if not outputs:
        raise_error(stage, location, "nenhum target para chemical mapping", impact="0", examples=[])
    return outputs
