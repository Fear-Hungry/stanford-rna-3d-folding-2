from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
import torch

from ..contracts import require_columns
from ..errors import raise_error
from .sequence_parser import parse_sequence_with_chains

if TYPE_CHECKING:
    from ..training.chemical_mapping import ChemicalExposureTarget
    from ..training.msa_covariance import MsaCovTarget
    from ..training.thermo_2d import ThermoBppTarget


_BASE_VEC = {
    "A": [1.0, 0.0, 0.0, 0.0],
    "C": [0.0, 1.0, 0.0, 0.0],
    "G": [0.0, 0.0, 1.0, 0.0],
    "U": [0.0, 0.0, 0.0, 1.0],
    "N": [0.25, 0.25, 0.25, 0.25],
}


@dataclass(frozen=True)
class TargetGraph:
    target_id: str
    resids: list[int]
    resnames: list[str]
    chain_index: torch.Tensor
    residue_index: torch.Tensor
    chem_exposure: torch.Tensor
    chem_source: str
    node_features: torch.Tensor
    bpp_pair_src: torch.Tensor
    bpp_pair_dst: torch.Tensor
    bpp_pair_prob: torch.Tensor
    msa_pair_src: torch.Tensor
    msa_pair_dst: torch.Tensor
    msa_pair_prob: torch.Tensor
    coords_init: torch.Tensor
    coords_true: torch.Tensor | None


def _sequence_rows(targets: pl.DataFrame, *, chain_separator: str, stage: str, location: str) -> pl.DataFrame:
    require_columns(targets, ["target_id", "sequence"], stage=stage, location=location, label="targets")
    rows: list[dict[str, object]] = []
    for target_id, sequence in targets.select("target_id", "sequence").iter_rows():
        tid = str(target_id)
        if not tid:
            raise_error(stage, location, "target_id vazio", impact="1", examples=[str(target_id)])
        parsed = parse_sequence_with_chains(
            sequence=str(sequence),
            chain_separator=chain_separator,
            stage=stage,
            location=location,
            target_id=tid,
        )
        length = len(parsed.residues)
        for idx, base in enumerate(parsed.residues, start=1):
            vec = _BASE_VEC[base]
            rows.append(
                {
                    "target_id": tid,
                    "resid": int(idx),
                    "residue_index_1d": int(parsed.residue_position_index_1d[idx - 1]),
                    "resname": base,
                    "chain_index": int(parsed.chain_index[idx - 1]),
                    "base_a": vec[0],
                    "base_c": vec[1],
                    "base_g": vec[2],
                    "base_u": vec[3],
                    "resid_norm": float(idx) / float(length),
                }
            )
    return pl.DataFrame(rows)


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


def _build_init_coords(length: int) -> torch.Tensor:
    positions = torch.arange(float(length), dtype=torch.float32)
    x = positions * 3.2
    y = torch.sin(positions * 0.4) * 1.5
    z = torch.cos(positions * 0.4) * 1.5
    return torch.stack([x, y, z], dim=1)


def build_target_graphs(
    *,
    targets: pl.DataFrame,
    pairings: pl.DataFrame,
    chemical_features: pl.DataFrame,
    labels: pl.DataFrame | None,
    thermo_bpp: dict[str, "ThermoBppTarget"],
    msa_cov: dict[str, "MsaCovTarget"],
    chemical_mapping: dict[str, "ChemicalExposureTarget"],
    chain_separator: str,
    stage: str,
    location: str,
) -> list[TargetGraph]:
    seq_df = _sequence_rows(targets, chain_separator=chain_separator, stage=stage, location=location)
    require_columns(pairings, ["target_id", "resid", "pair_prob"], stage=stage, location=location, label="pairings")
    require_columns(
        chemical_features,
        ["target_id", "resid", "p_open", "p_paired"],
        stage=stage,
        location=location,
        label="chemical_features",
    )
    _assert_unique_keys(pairings, label="pairings", stage=stage, location=location)
    _assert_unique_keys(chemical_features, label="chemical_features", stage=stage, location=location)

    joined = (
        seq_df.join(
            pairings.select(
                pl.col("target_id").cast(pl.Utf8),
                pl.col("resid").cast(pl.Int32),
                pl.col("pair_prob").cast(pl.Float64),
            ),
            on=["target_id", "resid"],
            how="left",
        )
        .join(
            chemical_features.select(
                pl.col("target_id").cast(pl.Utf8),
                pl.col("resid").cast(pl.Int32),
                pl.col("p_open").cast(pl.Float64),
                pl.col("p_paired").cast(pl.Float64),
            ),
            on=["target_id", "resid"],
            how="left",
        )
        .sort(["target_id", "resid"])
    )
    bad_cond = joined.filter(
        pl.col("pair_prob").is_null() | pl.col("p_open").is_null() | pl.col("p_paired").is_null()
    )
    if bad_cond.height > 0:
        examples = bad_cond.select("target_id", "resid").head(8).rows()
        raise_error(stage, location, "condicionamento incompleto (pairings/quimica)", impact=str(int(bad_cond.height)), examples=[str(item) for item in examples])

    labels_cast: pl.DataFrame | None = None
    if labels is not None:
        require_columns(labels, ["target_id", "resid", "x", "y", "z"], stage=stage, location=location, label="labels")
        _assert_unique_keys(labels, label="labels", stage=stage, location=location)
        labels_cast = labels.select(
            pl.col("target_id").cast(pl.Utf8),
            pl.col("resid").cast(pl.Int32),
            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
            pl.col("z").cast(pl.Float64),
        )

    graphs: list[TargetGraph] = []
    for target_id, part in joined.group_by("target_id", maintain_order=True):
        tid = str(target_id[0]) if isinstance(target_id, tuple) else str(target_id)
        rows = part.sort("resid")
        bpp_target = thermo_bpp.get(tid)
        if bpp_target is None:
            raise_error(stage, location, "BPP termodinamico ausente para target", impact="1", examples=[tid])
        if int(bpp_target.paired_marginal.numel()) != int(rows.height):
            raise_error(
                stage,
                location,
                "BPP termodinamico com comprimento inconsistente",
                impact="1",
                examples=[f"{tid}:bpp={int(bpp_target.paired_marginal.numel())}:seq={int(rows.height)}"],
            )
        msa_target = msa_cov.get(tid)
        if msa_target is None:
            raise_error(stage, location, "MSA/covariancia ausente para target", impact="1", examples=[tid])
        if int(msa_target.cov_marginal.numel()) != int(rows.height):
            raise_error(
                stage,
                location,
                "MSA/covariancia com comprimento inconsistente",
                impact="1",
                examples=[f"{tid}:msa={int(msa_target.cov_marginal.numel())}:seq={int(rows.height)}"],
            )
        chem_target = chemical_mapping.get(tid)
        if chem_target is None:
            raise_error(stage, location, "chemical mapping ausente para target", impact="1", examples=[tid])
        if int(chem_target.exposure.numel()) != int(rows.height):
            raise_error(
                stage,
                location,
                "chemical mapping com comprimento inconsistente",
                impact="1",
                examples=[f"{tid}:chem={int(chem_target.exposure.numel())}:seq={int(rows.height)}"],
            )
        features_base = torch.tensor(
            rows.select(
                "base_a",
                "base_c",
                "base_g",
                "base_u",
                "resid_norm",
                "pair_prob",
                "p_open",
                "p_paired",
            ).to_numpy(),
            dtype=torch.float32,
        )
        features = torch.cat(
            [
                features_base,
                bpp_target.paired_marginal.unsqueeze(-1).to(dtype=torch.float32),
                msa_target.cov_marginal.unsqueeze(-1).to(dtype=torch.float32),
                chem_target.exposure.unsqueeze(-1).to(dtype=torch.float32),
            ],
            dim=1,
        )
        coords_init = _build_init_coords(int(rows.height))
        chain_index_tensor = torch.tensor(rows.get_column("chain_index").to_numpy(), dtype=torch.long)
        residue_index_tensor = torch.tensor(rows.get_column("residue_index_1d").to_numpy(), dtype=torch.long)
        coords_true: torch.Tensor | None = None
        if labels_cast is not None:
            target_labels = labels_cast.filter(pl.col("target_id") == tid).sort("resid")
            if target_labels.height != rows.height:
                raise_error(
                    stage,
                    location,
                    "labels sem cobertura completa do target",
                    impact=str(abs(int(rows.height - target_labels.height))),
                    examples=[tid],
                )
            coords_true = torch.tensor(target_labels.select("x", "y", "z").to_numpy(), dtype=torch.float32)
        graphs.append(
            TargetGraph(
                target_id=tid,
                resids=[int(item) for item in rows.get_column("resid").to_list()],
                resnames=[str(item) for item in rows.get_column("resname").to_list()],
                chain_index=chain_index_tensor,
                residue_index=residue_index_tensor,
                chem_exposure=chem_target.exposure.to(dtype=torch.float32),
                chem_source=str(chem_target.source),
                node_features=features,
                bpp_pair_src=bpp_target.pair_src.to(dtype=torch.long),
                bpp_pair_dst=bpp_target.pair_dst.to(dtype=torch.long),
                bpp_pair_prob=bpp_target.pair_prob.to(dtype=torch.float32),
                msa_pair_src=msa_target.pair_src.to(dtype=torch.long),
                msa_pair_dst=msa_target.pair_dst.to(dtype=torch.long),
                msa_pair_prob=msa_target.pair_prob.to(dtype=torch.float32),
                coords_init=coords_init,
                coords_true=coords_true,
            )
        )
    if not graphs:
        raise_error(stage, location, "nenhum grafo de target construido", impact="0", examples=[])
    return graphs
