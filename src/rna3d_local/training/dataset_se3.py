from __future__ import annotations

from pathlib import Path

from ..contracts import parse_date_column, require_columns
from ..io_tables import read_table
from ..se3.graph_builder import TargetGraph, build_target_graphs
from .chemical_mapping import compute_chemical_exposure_mapping
from .msa_covariance import compute_msa_covariance
from .thermo_2d import compute_thermo_bpp


def _read_targets(path: Path, *, stage: str, location: str):
    targets = read_table(path, stage=stage, location=location)
    require_columns(targets, ["target_id", "sequence", "temporal_cutoff"], stage=stage, location=location, label="targets")
    return parse_date_column(targets, "temporal_cutoff", stage=stage, location=location, label="targets")


def load_training_graphs(
    *,
    targets_path: Path,
    pairings_path: Path,
    chemical_features_path: Path,
    labels_path: Path,
    thermo_backend: str,
    rnafold_bin: str,
    linearfold_bin: str,
    thermo_cache_dir: Path | None,
    msa_backend: str,
    mmseqs_bin: str,
    mmseqs_db: str,
    msa_cache_dir: Path | None,
    chain_separator: str,
    max_msa_sequences: int,
    max_cov_positions: int,
    max_cov_pairs: int,
    stage: str,
    location: str,
    thermo_num_workers: int = 1,
    msa_num_workers: int = 1,
) -> list[TargetGraph]:
    targets = _read_targets(targets_path, stage=stage, location=location)
    pairings = read_table(pairings_path, stage=stage, location=location)
    chemical = read_table(chemical_features_path, stage=stage, location=location)
    labels = read_table(labels_path, stage=stage, location=location)
    thermo = compute_thermo_bpp(
        targets=targets,
        backend=thermo_backend,
        rnafold_bin=rnafold_bin,
        linearfold_bin=linearfold_bin,
        cache_dir=thermo_cache_dir,
        chain_separator=chain_separator,
        stage=stage,
        location=location,
        num_workers=int(thermo_num_workers),
    )
    chem_mapping = compute_chemical_exposure_mapping(
        targets=targets,
        chemical_features=chemical,
        pdb_labels=labels,
        chain_separator=chain_separator,
        stage=stage,
        location=location,
    )
    msa = compute_msa_covariance(
        targets=targets,
        backend=msa_backend,
        mmseqs_bin=mmseqs_bin,
        mmseqs_db=mmseqs_db,
        cache_dir=msa_cache_dir,
        chain_separator=chain_separator,
        max_msa_sequences=max_msa_sequences,
        max_cov_positions=max_cov_positions,
        max_cov_pairs=max_cov_pairs,
        stage=stage,
        location=location,
        num_workers=int(msa_num_workers),
    )
    return build_target_graphs(
        targets=targets,
        pairings=pairings,
        chemical_features=chemical,
        labels=labels,
        thermo_bpp=thermo,
        msa_cov=msa,
        chemical_mapping=chem_mapping,
        chain_separator=chain_separator,
        stage=stage,
        location=location,
    )


def load_inference_graphs(
    *,
    targets_path: Path,
    pairings_path: Path,
    chemical_features_path: Path,
    thermo_backend: str,
    rnafold_bin: str,
    linearfold_bin: str,
    thermo_cache_dir: Path | None,
    msa_backend: str,
    mmseqs_bin: str,
    mmseqs_db: str,
    msa_cache_dir: Path | None,
    chain_separator: str,
    max_msa_sequences: int,
    max_cov_positions: int,
    max_cov_pairs: int,
    stage: str,
    location: str,
    thermo_num_workers: int = 1,
    msa_num_workers: int = 1,
) -> list[TargetGraph]:
    targets = _read_targets(targets_path, stage=stage, location=location)
    pairings = read_table(pairings_path, stage=stage, location=location)
    chemical = read_table(chemical_features_path, stage=stage, location=location)
    thermo = compute_thermo_bpp(
        targets=targets,
        backend=thermo_backend,
        rnafold_bin=rnafold_bin,
        linearfold_bin=linearfold_bin,
        cache_dir=thermo_cache_dir,
        chain_separator=chain_separator,
        stage=stage,
        location=location,
        num_workers=int(thermo_num_workers),
    )
    chem_mapping = compute_chemical_exposure_mapping(
        targets=targets,
        chemical_features=chemical,
        pdb_labels=None,
        chain_separator=chain_separator,
        stage=stage,
        location=location,
    )
    msa = compute_msa_covariance(
        targets=targets,
        backend=msa_backend,
        mmseqs_bin=mmseqs_bin,
        mmseqs_db=mmseqs_db,
        cache_dir=msa_cache_dir,
        chain_separator=chain_separator,
        max_msa_sequences=max_msa_sequences,
        max_cov_positions=max_cov_positions,
        max_cov_pairs=max_cov_pairs,
        stage=stage,
        location=location,
        num_workers=int(msa_num_workers),
    )
    return build_target_graphs(
        targets=targets,
        pairings=pairings,
        chemical_features=chemical,
        labels=None,
        thermo_bpp=thermo,
        msa_cov=msa,
        chemical_mapping=chem_mapping,
        chain_separator=chain_separator,
        stage=stage,
        location=location,
    )
