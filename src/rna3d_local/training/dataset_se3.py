from __future__ import annotations

from pathlib import Path

from ..contracts import parse_date_column, require_columns
from ..io_tables import read_table
from ..se3.graph_builder import TargetGraph, build_target_graphs


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
    stage: str,
    location: str,
) -> list[TargetGraph]:
    targets = _read_targets(targets_path, stage=stage, location=location)
    pairings = read_table(pairings_path, stage=stage, location=location)
    chemical = read_table(chemical_features_path, stage=stage, location=location)
    labels = read_table(labels_path, stage=stage, location=location)
    return build_target_graphs(
        targets=targets,
        pairings=pairings,
        chemical_features=chemical,
        labels=labels,
        stage=stage,
        location=location,
    )


def load_inference_graphs(
    *,
    targets_path: Path,
    pairings_path: Path,
    chemical_features_path: Path,
    stage: str,
    location: str,
) -> list[TargetGraph]:
    targets = _read_targets(targets_path, stage=stage, location=location)
    pairings = read_table(pairings_path, stage=stage, location=location)
    chemical = read_table(chemical_features_path, stage=stage, location=location)
    return build_target_graphs(
        targets=targets,
        pairings=pairings,
        chemical_features=chemical,
        labels=None,
        stage=stage,
        location=location,
    )
