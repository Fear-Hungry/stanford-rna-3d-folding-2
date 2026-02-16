from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch

from .errors import raise_error
from .generative.sampler import sample_methods_for_target
from .io_tables import write_table
from .training.dataset_se3 import load_inference_graphs
from .training.trainer_se3 import (
    TrainSe3Result,
    load_se3_runtime_models,
    run_backbone_for_graph,
    train_se3_generator as train_se3_generator_impl,
)
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class SampleSe3Result:
    candidates_path: Path
    manifest_path: Path


def train_se3_generator(
    *,
    repo_root: Path,
    targets_path: Path,
    pairings_path: Path,
    chemical_features_path: Path,
    labels_path: Path,
    config_path: Path,
    out_dir: Path,
    seed: int,
) -> TrainSe3Result:
    return train_se3_generator_impl(
        repo_root=repo_root,
        targets_path=targets_path,
        pairings_path=pairings_path,
        chemical_features_path=chemical_features_path,
        labels_path=labels_path,
        config_path=config_path,
        out_dir=out_dir,
        seed=seed,
    )


def sample_se3_ensemble(
    *,
    repo_root: Path,
    model_dir: Path,
    targets_path: Path,
    pairings_path: Path,
    chemical_features_path: Path,
    out_path: Path,
    method: str,
    n_samples: int,
    seed: int,
) -> SampleSe3Result:
    stage = "SAMPLE_SE3"
    location = "src/rna3d_local/se3_pipeline.py:sample_se3_ensemble"
    if n_samples <= 0:
        raise_error(stage, location, "n_samples deve ser > 0", impact="1", examples=[str(n_samples)])
    runtime = load_se3_runtime_models(model_dir=model_dir, stage=stage, location=location)
    graphs = load_inference_graphs(
        targets_path=targets_path,
        pairings_path=pairings_path,
        chemical_features_path=chemical_features_path,
        stage=stage,
        location=location,
    )
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for graph_index, graph in enumerate(graphs):
            h, x_cond = run_backbone_for_graph(runtime=runtime, node_features=graph.node_features, coords_init=graph.coords_init)
            samples = sample_methods_for_target(
                target_id=graph.target_id,
                h=h,
                x_cond=x_cond,
                method=method,
                n_samples=int(n_samples),
                base_seed=int(seed + (graph_index * 1000)),
                diffusion=runtime.diffusion,
                flow=runtime.flow,
                stage=stage,
                location=location,
            )
            for method_name, sample_rank, coords in samples:
                sample_id = f"{method_name}_{sample_rank}"
                confidence = 0.78 if method_name == "diffusion" else 0.75
                for idx, resid in enumerate(graph.resids):
                    rows.append(
                        {
                            "target_id": graph.target_id,
                            "sample_id": sample_id,
                            "resid": int(resid),
                            "resname": graph.resnames[idx],
                            "x": float(coords[idx, 0].item()),
                            "y": float(coords[idx, 1].item()),
                            "z": float(coords[idx, 2].item()),
                            "method": method_name,
                            "source": "generative_se3",
                            "confidence": float(confidence),
                        }
                    )
    if not rows:
        raise_error(stage, location, "nenhuma coordenada gerada em sample_se3_ensemble", impact="0", examples=[])
    out = pl.DataFrame(rows).sort(["target_id", "sample_id", "resid"])
    write_table(out, out_path)
    manifest_path = out_path.parent / "sample_se3_manifest.json"
    write_json(
        manifest_path,
        {
            "created_utc": utc_now_iso(),
            "params": {
                "method": str(method),
                "n_samples": int(n_samples),
                "seed": int(seed),
                "sequence_tower": runtime.config.sequence_tower,
                "sequence_heads": runtime.config.sequence_heads,
                "graph_backend": runtime.config.graph_backend,
                "radius_angstrom": runtime.config.radius_angstrom,
                "max_neighbors": runtime.config.max_neighbors,
                "graph_chunk_size": runtime.config.graph_chunk_size,
            },
            "paths": {
                "model_dir": rel_or_abs(model_dir, repo_root),
                "targets": rel_or_abs(targets_path, repo_root),
                "pairings": rel_or_abs(pairings_path, repo_root),
                "chemical_features": rel_or_abs(chemical_features_path, repo_root),
                "candidates": rel_or_abs(out_path, repo_root),
            },
            "stats": {
                "n_rows": int(out.height),
                "n_targets": int(out.get_column("target_id").n_unique()),
                "n_samples_unique": int(out.select("target_id", "sample_id").unique().height),
            },
            "sha256": {"candidates.parquet": sha256_file(out_path)},
        },
    )
    return SampleSe3Result(candidates_path=out_path, manifest_path=manifest_path)
