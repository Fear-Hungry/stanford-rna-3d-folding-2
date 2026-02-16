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
        thermo_backend=runtime.config.thermo_backend,
        rnafold_bin=runtime.config.rnafold_bin,
        linearfold_bin=runtime.config.linearfold_bin,
        thermo_cache_dir=(None if runtime.config.thermo_cache_dir is None else (repo_root / runtime.config.thermo_cache_dir).resolve()),
        msa_backend=runtime.config.msa_backend,
        mmseqs_bin=runtime.config.mmseqs_bin,
        mmseqs_db=runtime.config.mmseqs_db,
        msa_cache_dir=(None if runtime.config.msa_cache_dir is None else (repo_root / runtime.config.msa_cache_dir).resolve()),
        chain_separator=runtime.config.chain_separator,
        max_msa_sequences=runtime.config.max_msa_sequences,
        max_cov_positions=runtime.config.max_cov_positions,
        max_cov_pairs=runtime.config.max_cov_pairs,
        stage=stage,
        location=location,
    )
    rows: list[dict[str, object]] = []
    chemical_source_counts: dict[str, int] = {}
    with torch.no_grad():
        for graph_index, graph in enumerate(graphs):
            source = str(graph.chem_source)
            chemical_source_counts[source] = int(chemical_source_counts.get(source, 0) + 1)
            h, x_cond = run_backbone_for_graph(
                runtime=runtime,
                node_features=graph.node_features,
                coords_init=graph.coords_init,
                bpp_pair_src=graph.bpp_pair_src,
                bpp_pair_dst=graph.bpp_pair_dst,
                bpp_pair_prob=graph.bpp_pair_prob,
                msa_pair_src=graph.msa_pair_src,
                msa_pair_dst=graph.msa_pair_dst,
                msa_pair_prob=graph.msa_pair_prob,
                residue_index=graph.residue_index,
                chain_index=graph.chain_index,
                chem_exposure=graph.chem_exposure,
                chain_break_offset=runtime.config.chain_break_offset,
            )
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
                "thermo_backend": runtime.config.thermo_backend,
                "rnafold_bin": runtime.config.rnafold_bin,
                "linearfold_bin": runtime.config.linearfold_bin,
                "thermo_cache_dir": runtime.config.thermo_cache_dir,
                "msa_backend": runtime.config.msa_backend,
                "mmseqs_bin": runtime.config.mmseqs_bin,
                "mmseqs_db": runtime.config.mmseqs_db,
                "msa_cache_dir": runtime.config.msa_cache_dir,
                "chain_separator": runtime.config.chain_separator,
                "chain_break_offset": runtime.config.chain_break_offset,
                "max_msa_sequences": runtime.config.max_msa_sequences,
                "max_cov_positions": runtime.config.max_cov_positions,
                "max_cov_pairs": runtime.config.max_cov_pairs,
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
                "chemical_mapping_source_counts": chemical_source_counts,
            },
            "sha256": {"candidates.parquet": sha256_file(out_path)},
        },
    )
    return SampleSe3Result(candidates_path=out_path, manifest_path=manifest_path)
