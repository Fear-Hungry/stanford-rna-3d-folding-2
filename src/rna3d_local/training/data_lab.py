from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..utils import rel_or_abs, sha256_file, utc_now_iso, write_json
from .dataset_se3 import load_training_graphs
from .store_zarr import build_training_store_zarr, ensure_zarr_available


@dataclass(frozen=True)
class PreparePhase1DataLabResult:
    store_path: Path
    store_manifest_path: Path
    manifest_path: Path
    thermo_cache_dir: Path
    msa_cache_dir: Path


def prepare_phase1_data_lab(
    *,
    repo_root: Path,
    targets_path: Path,
    pairings_path: Path,
    chemical_features_path: Path,
    labels_path: Path,
    out_dir: Path,
    thermo_backend: str,
    rnafold_bin: str,
    linearfold_bin: str,
    msa_backend: str,
    mmseqs_bin: str,
    mmseqs_db: str,
    chain_separator: str,
    max_msa_sequences: int,
    max_cov_positions: int,
    max_cov_pairs: int,
    num_workers: int,
) -> PreparePhase1DataLabResult:
    stage = "PHASE1_DATALAB"
    location = "src/rna3d_local/training/data_lab.py:prepare_phase1_data_lab"
    ensure_zarr_available(stage=stage, location=location)
    out_dir.mkdir(parents=True, exist_ok=True)
    thermo_cache_dir = (out_dir / "thermo_cache").resolve()
    msa_cache_dir = (out_dir / "msa_cache").resolve()
    store_path = (out_dir / "training_store.zarr").resolve()
    store_manifest_path = (out_dir / "training_store_manifest.json").resolve()
    manifest_path = (out_dir / "phase1_data_lab_manifest.json").resolve()

    graphs = load_training_graphs(
        targets_path=targets_path,
        pairings_path=pairings_path,
        chemical_features_path=chemical_features_path,
        labels_path=labels_path,
        thermo_backend=thermo_backend,
        rnafold_bin=rnafold_bin,
        linearfold_bin=linearfold_bin,
        thermo_cache_dir=thermo_cache_dir,
        thermo_pair_min_prob=0.0,
        thermo_pair_max_per_node=0,
        msa_backend=msa_backend,
        mmseqs_bin=mmseqs_bin,
        mmseqs_db=mmseqs_db,
        msa_cache_dir=msa_cache_dir,
        chain_separator=chain_separator,
        max_msa_sequences=max_msa_sequences,
        max_cov_positions=max_cov_positions,
        max_cov_pairs=max_cov_pairs,
        stage=stage,
        location=location,
        thermo_num_workers=int(num_workers),
        msa_num_workers=int(num_workers),
    )
    store_build = build_training_store_zarr(
        repo_root=repo_root,
        graphs=graphs,
        store_path=store_path,
        manifest_path=store_manifest_path,
        stage=stage,
        location=location,
    )
    write_json(
        manifest_path,
        {
            "created_utc": utc_now_iso(),
            "paths": {
                "targets": rel_or_abs(targets_path, repo_root),
                "pairings": rel_or_abs(pairings_path, repo_root),
                "chemical_features": rel_or_abs(chemical_features_path, repo_root),
                "labels": rel_or_abs(labels_path, repo_root),
                "out_dir": rel_or_abs(out_dir, repo_root),
                "thermo_cache_dir": rel_or_abs(thermo_cache_dir, repo_root),
                "msa_cache_dir": rel_or_abs(msa_cache_dir, repo_root),
                "training_store": rel_or_abs(store_build.store_path, repo_root),
                "training_store_manifest": rel_or_abs(store_build.manifest_path, repo_root),
            },
            "params": {
                "thermo_backend": str(thermo_backend),
                "rnafold_bin": str(rnafold_bin),
                "linearfold_bin": str(linearfold_bin),
                "msa_backend": str(msa_backend),
                "mmseqs_bin": str(mmseqs_bin),
                "mmseqs_db": str(mmseqs_db),
                "chain_separator": str(chain_separator),
                "max_msa_sequences": int(max_msa_sequences),
                "max_cov_positions": int(max_cov_positions),
                "max_cov_pairs": int(max_cov_pairs),
                "num_workers": int(num_workers),
                "store_format": "zarr",
            },
            "stats": {
                "n_targets": int(store_build.n_targets),
                "n_residues_total": int(store_build.n_residues_total),
                "input_dim": int(store_build.input_dim),
                "max_target_length": int(store_build.max_target_length),
            },
            "sha256": {
                "training_store_manifest.json": sha256_file(store_build.manifest_path),
            },
        },
    )
    return PreparePhase1DataLabResult(
        store_path=store_build.store_path,
        store_manifest_path=store_build.manifest_path,
        manifest_path=manifest_path,
        thermo_cache_dir=thermo_cache_dir,
        msa_cache_dir=msa_cache_dir,
    )
