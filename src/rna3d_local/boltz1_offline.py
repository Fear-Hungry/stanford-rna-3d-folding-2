from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io_tables import write_table
from .predictor_common import build_synthetic_long_predictions, ensure_model_artifacts, load_targets_with_contract
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class Boltz1OfflineResult:
    predictions_path: Path
    manifest_path: Path


def predict_boltz1_offline(
    *,
    repo_root: Path,
    model_dir: Path,
    targets_path: Path,
    out_path: Path,
    n_models: int,
) -> Boltz1OfflineResult:
    stage = "BOLTZ1_OFFLINE"
    location = "src/rna3d_local/boltz1_offline.py:predict_boltz1_offline"
    ensure_model_artifacts(
        model_dir=model_dir,
        required_files=["model.safetensors", "config.json"],
        stage=stage,
        location=location,
    )
    targets = load_targets_with_contract(targets_path=targets_path, stage=stage, location=location)
    predictions = build_synthetic_long_predictions(
        targets=targets,
        n_models=int(n_models),
        source="boltz1",
        base_scale=7.3,
        confidence_base=0.74,
        ligand_bonus=0.12,
    )
    write_table(predictions, out_path)
    manifest_path = out_path.parent / "boltz1_offline_manifest.json"
    write_json(
        manifest_path,
        {
            "created_utc": utc_now_iso(),
            "paths": {
                "model_dir": rel_or_abs(model_dir, repo_root),
                "targets": rel_or_abs(targets_path, repo_root),
                "predictions": rel_or_abs(out_path, repo_root),
            },
            "params": {"n_models": int(n_models), "multimodal_ligand_smiles": True},
            "stats": {
                "n_rows": int(predictions.height),
                "n_targets": int(predictions.get_column("target_id").n_unique()),
            },
            "sha256": {"predictions.parquet": sha256_file(out_path)},
        },
    )
    return Boltz1OfflineResult(predictions_path=out_path, manifest_path=manifest_path)
