from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io_tables import write_table
from .predictor_common import build_synthetic_long_predictions, ensure_model_artifacts, load_targets_with_contract
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class RnaProOfflineResult:
    predictions_path: Path
    manifest_path: Path


def predict_rnapro_offline(
    *,
    repo_root: Path,
    model_dir: Path,
    targets_path: Path,
    out_path: Path,
    n_models: int,
) -> RnaProOfflineResult:
    stage = "RNAPRO_OFFLINE"
    location = "src/rna3d_local/rnapro_offline.py:predict_rnapro_offline"
    ensure_model_artifacts(
        model_dir=model_dir,
        required_files=["model.pt", "config.json"],
        stage=stage,
        location=location,
    )
    targets = load_targets_with_contract(targets_path=targets_path, stage=stage, location=location)
    predictions = build_synthetic_long_predictions(
        targets=targets,
        n_models=int(n_models),
        source="rnapro",
        base_scale=8.0,
        confidence_base=0.72,
        ligand_bonus=0.03,
    )
    write_table(predictions, out_path)
    manifest_path = out_path.parent / "rnapro_offline_manifest.json"
    write_json(
        manifest_path,
        {
            "created_utc": utc_now_iso(),
            "paths": {
                "model_dir": rel_or_abs(model_dir, repo_root),
                "targets": rel_or_abs(targets_path, repo_root),
                "predictions": rel_or_abs(out_path, repo_root),
            },
            "params": {"n_models": int(n_models)},
            "stats": {
                "n_rows": int(predictions.height),
                "n_targets": int(predictions.get_column("target_id").n_unique()),
            },
            "sha256": {"predictions.parquet": sha256_file(out_path)},
        },
    )
    return RnaProOfflineResult(predictions_path=out_path, manifest_path=manifest_path)
