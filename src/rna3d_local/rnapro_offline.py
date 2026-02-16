from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import raise_error
from .io_tables import read_table
from .predictor_common import (
    ensure_model_artifacts,
    load_model_entrypoint,
    load_targets_with_contract,
    render_entrypoint,
    run_external_entrypoint,
    validate_long_predictions,
)
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
        required_files=[
            "config.json",
            "rnapro-public-best-500m.ckpt",
            "test_templates.pt",
            "ccd_cache/components.cif",
            "ccd_cache/components.cif.rdkit_mol.pkl",
            "ccd_cache/clusters-by-entity-40.txt",
            "ribonanzanet2_checkpoint/pairwise.yaml",
            "ribonanzanet2_checkpoint/pytorch_model_fsdp.bin",
        ],
        stage=stage,
        location=location,
    )
    targets = load_targets_with_contract(targets_path=targets_path, stage=stage, location=location)
    entrypoint = load_model_entrypoint(model_dir=model_dir, stage=stage, location=location)
    cmd = render_entrypoint(entrypoint, model_dir=model_dir, targets_path=targets_path, out_path=out_path, n_models=int(n_models))
    run_external_entrypoint(model_dir=model_dir, entrypoint=cmd, stage=stage, location=location)
    if not out_path.exists():
        raise_error(stage, location, "runner rnapro nao gerou arquivo de saida", impact="1", examples=[str(out_path)])
    predictions = read_table(out_path, stage=stage, location=location)
    validate_long_predictions(predictions=predictions, targets=targets, n_models=int(n_models), stage=stage, location=location, label="rnapro_predictions")
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
            "params": {"n_models": int(n_models), "entrypoint": cmd},
            "stats": {
                "n_rows": int(predictions.height),
                "n_targets": int(predictions.get_column("target_id").n_unique()),
            },
            "sha256": {"predictions.parquet": sha256_file(out_path)},
        },
    )
    return RnaProOfflineResult(predictions_path=out_path, manifest_path=manifest_path)
