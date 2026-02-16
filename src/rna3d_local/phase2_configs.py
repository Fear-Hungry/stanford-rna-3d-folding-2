from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import raise_error
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class Phase2ConfigsResult:
    manifest_path: Path


def _write_config(*, path: Path, entrypoint: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"entrypoint": [str(x) for x in entrypoint]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_phase2_model_configs(
    *,
    repo_root: Path,
    assets_dir: Path,
    chain_separator: str = "|",
    manifest_path: Path | None = None,
) -> Phase2ConfigsResult:
    stage = "PHASE2_CONFIGS"
    location = "src/rna3d_local/phase2_configs.py:write_phase2_model_configs"
    assets_dir = assets_dir.resolve()
    if not assets_dir.exists():
        raise_error(stage, location, "assets_dir ausente", impact="1", examples=[str(assets_dir)])
    if not str(chain_separator):
        raise_error(stage, location, "chain_separator vazio", impact="1", examples=[repr(chain_separator)])

    model_dirs = {
        "chai1": assets_dir / "models" / "chai1",
        "boltz1": assets_dir / "models" / "boltz1",
        "rnapro": assets_dir / "models" / "rnapro",
    }
    missing_dirs = [str(path) for path in model_dirs.values() if not path.exists()]
    if missing_dirs:
        raise_error(stage, location, "diretorios de modelos ausentes em assets", impact=str(len(missing_dirs)), examples=missing_dirs[:8])

    # Fail-fast on missing required artifacts (weights) for each model.
    required_files: dict[str, list[str]] = {
        "chai1": [
            "conformers_v1.apkl",
            "esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt",
            "models_v2/feature_embedding.pt",
            "models_v2/bond_loss_input_proj.pt",
            "models_v2/token_embedder.pt",
            "models_v2/trunk.pt",
            "models_v2/diffusion_module.pt",
            "models_v2/confidence_head.pt",
        ],
        "boltz1": ["boltz1_conf.ckpt", "ccd.pkl"],
        "rnapro": [
            "rnapro-public-best-500m.ckpt",
            "test_templates.pt",
            "ccd_cache/components.cif",
            "ccd_cache/components.cif.rdkit_mol.pkl",
            "ccd_cache/clusters-by-entity-40.txt",
            "ribonanzanet2_checkpoint/pairwise.yaml",
            "ribonanzanet2_checkpoint/pytorch_model_fsdp.bin",
        ],
    }
    for name, base in model_dirs.items():
        missing = [str(base / rel) for rel in required_files.get(name, []) if not (base / rel).exists()]
        if missing:
            raise_error(stage, location, "artefatos obrigatorios do modelo ausentes", impact=str(len(missing)), examples=[f"{name}:{x}" for x in missing[:8]])

    # Write config.json entrypoints pointing at versioned runners in this repo.
    configs: dict[str, Path] = {}
    for model_name, model_dir in model_dirs.items():
        cfg = model_dir / "config.json"
        entrypoint = [
            "python",
            "-m",
            f"rna3d_local.runners.{model_name}",
            "--model-dir",
            "{model_dir}",
            "--targets",
            "{targets}",
            "--out",
            "{out}",
            "--n-models",
            "{n_models}",
            "--chain-separator",
            str(chain_separator),
        ]
        _write_config(path=cfg, entrypoint=entrypoint)
        configs[model_name] = cfg

    out_manifest = manifest_path if manifest_path is not None else (assets_dir / "runtime" / "phase2_configs_manifest.json")
    payload = {
        "created_utc": utc_now_iso(),
        "assets_dir": rel_or_abs(assets_dir, repo_root),
        "chain_separator": str(chain_separator),
        "configs": {
            name: {
                "path": str(path.relative_to(assets_dir)),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for name, path in configs.items()
        },
    }
    write_json(out_manifest, payload)
    return Phase2ConfigsResult(manifest_path=out_manifest)
