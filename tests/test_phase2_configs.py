from __future__ import annotations

import json
from pathlib import Path

import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.phase2_configs import write_phase2_model_configs


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_write_phase2_model_configs_success(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    # rnapro
    _touch(assets / "models" / "rnapro" / "rnapro-public-best-500m.ckpt")
    _touch(assets / "models" / "rnapro" / "test_templates.pt")
    _touch(assets / "models" / "rnapro" / "ccd_cache" / "components.cif")
    _touch(assets / "models" / "rnapro" / "ccd_cache" / "components.cif.rdkit_mol.pkl")
    _touch(assets / "models" / "rnapro" / "ccd_cache" / "clusters-by-entity-40.txt")
    _touch(assets / "models" / "rnapro" / "ribonanzanet2_checkpoint" / "pairwise.yaml")
    _touch(assets / "models" / "rnapro" / "ribonanzanet2_checkpoint" / "pytorch_model_fsdp.bin")
    # boltz1
    _touch(assets / "models" / "boltz1" / "boltz1_conf.ckpt")
    _touch(assets / "models" / "boltz1" / "ccd.pkl")
    # chai1
    _touch(assets / "models" / "chai1" / "conformers_v1.apkl")
    _touch(assets / "models" / "chai1" / "esm" / "traced_sdpa_esm2_t36_3B_UR50D_fp16.pt")
    for name in [
        "feature_embedding.pt",
        "bond_loss_input_proj.pt",
        "token_embedder.pt",
        "trunk.pt",
        "diffusion_module.pt",
        "confidence_head.pt",
    ]:
        _touch(assets / "models" / "chai1" / "models_v2" / name)

    out = write_phase2_model_configs(repo_root=tmp_path, assets_dir=assets, chain_separator="|")
    assert out.manifest_path.exists()
    cfg = json.loads((assets / "models" / "chai1" / "config.json").read_text(encoding="utf-8"))
    assert cfg["entrypoint"][2] == "rna3d_local.runners.chai1"


def test_write_phase2_model_configs_fails_when_missing_weights(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    (assets / "models" / "rnapro").mkdir(parents=True)
    (assets / "models" / "boltz1").mkdir(parents=True)
    (assets / "models" / "chai1").mkdir(parents=True)
    with pytest.raises(PipelineError, match="artefatos obrigatorios do modelo ausentes"):
        write_phase2_model_configs(repo_root=tmp_path, assets_dir=assets, chain_separator="|")
