from __future__ import annotations

import json
from pathlib import Path

import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.phase2_assets import build_phase2_assets_manifest


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_phase2_assets_manifest_success(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    _touch(assets / "models" / "rnapro" / "rnapro-public-best-500m.ckpt")
    _touch(assets / "models" / "rnapro" / "config.json", "{\"entrypoint\":[\"python\",\"-c\",\"print(\\\"ok\\\")\"]}")
    _touch(assets / "models" / "boltz1" / "boltz1_conf.ckpt")
    _touch(assets / "models" / "boltz1" / "ccd.pkl")
    _touch(assets / "models" / "boltz1" / "config.json", "{\"entrypoint\":[\"python\",\"-c\",\"print(\\\"ok\\\")\"]}")
    _touch(assets / "models" / "chai1" / "conformers_v1.apkl")
    _touch(assets / "models" / "chai1" / "esm" / "traced_sdpa_esm2_t36_3B_UR50D_fp16.pt")
    _touch(assets / "models" / "chai1" / "models_v2" / "feature_embedding.pt")
    _touch(assets / "models" / "chai1" / "models_v2" / "bond_loss_input_proj.pt")
    _touch(assets / "models" / "chai1" / "models_v2" / "token_embedder.pt")
    _touch(assets / "models" / "chai1" / "models_v2" / "trunk.pt")
    _touch(assets / "models" / "chai1" / "models_v2" / "diffusion_module.pt")
    _touch(assets / "models" / "chai1" / "models_v2" / "confidence_head.pt")
    _touch(assets / "models" / "chai1" / "config.json", "{\"entrypoint\":[\"python\",\"-c\",\"print(\\\"ok\\\")\"]}")
    _touch(assets / "wheels" / "a.whl")
    out = build_phase2_assets_manifest(repo_root=tmp_path, assets_dir=assets)
    assert out.manifest_path.exists()


def test_build_phase2_assets_manifest_allows_model_config_without_entrypoint(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    _touch(assets / "models" / "rnapro" / "rnapro-public-best-500m.ckpt")
    _touch(assets / "models" / "rnapro" / "config.json", "{\"train_template_kernel\":\"x\"}")
    _touch(assets / "models" / "boltz1" / "boltz1_conf.ckpt")
    _touch(assets / "models" / "boltz1" / "ccd.pkl")
    _touch(assets / "models" / "boltz1" / "config.json", "{\"entrypoint\":[\"python\",\"-c\",\"print(\\\"ok\\\")\"]}")
    _touch(assets / "models" / "chai1" / "conformers_v1.apkl")
    _touch(assets / "models" / "chai1" / "esm" / "traced_sdpa_esm2_t36_3B_UR50D_fp16.pt")
    _touch(assets / "models" / "chai1" / "models_v2" / "feature_embedding.pt")
    _touch(assets / "models" / "chai1" / "models_v2" / "bond_loss_input_proj.pt")
    _touch(assets / "models" / "chai1" / "models_v2" / "token_embedder.pt")
    _touch(assets / "models" / "chai1" / "models_v2" / "trunk.pt")
    _touch(assets / "models" / "chai1" / "models_v2" / "diffusion_module.pt")
    _touch(assets / "models" / "chai1" / "models_v2" / "confidence_head.pt")
    _touch(assets / "models" / "chai1" / "config.json", "{\"entrypoint\":[\"python\",\"-c\",\"print(\\\"ok\\\")\"]}")
    _touch(assets / "wheels" / "a.whl")

    out = build_phase2_assets_manifest(repo_root=tmp_path, assets_dir=assets)
    payload = json.loads(out.manifest_path.read_text(encoding="utf-8"))
    sections = payload["sections"]
    assert sections["models/rnapro"]["entrypoint"] is None


def test_build_phase2_assets_manifest_fails_when_section_empty(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    (assets / "models" / "rnapro").mkdir(parents=True)
    (assets / "models" / "boltz1").mkdir(parents=True)
    (assets / "models" / "chai1").mkdir(parents=True)
    (assets / "wheels").mkdir(parents=True)
    with pytest.raises(PipelineError, match="secao de assets vazia"):
        build_phase2_assets_manifest(repo_root=tmp_path, assets_dir=assets)
