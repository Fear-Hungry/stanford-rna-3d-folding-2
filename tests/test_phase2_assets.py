from __future__ import annotations

from pathlib import Path

import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.phase2_assets import build_phase2_assets_manifest


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_phase2_assets_manifest_success(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    _touch(assets / "models" / "rnapro" / "model.pt")
    _touch(assets / "models" / "rnapro" / "config.json", "{\"entrypoint\":[\"python\",\"-c\",\"print(\\\"ok\\\")\"]}")
    _touch(assets / "models" / "boltz1" / "model.safetensors")
    _touch(assets / "models" / "boltz1" / "config.json", "{\"entrypoint\":[\"python\",\"-c\",\"print(\\\"ok\\\")\"]}")
    _touch(assets / "models" / "chai1" / "model.bin")
    _touch(assets / "models" / "chai1" / "config.json", "{\"entrypoint\":[\"python\",\"-c\",\"print(\\\"ok\\\")\"]}")
    _touch(assets / "wheels" / "a.whl")
    out = build_phase2_assets_manifest(repo_root=tmp_path, assets_dir=assets)
    assert out.manifest_path.exists()


def test_build_phase2_assets_manifest_fails_when_section_empty(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    (assets / "models" / "rnapro").mkdir(parents=True)
    (assets / "models" / "boltz1").mkdir(parents=True)
    (assets / "models" / "chai1").mkdir(parents=True)
    (assets / "wheels").mkdir(parents=True)
    with pytest.raises(PipelineError, match="secao de assets vazia"):
        build_phase2_assets_manifest(repo_root=tmp_path, assets_dir=assets)
