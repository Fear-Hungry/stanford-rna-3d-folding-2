from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import raise_error
from .predictor_common import load_model_entrypoint
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class Phase2AssetsResult:
    manifest_path: Path


def _list_files_recursive(base: Path) -> list[Path]:
    return sorted([p for p in base.rglob("*") if p.is_file()], key=lambda x: str(x))


def build_phase2_assets_manifest(
    *,
    repo_root: Path,
    assets_dir: Path,
    manifest_path: Path | None = None,
) -> Phase2AssetsResult:
    stage = "PHASE2_ASSETS"
    location = "src/rna3d_local/phase2_assets.py:build_phase2_assets_manifest"
    if not assets_dir.exists():
        raise_error(stage, location, "assets_dir ausente", impact="1", examples=[str(assets_dir)])
    required_dirs = [
        assets_dir / "models" / "rnapro",
        assets_dir / "models" / "boltz1",
        assets_dir / "models" / "chai1",
        assets_dir / "wheels",
    ]
    missing_dirs = [str(path) for path in required_dirs if not path.exists()]
    if missing_dirs:
        raise_error(stage, location, "diretorios obrigatorios de assets ausentes", impact=str(len(missing_dirs)), examples=missing_dirs[:8])

    sections: dict[str, dict[str, object]] = {}
    for section_path in required_dirs:
        files = _list_files_recursive(section_path)
        if not files:
            raise_error(stage, location, "secao de assets vazia", impact="1", examples=[str(section_path)])
        section_key = str(section_path.relative_to(assets_dir))
        entrypoint = None
        if section_key.startswith("models/"):
            entrypoint = load_model_entrypoint(model_dir=section_path, stage=stage, location=location)
        sections[section_key] = {
            "n_files": len(files),
            "entrypoint": entrypoint,
            "files": [
                {
                    "path": str(file.relative_to(assets_dir)),
                    "sha256": sha256_file(file),
                    "size_bytes": int(file.stat().st_size),
                }
                for file in files
            ],
        }

    out_manifest = manifest_path if manifest_path is not None else (assets_dir / "runtime" / "manifest.json")
    payload = {
        "created_utc": utc_now_iso(),
        "assets_dir": rel_or_abs(assets_dir, repo_root),
        "sections": sections,
    }
    write_json(out_manifest, payload)
    return Phase2AssetsResult(manifest_path=out_manifest)
