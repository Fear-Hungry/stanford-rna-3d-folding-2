from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .errors import raise_error
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class WheelhouseResult:
    wheels_dir: Path
    manifest_path: Path


def _ensure_disk_budget(*, dest_dir: Path, needed_bytes: int, stage: str, location: str) -> None:
    if needed_bytes <= 0:
        return
    usage = shutil.disk_usage(dest_dir)
    margin = 256 * 1024 * 1024
    if int(usage.free) < int(needed_bytes) + int(margin):
        raise_error(
            stage,
            location,
            "espaco em disco insuficiente para wheelhouse",
            impact="1",
            examples=[f"free={int(usage.free)}", f"needed={int(needed_bytes)}", str(dest_dir)],
        )


def _run(cmd: list[str], *, stage: str, location: str, timeout_seconds: int = 60 * 60) -> None:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=int(timeout_seconds))
    except FileNotFoundError as exc:
        raise_error(stage, location, "comando nao encontrado", impact="1", examples=[str(exc)])
    except subprocess.TimeoutExpired:
        raise_error(stage, location, "pip excedeu timeout", impact="1", examples=[str(cmd[:4])])
    if proc.returncode != 0:
        stderr_txt = proc.stderr.decode("utf-8", errors="replace").strip()
        stdout_txt = proc.stdout.decode("utf-8", errors="replace").strip()
        snippet = stderr_txt[:240] if stderr_txt else (stdout_txt[:240] if stdout_txt else f"returncode={proc.returncode}")
        raise_error(stage, location, "pip falhou", impact="1", examples=[snippet])


PHASE2_REQUIREMENTS = [
    # Core helpers we rely on in runners/pipeline.
    "polars==1.8.2",
    "pyarrow==18.1.0",
    # Boltz
    "boltz==2.0.3",
    # Chai-1
    "chai_lab==0.6.1",
    # Common deps (avoid torch/numpy/pandas by default; assume Kaggle base provides them).
    "antipickle==0.2.0",
    "beartype==0.18.5",
    "biopython==1.84",
    "chembl-structure-pipeline==1.2.2",
    "click==8.1.7",
    "dm-tree==0.1.8",
    "einx==0.3.0",
    "einops==0.8.0",
    "fairscale==0.4.13",
    "filelock==3.16.1",
    "gemmi==0.6.5",
    "hydra-core==1.3.2",
    "jaxtyping==0.2.38",
    "matplotlib==3.9.2",
    "mashumaro==3.14",
    "modelcif==1.2",
    "numba==0.61.0",
    "pandera==0.24.0",
    "pydantic==2.10.6",
    "pydantic-core==2.27.2",
    "pytorch-lightning==2.5.0",
    "pyyaml==6.0.2",
    # NOTE: PyPI wheels for cp312 currently go up to 2024.3.2; newer releases may not ship cp312 wheels.
    # We install phase2 wheels on Kaggle with `--no-deps` and provide a compatible rdkit wheel explicitly.
    "rdkit==2024.3.2",
    "requests==2.32.3",
    "scikit-learn==1.6.1",
    "scipy==1.13.1",
    "tmtools==0.3.0",
    "trifast==0.1.11",
    "tqdm==4.67.1",
    "typer-slim==0.12.5",
    "types-requests==2.32.0.20241016",
    "typing-extensions==4.12.2",
    "wandb==0.18.7",
]


def _run_capture(
    cmd: list[str],
    *,
    stage: str,
    location: str,
    timeout_seconds: int = 60 * 60,
) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=int(timeout_seconds))
    except FileNotFoundError as exc:
        raise_error(stage, location, "comando nao encontrado", impact="1", examples=[str(exc)])
    except subprocess.TimeoutExpired:
        raise_error(stage, location, "pip excedeu timeout", impact="1", examples=[str(cmd[:4])])
    stderr_txt = proc.stderr.decode("utf-8", errors="replace").strip()
    stdout_txt = proc.stdout.decode("utf-8", errors="replace").strip()
    snippet = stderr_txt[:240] if stderr_txt else (stdout_txt[:240] if stdout_txt else f"returncode={proc.returncode}")
    return int(proc.returncode), snippet


def _is_universal_wheel(path: Path) -> bool:
    name = path.name
    return name.endswith("-py3-none-any.whl") or name.endswith("-py2.py3-none-any.whl")


def _download_or_build_wheel(
    req: str,
    *,
    wheels_dir: Path,
    platform: str,
    python_version: str,
    abi: str,
    stage: str,
    location: str,
    timeout_seconds: int,
) -> dict[str, object]:
    download_cmd = [
        "python",
        "-m",
        "pip",
        "download",
        "--only-binary",
        ":all:",
        "--no-deps",
        "--dest",
        str(wheels_dir),
        "--platform",
        str(platform),
        "--implementation",
        "cp",
        "--python-version",
        str(python_version),
        "--abi",
        str(abi),
        str(req),
    ]
    rc, snippet = _run_capture(download_cmd, stage=stage, location=location, timeout_seconds=int(timeout_seconds))
    if rc == 0:
        return {"req": req, "mode": "download_wheel", "note": None}

    # Fallback: build a universal wheel from sdist (pure-python only).
    before = {p.name for p in wheels_dir.glob("*.whl")}
    build_cmd = ["python", "-m", "pip", "wheel", "--no-deps", str(req), "-w", str(wheels_dir)]
    rc2, snippet2 = _run_capture(build_cmd, stage=stage, location=location, timeout_seconds=int(timeout_seconds))
    after = {p.name for p in wheels_dir.glob("*.whl")}
    created = sorted(after - before)
    if rc2 != 0:
        raise_error(
            stage,
            location,
            "pip download falhou e pip wheel falhou (sem wheel para o alvo?)",
            impact="1",
            examples=[f"req={req}", f"download={snippet}", f"wheel={snippet2}"],
        )

    # Ensure produced wheel is universal; otherwise it won't work in Kaggle (py312).
    candidates = [wheels_dir / name for name in created] if created else []
    universal = [p for p in candidates if p.exists() and _is_universal_wheel(p)]
    if not universal:
        raise_error(
            stage,
            location,
            "pacote sem wheel no PyPI e build local nao gerou wheel universal",
            impact=str(len(created) or 1),
            examples=[f"req={req}", f"download={snippet}", f"created={created[:4]}"],
        )

    return {"req": req, "mode": "built_universal_wheel", "note": f"download_failed={snippet}"}


def build_wheelhouse(
    *,
    repo_root: Path,
    wheels_dir: Path,
    python_version: str = "3.12",
    platform: str = "manylinux2014_x86_64",
    profile: str = "phase2",
    include_project_wheel: bool = True,
    timeout_seconds: int = 60 * 60,
    manifest_path: Path | None = None,
) -> WheelhouseResult:
    stage = "WHEELHOUSE"
    location = "src/rna3d_local/wheelhouse.py:build_wheelhouse"
    wheels_dir = wheels_dir.resolve()
    wheels_dir.mkdir(parents=True, exist_ok=True)

    if profile != "phase2":
        raise_error(stage, location, "profile invalido", impact="1", examples=[profile])

    requirements = list(PHASE2_REQUIREMENTS)

    # Preflight disk: rough budget (requirements vary); reserve at least 2GB.
    _ensure_disk_budget(dest_dir=wheels_dir, needed_bytes=2 * 1024 * 1024 * 1024, stage=stage, location=location)

    # Build project wheel (pure python) so Kaggle can install our CLI offline if desired.
    if include_project_wheel:
        _run(
            [
                "python",
                "-m",
                "pip",
                "wheel",
                ".",
                "-w",
                str(wheels_dir),
                "--no-deps",
            ],
            stage=stage,
            location=location,
            timeout_seconds=int(timeout_seconds),
        )

    # Download pinned wheels targeting Kaggle python (typically 3.12).
    pyver = str(python_version).replace(".", "")
    if pyver not in {"312", "311", "310"}:
        raise_error(stage, location, "python_version nao suportado para wheelhouse", impact="1", examples=[python_version])
    abi = f"cp{pyver}"
    audit: list[dict[str, object]] = []
    for req in requirements:
        audit.append(
            _download_or_build_wheel(
                str(req),
                wheels_dir=wheels_dir,
                platform=str(platform),
                python_version=str(python_version),
                abi=str(abi),
                stage=stage,
                location=location,
                timeout_seconds=int(timeout_seconds),
            )
        )

    files = sorted([p for p in wheels_dir.glob("*.whl")], key=lambda p: p.name)
    if not files:
        raise_error(stage, location, "wheelhouse vazio apos download", impact="1", examples=[str(wheels_dir)])

    out_manifest = manifest_path if manifest_path is not None else (wheels_dir.parent / "runtime" / "wheelhouse_manifest.json")
    payload = {
        "created_utc": utc_now_iso(),
        "repo_root": rel_or_abs(repo_root, repo_root),
        "wheels_dir": rel_or_abs(wheels_dir, repo_root),
        "profile": profile,
        "python_version": python_version,
        "platform": platform,
        "include_project_wheel": bool(include_project_wheel),
        "requirements": requirements,
        "audit": audit,
        "wheels": [
            {"name": p.name, "size_bytes": int(p.stat().st_size), "sha256": sha256_file(p)} for p in files
        ],
    }
    write_json(out_manifest, payload)
    return WheelhouseResult(wheels_dir=wheels_dir, manifest_path=out_manifest)
