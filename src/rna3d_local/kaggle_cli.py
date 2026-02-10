from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .errors import raise_error


def _ensure_kaggle_credentials(location: str) -> None:
    # Kaggle CLI uses ~/.kaggle/kaggle.json by default.
    cfg_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    cred = cfg_dir / "kaggle.json"
    if not cred.exists():
        raise_error(
            "DOWNLOAD",
            location,
            "credenciais Kaggle nao encontradas (kaggle.json ausente)",
            impact="1",
            examples=[str(cred)],
        )


def run_kaggle(args: list[str], *, cwd: Path | None, location: str) -> None:
    """
    Run kaggle CLI with explicit failures (no silent fallback).
    """
    _ensure_kaggle_credentials(location)
    try:
        proc = subprocess.run(
            ["kaggle", *args],
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise_error(
            "DOWNLOAD",
            location,
            "kaggle CLI nao encontrado no PATH",
            impact="1",
            examples=[str(e)],
        )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        raise_error(
            "DOWNLOAD",
            location,
            "falha ao executar kaggle CLI",
            impact=str(proc.returncode),
            examples=tail,
        )

