from __future__ import annotations

import json
import os
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .download import download_dataset_usalign, download_metric_py_from_kernel
from .errors import raise_error
from .utils import sha256_file


DEFAULT_METRIC_KERNEL = "rhijudas/tm-score-permutechains"


@dataclass(frozen=True)
class VendorResult:
    metric_py: Path
    metric_sha256: str
    usalign_bin: Path
    usalign_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def vendor_all(*, repo_root: Path, metric_kernel: str = DEFAULT_METRIC_KERNEL) -> VendorResult:
    metric_dir = repo_root / "vendor" / "tm_score_permutechains"
    usalign_dir = repo_root / "vendor" / "usalign"
    metric_dir.mkdir(parents=True, exist_ok=True)
    usalign_dir.mkdir(parents=True, exist_ok=True)

    location = "src/rna3d_local/vendor.py:vendor_all"

    # --- metric.py ---
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src_metric = download_metric_py_from_kernel(kernel_ref=metric_kernel, out_dir=tmp)
        dst_metric = metric_dir / "metric.py"
        dst_metric.write_bytes(src_metric.read_bytes())

    metric_sha = sha256_file(dst_metric)
    (metric_dir / "SOURCE.json").write_text(
        json.dumps(
            {
                "download_utc": _utc_now(),
                "origin": {"type": "kaggle_kernel_output", "ref": metric_kernel},
                "sha256": {"metric.py": metric_sha},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # --- USalign ---
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        zip_path = download_dataset_usalign(out_dir=tmp)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                members = zf.namelist()
                # Expect a USalign binary at the root.
                candidates = [m for m in members if m.endswith("USalign") and not m.endswith("/")]
                if not candidates:
                    raise_error(
                        "VENDOR",
                        location,
                        "USalign nao encontrado no zip",
                        impact=str(len(members)),
                        examples=members[:8],
                    )
                if len(candidates) != 1:
                    raise_error(
                        "VENDOR",
                        location,
                        "multiplos candidatos USalign no zip",
                        impact=str(len(candidates)),
                        examples=candidates[:8],
                    )
                member = candidates[0]
                zf.extract(member, path=tmp)
                extracted = tmp / member
        except zipfile.BadZipFile as e:
            raise_error(
                "VENDOR",
                location,
                "zip invalido ao extrair usalign",
                impact="1",
                examples=[str(e)],
            )

        dst_bin = usalign_dir / "USalign"
        dst_bin.write_bytes(extracted.read_bytes())

    os.chmod(dst_bin, os.stat(dst_bin).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    usalign_sha = sha256_file(dst_bin)
    (usalign_dir / "SOURCE.json").write_text(
        json.dumps(
            {
                "download_utc": _utc_now(),
                "origin": {"type": "kaggle_dataset", "ref": "metric/usalign"},
                "sha256": {"USalign": usalign_sha},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # Basic sanity: file should be executable
    mode = os.stat(dst_bin).st_mode
    if not (mode & stat.S_IXUSR):
        raise_error(
            "VENDOR",
            location,
            "USalign nao esta executavel apos chmod",
            impact="1",
            examples=[str(dst_bin)],
        )

    return VendorResult(metric_py=dst_metric, metric_sha256=metric_sha, usalign_bin=dst_bin, usalign_sha256=usalign_sha)

