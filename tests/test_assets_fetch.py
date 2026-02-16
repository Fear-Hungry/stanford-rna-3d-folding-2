from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rna3d_local.assets_fetch import DownloadSpec, _download_one
from rna3d_local.errors import PipelineError


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def test_download_one_supports_file_scheme(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    payload = b"abc123"
    src.write_bytes(payload)
    assets = tmp_path / "assets"
    spec = DownloadSpec(
        name="src.bin",
        urls=[src.as_uri()],
        dest_relpath="models/test/dest.bin",
        expected_sha256=_sha256_bytes(payload),
    )
    meta = _download_one(
        spec,
        assets_dir=assets,
        timeout_seconds=5,
        max_bytes=None,
        dry_run=False,
        stage="TEST",
        location="tests/test_assets_fetch.py:test_download_one_supports_file_scheme",
    )
    assert (assets / "models" / "test" / "dest.bin").exists()
    assert meta["status"] == "downloaded"


def test_download_one_uses_fallback_url(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    payload = b"fallback"
    src.write_bytes(payload)
    assets = tmp_path / "assets"
    spec = DownloadSpec(
        name="fallback.bin",
        urls=["file:///this/does/not/exist.bin", src.as_uri()],
        dest_relpath="models/test/fallback.bin",
        expected_sha256=_sha256_bytes(payload),
    )
    meta = _download_one(
        spec,
        assets_dir=assets,
        timeout_seconds=5,
        max_bytes=None,
        dry_run=False,
        stage="TEST",
        location="tests/test_assets_fetch.py:test_download_one_uses_fallback_url",
    )
    assert meta["status"] == "downloaded"
    assert (assets / "models" / "test" / "fallback.bin").read_bytes() == payload


def test_download_one_fails_on_sha256_mismatch(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"zzz")
    assets = tmp_path / "assets"
    spec = DownloadSpec(
        name="badsha.bin",
        urls=[src.as_uri()],
        dest_relpath="models/test/badsha.bin",
        expected_sha256="0" * 64,
    )
    with pytest.raises(PipelineError, match="sha256 do download nao bate com esperado"):
        _download_one(
            spec,
            assets_dir=assets,
            timeout_seconds=5,
            max_bytes=None,
            dry_run=False,
            stage="TEST",
            location="tests/test_assets_fetch.py:test_download_one_fails_on_sha256_mismatch",
        )

