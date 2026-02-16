from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rna3d_local import wheelhouse as wh
from rna3d_local.errors import PipelineError


def test_build_wheelhouse_falls_back_to_building_universal_wheel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheels_dir = tmp_path / "wheels"
    wheels_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(wh, "PHASE2_REQUIREMENTS", ["fairscale==0.4.13"])

    def fake_run(cmd, stdout, stderr, check, timeout):  # noqa: ANN001
        if cmd[:4] == ["python", "-m", "pip", "download"]:
            return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"ERROR: no wheels\n")
        if cmd[:4] == ["python", "-m", "pip", "wheel"]:
            (wheels_dir / "fairscale-0.4.13-py3-none-any.whl").write_bytes(b"x")
            return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")
        raise AssertionError(f"unexpected cmd: {cmd}")

    monkeypatch.setattr(wh.subprocess, "run", fake_run)

    out = wh.build_wheelhouse(
        repo_root=tmp_path,
        wheels_dir=wheels_dir,
        include_project_wheel=False,
        python_version="3.12",
    )
    assert (wheels_dir / "fairscale-0.4.13-py3-none-any.whl").exists()
    assert out.manifest_path.exists()


def test_build_wheelhouse_errors_if_build_produces_non_universal_wheel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheels_dir = tmp_path / "wheels"
    wheels_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(wh, "PHASE2_REQUIREMENTS", ["fairscale==0.4.13"])

    def fake_run(cmd, stdout, stderr, check, timeout):  # noqa: ANN001
        if cmd[:4] == ["python", "-m", "pip", "download"]:
            return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"ERROR: no wheels\n")
        if cmd[:4] == ["python", "-m", "pip", "wheel"]:
            (wheels_dir / "fairscale-0.4.13-cp314-cp314-manylinux_2_17_x86_64.whl").write_bytes(b"x")
            return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")
        raise AssertionError(f"unexpected cmd: {cmd}")

    monkeypatch.setattr(wh.subprocess, "run", fake_run)

    with pytest.raises(PipelineError, match="build local nao gerou wheel universal"):
        wh.build_wheelhouse(
            repo_root=tmp_path,
            wheels_dir=wheels_dir,
            include_project_wheel=False,
            python_version="3.12",
        )

