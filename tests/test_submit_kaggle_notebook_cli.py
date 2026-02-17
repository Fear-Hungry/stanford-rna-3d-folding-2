from __future__ import annotations

import json
from pathlib import Path

import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.submit_kaggle_notebook import submit_kaggle_notebook


class _Completed:
    def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = int(returncode)
        self.stdout = str(stdout)
        self.stderr = str(stderr)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_submit_kaggle_notebook_fails_fast_if_cli_lacks_kernel_flags(tmp_path: Path, monkeypatch) -> None:
    sample = tmp_path / "sample.csv"
    submission = tmp_path / "submission.csv"
    notebook_out = tmp_path / "kaggle_submission.csv"
    score_json = tmp_path / "score.json"

    csv = "ID,resname,resid,x_1,y_1,z_1\nT1_1,A,1,0,0,0\n"
    _write_text(sample, csv)
    _write_text(submission, csv)
    _write_text(notebook_out, csv)
    score_json.write_text(json.dumps({"score": 1.0}), encoding="utf-8")

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        if cmd[:3] == ["kaggle", "competitions", "submit"] and "-h" in cmd:
            return _Completed(returncode=0, stdout="usage: kaggle competitions submit ...\n", stderr="")
        if cmd == ["kaggle", "--version"]:
            return _Completed(returncode=0, stdout="Kaggle API 0.0.0\n", stderr="")
        return _Completed(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(PipelineError, match="sem suporte"):
        submit_kaggle_notebook(
            competition="comp",
            notebook_ref="owner/notebook",
            notebook_version="Version 1",
            notebook_file="submission.csv",
            sample_path=sample,
            submission_path=submission,
            notebook_output_path=notebook_out,
            score_json_path=score_json,
            baseline_score=0.5,
            message="msg",
            execute_submit=True,
        )


def test_submit_kaggle_notebook_accepts_kernel_flags_and_submits(tmp_path: Path, monkeypatch) -> None:
    sample = tmp_path / "sample.csv"
    submission = tmp_path / "submission.csv"
    notebook_out = tmp_path / "kaggle_submission.csv"
    score_json = tmp_path / "score.json"

    csv = "ID,resname,resid,x_1,y_1,z_1\nT1_1,A,1,0,0,0\n"
    _write_text(sample, csv)
    _write_text(submission, csv)
    _write_text(notebook_out, csv)
    score_json.write_text(json.dumps({"score": 1.0}), encoding="utf-8")

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        if cmd[:3] == ["kaggle", "competitions", "submit"] and "-h" in cmd:
            return _Completed(returncode=0, stdout="  -k, --kernel KERNEL\n  -v, --version VERSION\n", stderr="")
        if cmd[:3] == ["kaggle", "competitions", "submit"] and "-h" not in cmd:
            return _Completed(returncode=0, stdout="Submitted\n", stderr="")
        return _Completed(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    out = submit_kaggle_notebook(
        competition="comp",
        notebook_ref="owner/notebook",
        notebook_version="Version 1",
        notebook_file="submission.csv",
        sample_path=sample,
        submission_path=submission,
        notebook_output_path=notebook_out,
        score_json_path=score_json,
        baseline_score=0.5,
        message="msg",
        execute_submit=True,
    )
    assert out.report_path.exists()

