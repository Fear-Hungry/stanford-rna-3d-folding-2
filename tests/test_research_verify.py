from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.research import run_experiment, verify_run


def _build_run(tmp_path: Path, *, status: str = "success") -> Path:
    config = {
        "experiment_name": "smoke_verify",
        "solver": {
            "type": "command_json",
            "timeout_s": 30,
            "command": [
                sys.executable,
                "-c",
                f"import json; print(json.dumps(dict(instance_id='inst', solver_status='{status}', objective=0.42, feasible=True, gap=0.0)))",
            ],
        },
        "seeds": [123],
        "required_outputs": [],
        "repro_command": [sys.executable, "-c", "print('ok')"],
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    res = run_experiment(
        repo_root=tmp_path,
        config_path=cfg_path,
        run_id="run_verify",
        out_base_dir=tmp_path / "runs" / "research" / "experiments",
        allow_existing_run_dir=False,
    )
    return res.run_dir


def test_verify_run_accepts_valid_artifacts(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    out = verify_run(repo_root=tmp_path, run_dir=run_dir, allowed_statuses=("success", "optimal"))
    assert out.accepted is True
    verify = json.loads(out.verify_path.read_text(encoding="utf-8"))
    assert verify["accepted"] is True


def test_verify_run_fails_on_bad_status(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    results_path = run_dir / "results.parquet"
    df = pl.read_parquet(results_path).with_columns(pl.lit("bad_status").alias("solver_status"))
    df.write_parquet(results_path)

    with pytest.raises(PipelineError):
        verify_run(repo_root=tmp_path, run_dir=run_dir, allowed_statuses=("success",))

    verify_path = run_dir / "verify.json"
    assert verify_path.exists()
    verify = json.loads(verify_path.read_text(encoding="utf-8"))
    assert verify["accepted"] is False


def test_verify_run_enforces_kaggle_gate_score_improvement(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    submission = tmp_path / "submission.csv"
    score_json = tmp_path / "score.json"
    sample.write_text(
        "ID,resname,resid,x_1,y_1,z_1\n"
        "T1_1,A,1,0,0,0\n",
        encoding="utf-8",
    )
    submission.write_text(
        "ID,resname,resid,x_1,y_1,z_1\n"
        "T1_1,A,1,0.1,0.2,0.3\n",
        encoding="utf-8",
    )
    score_json.write_text(json.dumps({"score": 0.2000}), encoding="utf-8")

    config = {
        "experiment_name": "kaggle_gate_fail_score",
        "solver": {
            "type": "command_json",
            "timeout_s": 30,
            "command": [
                sys.executable,
                "-c",
                "import json; print(json.dumps(dict(instance_id='inst', solver_status='success', objective=0.2, feasible=True, gap=0.0)))",
            ],
        },
        "seeds": [123],
        "required_outputs": [],
        "repro_command": [sys.executable, "-c", "print('ok')"],
        "kaggle_gate": {
            "sample_submission": str(sample),
            "submission": str(submission),
            "score_json": str(score_json),
            "baseline_score": 0.2000,
            "min_improvement": 0.0010,
            "max_submission_mb": 1.0,
        },
    }
    cfg_path = tmp_path / "cfg_kaggle_fail.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    run = run_experiment(
        repo_root=tmp_path,
        config_path=cfg_path,
        run_id="run_kaggle_gate_fail",
        out_base_dir=tmp_path / "runs" / "research" / "experiments",
        allow_existing_run_dir=False,
    )

    with pytest.raises(PipelineError):
        verify_run(repo_root=tmp_path, run_dir=run.run_dir, allowed_statuses=("success",))

    verify = json.loads((run.run_dir / "verify.json").read_text(encoding="utf-8"))
    assert verify["accepted"] is False
    assert verify["kaggle_gate_pass"] is False


def test_verify_run_enforces_kaggle_gate_and_accepts_on_improvement(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    submission = tmp_path / "submission.csv"
    score_json = tmp_path / "score.json"
    sample.write_text(
        "ID,resname,resid,x_1,y_1,z_1\n"
        "T1_1,A,1,0,0,0\n",
        encoding="utf-8",
    )
    submission.write_text(
        "ID,resname,resid,x_1,y_1,z_1\n"
        "T1_1,A,1,0.1,0.2,0.3\n",
        encoding="utf-8",
    )
    score_json.write_text(json.dumps({"score": 0.2055}), encoding="utf-8")

    config = {
        "experiment_name": "kaggle_gate_ok",
        "solver": {
            "type": "command_json",
            "timeout_s": 30,
            "command": [
                sys.executable,
                "-c",
                "import json; print(json.dumps(dict(instance_id='inst', solver_status='success', objective=0.2055, feasible=True, gap=0.0)))",
            ],
        },
        "seeds": [123],
        "required_outputs": [],
        "repro_command": [sys.executable, "-c", "print('ok')"],
        "kaggle_gate": {
            "sample_submission": str(sample),
            "submission": str(submission),
            "score_json": str(score_json),
            "baseline_score": 0.2000,
            "min_improvement": 0.0010,
            "max_submission_mb": 1.0,
        },
    }
    cfg_path = tmp_path / "cfg_kaggle_ok.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    run = run_experiment(
        repo_root=tmp_path,
        config_path=cfg_path,
        run_id="run_kaggle_gate_ok",
        out_base_dir=tmp_path / "runs" / "research" / "experiments",
        allow_existing_run_dir=False,
    )

    out = verify_run(repo_root=tmp_path, run_dir=run.run_dir, allowed_statuses=("success",))
    assert out.accepted is True
    verify = json.loads((run.run_dir / "verify.json").read_text(encoding="utf-8"))
    assert verify["accepted"] is True
    assert verify["kaggle_gate_pass"] is True
