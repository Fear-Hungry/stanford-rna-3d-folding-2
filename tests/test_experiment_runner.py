from __future__ import annotations

import json
from pathlib import Path

import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.experiments import run_experiment


def _write_recipe(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_experiment_success_creates_artifacts(tmp_path: Path) -> None:
    recipe_path = tmp_path / "recipe.json"
    _write_recipe(
        recipe_path,
        {
            "id": "TEST",
            "tag": "runner_ok",
            "variables": {},
            "steps": [
                {
                    "name": "write_artifact",
                    "argv": [
                        "python",
                        "-c",
                        "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('ok')",
                        "{run_dir}/artifact.txt",
                    ],
                }
            ],
            "artifacts": ["{run_dir}/artifact.txt"],
        },
    )
    out = run_experiment(
        repo_root=tmp_path,
        recipe_path=recipe_path,
        runs_dir=tmp_path / "runs",
        tag_override=None,
        var_overrides=[],
        dry_run=False,
    )
    assert out is not None
    assert out.run_dir.exists()
    assert (out.run_dir / "artifact.txt").exists()
    assert out.meta_path.exists()
    assert out.report_path.exists()


def test_run_experiment_fails_fast_on_nonzero_returncode(tmp_path: Path) -> None:
    recipe_path = tmp_path / "recipe.json"
    _write_recipe(
        recipe_path,
        {
            "id": "TEST",
            "tag": "runner_fail",
            "variables": {},
            "steps": [{"name": "fail", "argv": ["python", "-c", "import sys; sys.exit(3)"]}],
            "artifacts": [],
        },
    )
    with pytest.raises(PipelineError, match="comando falhou durante execucao do experimento"):
        run_experiment(
            repo_root=tmp_path,
            recipe_path=recipe_path,
            runs_dir=tmp_path / "runs",
            tag_override=None,
            var_overrides=[],
            dry_run=False,
        )
    run_dirs = [p for p in (tmp_path / "runs").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "run_report.json").exists()


def test_run_experiment_fails_on_missing_placeholder(tmp_path: Path) -> None:
    recipe_path = tmp_path / "recipe.json"
    _write_recipe(
        recipe_path,
        {
            "id": "TEST",
            "tag": "runner_missing",
            "variables": {},
            "steps": [{"name": "noop", "argv": ["echo", "{missing_key}"]}],
            "artifacts": [],
        },
    )
    with pytest.raises(PipelineError, match="placeholder ausente na recipe"):
        run_experiment(
            repo_root=tmp_path,
            recipe_path=recipe_path,
            runs_dir=tmp_path / "runs",
            tag_override=None,
            var_overrides=[],
            dry_run=True,
        )
