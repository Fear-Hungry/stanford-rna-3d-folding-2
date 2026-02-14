from __future__ import annotations

from pathlib import Path

from rna3d_local.cli_commands_data import _score_meta


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_score_meta_includes_required_fields_and_official_regime(tmp_path: Path) -> None:
    repo = tmp_path
    sample = repo / "data" / "sample.csv"
    solution = repo / "data" / "solution.csv"
    metric_py = repo / "vendor" / "tm_score_permutechains" / "metric.py"
    usalign = repo / "vendor" / "usalign" / "USalign"
    submission = repo / "runs" / "submission.csv"
    dataset_dir = repo / "data" / "derived" / "public_validation"
    official_sample = repo / "input" / "stanford-rna-3d-folding-2" / "sample_submission.csv"

    sample_text = "ID,resname,resid,x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3,x_4,y_4,z_4,x_5,y_5,z_5\nA_1,A,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
    _write(sample, sample_text)
    _write(official_sample, sample_text)
    _write(
        solution,
        "ID,resname,resid,x_1,y_1,z_1,chain,copy\nA_1,A,1,0,0,0,A,1\n",
    )
    _write(metric_py, "def score(*args, **kwargs):\n    return 0.0\n")
    _write(usalign, "dummy")
    _write(submission, sample_text)

    meta = _score_meta(
        repo=repo,
        dataset_dir=dataset_dir,
        manifest={
            "dataset_type": "public_validation",
            "sha256": {"metric.py": "metric_sha", "USalign": "usalign_sha"},
        },
        sample_path=sample,
        solution_path=solution,
        metric_py=metric_py,
        usalign_bin=usalign,
        submission_path=submission,
        location="tests/test_score_metadata.py:test_score_meta_includes_required_fields_and_official_regime",
    )
    assert meta["dataset_type"] == "public_validation"
    assert meta["n_models"] == 5
    assert meta["metric_sha256"] == "metric_sha"
    assert meta["usalign_sha256"] == "usalign_sha"
    assert meta["regime_id"] == "kaggle_official_5model"
    assert isinstance(meta["sample_columns"], list) and len(meta["sample_columns"]) > 0
