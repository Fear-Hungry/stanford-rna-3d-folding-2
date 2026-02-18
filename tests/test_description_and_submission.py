from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rna3d_local.description_family import infer_description_family
from rna3d_local.errors import PipelineError
from rna3d_local.submission import check_submission, export_submission


def _write_csv(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_csv(path)


def _make_sample(path: Path) -> None:
    rows = []
    for resid, base in enumerate("AC", start=1):
        rows.append(
            {
                "ID": f"Q1_{resid}",
                "resname": base,
                "resid": resid,
                "x_1": 0.0,
                "y_1": 0.0,
                "z_1": 0.0,
                "x_2": 0.0,
                "y_2": 0.0,
                "z_2": 0.0,
            }
        )
    pl.DataFrame(rows).write_csv(path)


def test_infer_description_family_rules_outputs_prior(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    _write_csv(
        targets,
        [
            {"target_id": "Q1", "sequence": "AC", "temporal_cutoff": "2024-01-01", "description": "crystal structure of xanthine riboswitch"},
            {"target_id": "Q2", "sequence": "AC", "temporal_cutoff": "2024-01-01", "description": "unknown RNA"},
        ],
    )
    template_map = tmp_path / "template_family_map.csv"
    _write_csv(
        template_map,
        [
            {"template_uid": "ext:T1", "family_label": "riboswitch"},
            {"template_uid": "ext:T2", "family_label": "unknown"},
        ],
    )
    out = infer_description_family(
        repo_root=tmp_path,
        targets_path=targets,
        out_dir=tmp_path / "desc",
        backend="rules",
        llm_model_path=None,
        template_family_map_path=template_map,
    )
    assert out.target_family_path.exists()
    assert out.family_prior_path is not None and out.family_prior_path.exists()
    prior = pl.read_parquet(out.family_prior_path)
    assert prior.height >= 1


def test_export_and_check_submission_strict(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    _make_sample(sample)
    long = tmp_path / "predictions.parquet"
    pl.DataFrame(
        [
            {"target_id": "Q1", "model_id": 1, "resid": 1, "resname": "A", "x": 1.0, "y": 2.0, "z": 3.0},
            {"target_id": "Q1", "model_id": 1, "resid": 2, "resname": "C", "x": 1.1, "y": 2.1, "z": 3.1},
            {"target_id": "Q1", "model_id": 2, "resid": 1, "resname": "A", "x": 4.0, "y": 5.0, "z": 6.0},
            {"target_id": "Q1", "model_id": 2, "resid": 2, "resname": "C", "x": 4.1, "y": 5.1, "z": 6.1},
        ]
    ).write_parquet(long)
    submission = tmp_path / "submission.csv"
    export_submission(sample_path=sample, predictions_long_path=long, out_path=submission)
    check_submission(sample_path=sample, submission_path=submission)

    bad = pl.read_csv(submission)
    bad = bad.with_columns(pl.lit(float("nan")).alias("x_1"))
    bad_path = tmp_path / "bad_submission.csv"
    bad.write_csv(bad_path)
    with pytest.raises(PipelineError, match="coordenadas invalidas"):
        check_submission(sample_path=sample, submission_path=bad_path)

    bad2 = pl.read_csv(submission)
    bad2 = bad2.with_columns(pl.lit(2000.0).alias("x_1"))
    bad2_path = tmp_path / "bad_submission_oor.csv"
    bad2.write_csv(bad2_path)
    with pytest.raises(PipelineError, match="coordenadas invalidas"):
        check_submission(sample_path=sample, submission_path=bad2_path)
