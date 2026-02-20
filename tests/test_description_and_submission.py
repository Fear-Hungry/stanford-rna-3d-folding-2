from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import rna3d_local.submission as submission_mod
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

    bad3 = pl.read_csv(submission)
    bad3 = bad3.with_columns(pl.lit("X").alias("resname"))
    bad3_path = tmp_path / "bad_submission_fixed_value.csv"
    bad3.write_csv(bad3_path)
    with pytest.raises(PipelineError, match="valores fixos da submissao nao batem com sample"):
        check_submission(sample_path=sample, submission_path=bad3_path)


def test_export_submission_fills_dummy_for_missing_keys_non_streaming(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    _make_sample(sample)
    long = tmp_path / "predictions_partial.parquet"
    pl.DataFrame(
        [
            {"target_id": "Q1", "model_id": 1, "resid": 1, "resname": "A", "x": 1.0, "y": 2.0, "z": 3.0},
        ]
    ).write_parquet(long)
    submission = tmp_path / "submission_partial.csv"
    export_submission(sample_path=sample, predictions_long_path=long, out_path=submission)
    out = pl.read_csv(submission)
    row1 = out.filter(pl.col("ID") == "Q1_1").row(0, named=True)
    row2 = out.filter(pl.col("ID") == "Q1_2").row(0, named=True)
    assert float(row1["x_2"]) == pytest.approx(3.0)
    assert float(row2["x_1"]) == pytest.approx(6.0)
    assert float(row2["x_2"]) == pytest.approx(6.0)
    check_submission(sample_path=sample, submission_path=submission)


def test_export_submission_fills_dummy_for_missing_target_streaming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample = tmp_path / "sample.csv"
    _make_sample(sample)
    long = tmp_path / "predictions_other_target.parquet"
    pl.DataFrame(
        [
            {"target_id": "Q2", "model_id": 1, "resid": 1, "resname": "A", "x": 1.0, "y": 2.0, "z": 3.0},
            {"target_id": "Q2", "model_id": 2, "resid": 1, "resname": "A", "x": 4.0, "y": 5.0, "z": 6.0},
        ]
    ).write_parquet(long)
    monkeypatch.setenv("RNA3D_EXPORT_STREAMING", "1")
    submission = tmp_path / "submission_streaming_dummy.csv"
    export_submission(sample_path=sample, predictions_long_path=long, out_path=submission)
    out = pl.read_csv(submission)
    row1 = out.filter(pl.col("ID") == "Q1_1").row(0, named=True)
    row2 = out.filter(pl.col("ID") == "Q1_2").row(0, named=True)
    assert float(row1["x_1"]) == pytest.approx(3.0)
    assert float(row1["x_2"]) == pytest.approx(3.0)
    assert float(row2["x_1"]) == pytest.approx(6.0)
    assert float(row2["x_2"]) == pytest.approx(6.0)
    check_submission(sample_path=sample, submission_path=submission)


def test_export_submission_survives_target_level_exception_with_dummy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample = tmp_path / "sample.csv"
    _make_sample(sample)
    long = tmp_path / "predictions_ok.parquet"
    pl.DataFrame(
        [
            {"target_id": "Q1", "model_id": 1, "resid": 1, "resname": "A", "x": 1.0, "y": 2.0, "z": 3.0},
            {"target_id": "Q1", "model_id": 1, "resid": 2, "resname": "C", "x": 1.1, "y": 2.1, "z": 3.1},
            {"target_id": "Q1", "model_id": 2, "resid": 1, "resname": "A", "x": 4.0, "y": 5.0, "z": 6.0},
            {"target_id": "Q1", "model_id": 2, "resid": 2, "resname": "C", "x": 4.1, "y": 5.1, "z": 6.1},
        ]
    ).write_parquet(long)

    def _boom(**kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError(f"boom:{kwargs.get('target_id', 'unknown')}")

    monkeypatch.setattr(submission_mod, "_load_target_pred_map", _boom)
    monkeypatch.setenv("RNA3D_FAILSAFE_PER_TARGET", "1")

    submission = tmp_path / "submission_target_failsafe.csv"
    export_submission(sample_path=sample, predictions_long_path=long, out_path=submission)
    out = pl.read_csv(submission)
    row1 = out.filter(pl.col("ID") == "Q1_1").row(0, named=True)
    row2 = out.filter(pl.col("ID") == "Q1_2").row(0, named=True)
    assert float(row1["x_1"]) == pytest.approx(3.0)
    assert float(row1["x_2"]) == pytest.approx(3.0)
    assert float(row2["x_1"]) == pytest.approx(6.0)
    assert float(row2["x_2"]) == pytest.approx(6.0)
    check_submission(sample_path=sample, submission_path=submission)


def test_export_submission_dummy_wraps_high_resid_under_contract_limit(tmp_path: Path) -> None:
    sample = tmp_path / "sample_high_resid.csv"
    pl.DataFrame(
        [
            {
                "ID": "QH_4000",
                "resname": "A",
                "resid": 4000,
                "x_1": 0.0,
                "y_1": 0.0,
                "z_1": 0.0,
                "x_2": 0.0,
                "y_2": 0.0,
                "z_2": 0.0,
            }
        ]
    ).write_csv(sample)
    long = tmp_path / "predictions_empty.parquet"
    pl.DataFrame(
        [
            {"target_id": "QX", "model_id": 1, "resid": 1, "resname": "A", "x": 1.0, "y": 2.0, "z": 3.0},
        ]
    ).write_parquet(long)
    submission = tmp_path / "submission_high_resid_dummy.csv"
    export_submission(sample_path=sample, predictions_long_path=long, out_path=submission)
    out = pl.read_csv(submission)
    row = out.row(0, named=True)
    assert float(row["x_1"]) == pytest.approx(300.0)
    assert float(row["x_2"]) == pytest.approx(300.0)
    assert abs(float(row["x_1"])) <= 1000.0
    check_submission(sample_path=sample, submission_path=submission)
