from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.ensemble import blend_predictions
from rna3d_local.datasets import prepare_train_labels_parquet
from rna3d_local.errors import PipelineError
from rna3d_local.export import export_submission_from_long
from rna3d_local.gating import assert_submission_allowed
from rna3d_local.rnapro import RnaProConfig, infer_rnapro, train_rnapro
from rna3d_local.retrieval import retrieve_template_candidates
from rna3d_local.tbm_predictor import predict_tbm
from rna3d_local.template_db import build_template_db


def _write_csv(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_csv(path)


def _make_sample(path: Path, *, targets: dict[str, str], n_models: int) -> None:
    rows: list[dict] = []
    for tid, seq in targets.items():
        for i, base in enumerate(seq, start=1):
            row = {"ID": f"{tid}_{i}", "resname": base, "resid": i}
            for mid in range(1, n_models + 1):
                row[f"x_{mid}"] = 0.0
                row[f"y_{mid}"] = 0.0
                row[f"z_{mid}"] = 0.0
            rows.append(row)
    _write_csv(path, rows)


def _setup_small_dataset(tmp_path: Path) -> dict[str, Path]:
    train_sequences = tmp_path / "train_sequences.csv"
    train_labels = tmp_path / "train_labels.csv"
    labels_parquet_dir = tmp_path / "labels_parquet"
    external_templates = tmp_path / "external_templates.csv"
    test_sequences = tmp_path / "test_sequences.csv"
    sample = tmp_path / "sample_submission.csv"

    _write_csv(
        train_sequences,
        [
            {"target_id": "T1", "sequence": "ACGU", "temporal_cutoff": "2020-01-01"},
            {"target_id": "T2", "sequence": "ACGA", "temporal_cutoff": "2021-01-01"},
        ],
    )
    _write_csv(
        train_labels,
        [
            {"ID": "T1_1", "resname": "A", "resid": 1, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "T1_2", "resname": "C", "resid": 2, "x_1": 1.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "T1_3", "resname": "G", "resid": 3, "x_1": 2.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "T1_4", "resname": "U", "resid": 4, "x_1": 3.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "T2_1", "resname": "A", "resid": 1, "x_1": 0.0, "y_1": 1.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "T2_2", "resname": "C", "resid": 2, "x_1": 1.0, "y_1": 1.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "T2_3", "resname": "G", "resid": 3, "x_1": 2.0, "y_1": 1.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "T2_4", "resname": "A", "resid": 4, "x_1": 3.0, "y_1": 1.0, "z_1": 0.0, "chain": "A", "copy": 1},
        ],
    )
    _write_csv(
        external_templates,
        [
            {"template_id": "E1", "sequence": "ACGU", "release_date": "2019-06-01", "resid": 1, "resname": "A", "x": 0.0, "y": 2.0, "z": 0.0, "source": "ext"},
            {"template_id": "E1", "sequence": "ACGU", "release_date": "2019-06-01", "resid": 2, "resname": "C", "x": 1.0, "y": 2.0, "z": 0.0, "source": "ext"},
            {"template_id": "E1", "sequence": "ACGU", "release_date": "2019-06-01", "resid": 3, "resname": "G", "x": 2.0, "y": 2.0, "z": 0.0, "source": "ext"},
            {"template_id": "E1", "sequence": "ACGU", "release_date": "2019-06-01", "resid": 4, "resname": "U", "x": 3.0, "y": 2.0, "z": 0.0, "source": "ext"},
        ],
    )
    _write_csv(
        test_sequences,
        [
            {"target_id": "Q1", "sequence": "ACGU", "temporal_cutoff": "2022-01-01"},
            {"target_id": "Q2", "sequence": "ACGA", "temporal_cutoff": "2022-01-01"},
        ],
    )
    prepare_train_labels_parquet(
        repo_root=tmp_path,
        train_labels_csv=train_labels,
        out_dir=labels_parquet_dir,
        rows_per_file=4,
        compression="zstd",
    )
    _make_sample(sample, targets={"Q1": "ACGU", "Q2": "ACGA"}, n_models=2)
    return {
        "train_sequences": train_sequences,
        "train_labels_parquet_dir": labels_parquet_dir,
        "external_templates": external_templates,
        "test_sequences": test_sequences,
        "sample": sample,
    }


def test_tbm_rnapro_ensemble_export_and_gating(tmp_path: Path) -> None:
    paths = _setup_small_dataset(tmp_path)
    repo = tmp_path

    template_dir = tmp_path / "template_db"
    build_template_db(
        repo_root=repo,
        train_sequences_path=paths["train_sequences"],
        train_labels_parquet_dir=paths["train_labels_parquet_dir"],
        external_templates_path=paths["external_templates"],
        out_dir=template_dir,
    )
    templates_cols = set(pl.read_parquet(template_dir / "templates.parquet").columns)
    index_cols = set(pl.read_parquet(template_dir / "template_index.parquet").columns)
    assert "sequence" not in templates_cols
    assert "sequence" in index_cols
    retrieval_path = tmp_path / "retrieval.parquet"
    retrieve_template_candidates(
        repo_root=repo,
        template_index_path=template_dir / "template_index.parquet",
        target_sequences_path=paths["test_sequences"],
        out_path=retrieval_path,
        top_k=2,
        kmer_size=2,
    )

    tbm_path = tmp_path / "tbm.parquet"
    predict_tbm(
        repo_root=repo,
        retrieval_candidates_path=retrieval_path,
        templates_path=template_dir / "templates.parquet",
        target_sequences_path=paths["test_sequences"],
        out_path=tbm_path,
        n_models=2,
        min_coverage=0.30,
    )
    assert tbm_path.exists()

    model_dir = tmp_path / "rnapro_model"
    train_rnapro(
        repo_root=repo,
        train_sequences_path=paths["train_sequences"],
        train_labels_parquet_dir=paths["train_labels_parquet_dir"],
        out_dir=model_dir,
        config=RnaProConfig(n_models=2, feature_dim=32, kmer_size=2, seed=7, min_coverage=0.30),
    )
    rnp_path = tmp_path / "rnapro.parquet"
    infer_rnapro(
        repo_root=repo,
        model_dir=model_dir,
        target_sequences_path=paths["test_sequences"],
        out_path=rnp_path,
        n_models=2,
        min_coverage=0.30,
    )
    assert rnp_path.exists()

    ens_path = tmp_path / "ensemble.parquet"
    blend_predictions(
        tbm_predictions_path=tbm_path,
        rnapro_predictions_path=rnp_path,
        out_path=ens_path,
        tbm_weight=0.7,
        rnapro_weight=0.3,
    )
    assert ens_path.exists()

    submission = tmp_path / "submission.csv"
    export_submission_from_long(
        sample_submission_path=paths["sample"],
        predictions_long_path=ens_path,
        out_submission_path=submission,
    )
    assert submission.exists()

    score_json = tmp_path / "score.json"
    score_json.write_text(json.dumps({"score": 0.52}), encoding="utf-8")
    report = tmp_path / "gating_report.json"
    assert_submission_allowed(
        sample_path=paths["sample"],
        submission_path=submission,
        report_path=report,
        is_smoke=False,
        is_partial=False,
        score_json_path=score_json,
        baseline_score=0.50,
        allow_regression=False,
    )
    assert report.exists()

    with pytest.raises(PipelineError):
        assert_submission_allowed(
            sample_path=paths["sample"],
            submission_path=submission,
            report_path=tmp_path / "blocked_report.json",
            is_smoke=True,
            is_partial=False,
            score_json_path=score_json,
            baseline_score=0.50,
            allow_regression=False,
        )

    # Strict improvement gate: tie with baseline must be blocked.
    tie_score_json = tmp_path / "score_tie.json"
    tie_score_json.write_text(json.dumps({"score": 0.50}), encoding="utf-8")
    with pytest.raises(PipelineError):
        assert_submission_allowed(
            sample_path=paths["sample"],
            submission_path=submission,
            report_path=tmp_path / "blocked_tie_report.json",
            is_smoke=False,
            is_partial=False,
            score_json_path=tie_score_json,
            baseline_score=0.50,
            allow_regression=False,
        )


def test_retrieval_fails_when_temporal_filter_removes_all(tmp_path: Path) -> None:
    paths = _setup_small_dataset(tmp_path)
    repo = tmp_path
    template_dir = tmp_path / "template_db"
    build_template_db(
        repo_root=repo,
        train_sequences_path=paths["train_sequences"],
        train_labels_parquet_dir=paths["train_labels_parquet_dir"],
        external_templates_path=paths["external_templates"],
        out_dir=template_dir,
    )
    old_targets = tmp_path / "too_old.csv"
    _write_csv(
        old_targets,
        [{"target_id": "Q0", "sequence": "ACGU", "temporal_cutoff": "1900-01-01"}],
    )
    with pytest.raises(PipelineError):
        retrieve_template_candidates(
            repo_root=repo,
            template_index_path=template_dir / "template_index.parquet",
            target_sequences_path=old_targets,
            out_path=tmp_path / "retrieval.parquet",
            top_k=2,
            kmer_size=2,
        )


def test_build_template_db_fails_on_invalid_external_templates(tmp_path: Path) -> None:
    paths = _setup_small_dataset(tmp_path)
    repo = tmp_path
    future_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    external_df = pl.read_csv(paths["external_templates"]).with_columns(pl.lit(future_date).alias("release_date"))
    external_df.write_csv(paths["external_templates"])

    with pytest.raises(PipelineError):
        build_template_db(
            repo_root=repo,
            train_sequences_path=paths["train_sequences"],
            train_labels_parquet_dir=paths["train_labels_parquet_dir"],
            external_templates_path=paths["external_templates"],
            out_dir=tmp_path / "template_db",
        )
