from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.rnapro import infer_rnapro
from rna3d_local.template_pt import convert_templates_to_pt_files


def _write_csv(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_csv(path)


def _write_targets(path: Path) -> None:
    _write_csv(
        path,
        [
            {"target_id": "Q1", "sequence": "ACG", "temporal_cutoff": "2022-01-01"},
            {"target_id": "Q2", "sequence": "AU", "temporal_cutoff": "2022-01-01"},
        ],
    )


def _write_template_submission(path: Path) -> None:
    _write_csv(
        path,
        [
            {"ID": "Q1_1", "resname": "A", "resid": 1, "x_1": 1.0, "y_1": 0.0, "z_1": 0.0, "x_2": 1.1, "y_2": 0.0, "z_2": 0.0},
            {"ID": "Q1_2", "resname": "C", "resid": 2, "x_1": 2.0, "y_1": 0.0, "z_1": 0.0, "x_2": 2.1, "y_2": 0.0, "z_2": 0.0},
            {"ID": "Q1_3", "resname": "G", "resid": 3, "x_1": 3.0, "y_1": 0.0, "z_1": 0.0, "x_2": 3.1, "y_2": 0.0, "z_2": 0.0},
            {"ID": "Q2_1", "resname": "A", "resid": 1, "x_1": 4.0, "y_1": 0.0, "z_1": 0.0, "x_2": 4.1, "y_2": 0.0, "z_2": 0.0},
            {"ID": "Q2_2", "resname": "U", "resid": 2, "x_1": 5.0, "y_1": 0.0, "z_1": 0.0, "x_2": 5.1, "y_2": 0.0, "z_2": 0.0},
        ],
    )


def _write_model_json(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "feature_dim": 8,
            "kmer_size": 2,
            "n_models": 2,
            "seed": 7,
            "min_coverage": 0.5,
        }
    }
    (model_dir / "model.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_convert_templates_to_pt_and_infer_precomputed(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    submission = tmp_path / "template_submission.csv"
    out_dir = tmp_path / "template_pt"
    model_dir = tmp_path / "model"
    pred_out = tmp_path / "pred.parquet"

    _write_targets(targets)
    _write_template_submission(submission)
    _write_model_json(model_dir)

    manifest = convert_templates_to_pt_files(
        repo_root=tmp_path,
        templates_submission_path=submission,
        target_sequences_path=targets,
        out_dir=out_dir,
        n_models=2,
        template_source="tbm",
    )
    assert manifest.exists()
    assert (out_dir / "Q1" / "template_features.pt").exists()
    assert (out_dir / "Q2" / "template_features.pt").exists()

    infer_rnapro(
        repo_root=tmp_path,
        model_dir=model_dir,
        target_sequences_path=targets,
        out_path=pred_out,
        n_models=2,
        min_coverage=0.5,
        use_template="ca_precomputed",
        template_features_dir=out_dir,
        template_source="tbm",
        chunk_size=2,
    )
    assert pred_out.exists()
    df = pl.read_parquet(pred_out)
    assert df.height == 10
    assert sorted(df.get_column("target_id").unique().to_list()) == ["Q1", "Q2"]
    assert sorted(df.get_column("model_id").unique().to_list()) == [1, 2]
    assert df.get_column("branch").n_unique() == 1


def test_convert_templates_to_pt_fails_when_target_missing(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    submission = tmp_path / "template_submission.csv"
    out_dir = tmp_path / "template_pt"

    _write_targets(targets)
    _write_csv(
        submission,
        [
            {"ID": "Q1_1", "resname": "A", "resid": 1, "x_1": 1.0, "y_1": 0.0, "z_1": 0.0},
            {"ID": "Q1_2", "resname": "C", "resid": 2, "x_1": 2.0, "y_1": 0.0, "z_1": 0.0},
            {"ID": "Q1_3", "resname": "G", "resid": 3, "x_1": 3.0, "y_1": 0.0, "z_1": 0.0},
        ],
    )
    with pytest.raises(PipelineError):
        convert_templates_to_pt_files(
            repo_root=tmp_path,
            templates_submission_path=submission,
            target_sequences_path=targets,
            out_dir=out_dir,
            n_models=1,
            template_source="tbm",
        )


def test_infer_precomputed_fails_when_template_file_missing(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    submission = tmp_path / "template_submission.csv"
    out_dir = tmp_path / "template_pt"
    model_dir = tmp_path / "model"
    pred_out = tmp_path / "pred.parquet"

    _write_targets(targets)
    _write_template_submission(submission)
    _write_model_json(model_dir)
    convert_templates_to_pt_files(
        repo_root=tmp_path,
        templates_submission_path=submission,
        target_sequences_path=targets,
        out_dir=out_dir,
        n_models=2,
        template_source="tbm",
    )
    (out_dir / "Q2" / "template_features.pt").unlink()
    with pytest.raises(PipelineError):
        infer_rnapro(
            repo_root=tmp_path,
            model_dir=model_dir,
            target_sequences_path=targets,
            out_path=pred_out,
            n_models=2,
            min_coverage=0.5,
            use_template="ca_precomputed",
            template_features_dir=out_dir,
            template_source="tbm",
        )


def test_infer_precomputed_rerank_skips_incomplete_models(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    model_dir = tmp_path / "model"
    template_dir = tmp_path / "template_pt"
    pred_out = tmp_path / "pred.parquet"

    seq = "ACGUACGUAC"
    _write_csv(
        targets,
        [
            {"target_id": "QX", "sequence": seq, "temporal_cutoff": "2022-01-01"},
        ],
    )
    _write_model_json(model_dir)

    n_res = len(seq)
    coords = np.zeros((3, n_res, 3), dtype=np.float32)
    mask = np.ones((3, n_res), dtype=bool)
    for ridx in range(n_res):
        coords[0, ridx, :] = [float(ridx), 0.0, 0.0]
        coords[1, ridx, :] = [float(ridx), 0.0, 0.0]
        coords[2, ridx, :] = [float(n_res - ridx), 0.0, 0.0]
    mask[2, n_res - 1] = False

    target_pt_dir = template_dir / "QX"
    target_pt_dir.mkdir(parents=True, exist_ok=True)
    with (target_pt_dir / "template_features.pt").open("wb") as f:
        pickle.dump(
            {
                "target_id": "QX",
                "sequence": seq,
                "model_ids": [1, 2, 3],
                "coordinates": coords,
                "mask": mask,
            },
            f,
            protocol=4,
        )

    infer_rnapro(
        repo_root=tmp_path,
        model_dir=model_dir,
        target_sequences_path=targets,
        out_path=pred_out,
        n_models=2,
        min_coverage=0.5,
        use_template="ca_precomputed",
        template_features_dir=template_dir,
        template_source="tbm",
        chunk_size=2,
        qa_top_pool=3,
        diversity_lambda=0.15,
    )

    df = pl.read_parquet(pred_out)
    assert df.height == (2 * n_res)
    assert sorted(df.get_column("model_id").unique().to_list()) == [1, 2]
    assert df.get_column("coverage").min() == pytest.approx(1.0)
