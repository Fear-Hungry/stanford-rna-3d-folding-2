from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rna3d_local.candidate_pool import add_labels_to_candidate_pool, build_candidate_pool_from_predictions
from rna3d_local.errors import PipelineError


def _write_prediction(path: Path, *, source: str, target_id: str, model_ids: list[int], seq: str) -> None:
    rows: list[dict] = []
    for model_id in model_ids:
        for resid, base in enumerate(seq, start=1):
            rows.append(
                {
                    "branch": source,
                    "target_id": target_id,
                    "ID": f"{target_id}_{resid}",
                    "resid": resid,
                    "resname": base,
                    "model_id": model_id,
                    "x": float(resid + model_id),
                    "y": float(model_id),
                    "z": 0.0,
                    "template_uid": f"{source}:{target_id}:m{model_id}",
                    "coverage": 1.0,
                    "similarity": 0.5 + (0.01 * float(model_id)),
                    "qa_score": 0.1 * float(model_id),
                }
            )
    pl.DataFrame(rows).write_parquet(path)


def test_build_candidate_pool_from_multiple_sources(tmp_path: Path) -> None:
    tbm = tmp_path / "tbm.parquet"
    rnp = tmp_path / "rnp.parquet"
    _write_prediction(tbm, source="tbm", target_id="T1", model_ids=[1, 2], seq="ACGU")
    _write_prediction(rnp, source="rnapro", target_id="T1", model_ids=[1, 2], seq="ACGU")

    out_pool = tmp_path / "candidate_pool.parquet"
    pool_path, manifest_path = build_candidate_pool_from_predictions(
        repo_root=tmp_path,
        prediction_entries=[("tbm", tbm), ("rnapro", rnp)],
        out_path=out_pool,
    )
    assert pool_path.exists()
    assert manifest_path.exists()
    df = pl.read_parquet(pool_path)
    assert df.height == 4
    assert {"target_id", "source", "candidate_id", "coords", "dist_off_1", "qa_score_base"} <= set(df.columns)
    assert df.get_column("resid_count").to_list() == [4, 4, 4, 4]


def test_build_candidate_pool_fails_on_duplicate_resid(tmp_path: Path) -> None:
    bad = tmp_path / "bad.parquet"
    pl.DataFrame(
        [
            {
                "branch": "tbm",
                "target_id": "T1",
                "ID": "T1_1",
                "resid": 1,
                "resname": "A",
                "model_id": 1,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "template_uid": "tbm:T1:m1",
                "coverage": 1.0,
                "similarity": 1.0,
            },
            {
                "branch": "tbm",
                "target_id": "T1",
                "ID": "T1_1_dup",
                "resid": 1,
                "resname": "A",
                "model_id": 1,
                "x": 1.0,
                "y": 0.0,
                "z": 0.0,
                "template_uid": "tbm:T1:m1",
                "coverage": 1.0,
                "similarity": 1.0,
            },
        ]
    ).write_parquet(bad)
    with pytest.raises(PipelineError):
        build_candidate_pool_from_predictions(
            repo_root=tmp_path,
            prediction_entries=[("tbm", bad)],
            out_path=tmp_path / "out.parquet",
        )


def test_add_labels_to_candidate_pool_success(tmp_path: Path) -> None:
    pool = tmp_path / "candidate_pool.parquet"
    pl.DataFrame(
        [
            {
                "target_id": "T1",
                "model_id": 1,
                "candidate_id": "tbm:T1:1",
                "source": "tbm",
                "resid_count": 2,
                "coords": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                "resids": [1, 2],
            }
        ]
    ).write_parquet(pool)

    solution = tmp_path / "solution.csv"
    pl.DataFrame(
        [
            {"ID": "T1_1", "resid": 1, "x_1": 1.0, "y_1": 0.0, "z_1": 0.0},
            {"ID": "T1_2", "resid": 2, "x_1": 2.0, "y_1": 0.0, "z_1": 0.0},
        ]
    ).write_csv(solution)

    out = tmp_path / "labeled.parquet"
    existing_manifest = tmp_path / "candidate_pool_labels_manifest.json"
    if existing_manifest.exists():
        existing_manifest.unlink()

    out_path, manifest_path = add_labels_to_candidate_pool(
        candidate_pool_path=pool,
        solution_path=solution,
        out_path=out,
        label_col="label",
        label_source_col="label_source",
        label_source_name="solution_rmsd_inv1",
        memory_budget_mb=8192,
        max_rows_in_memory=500000,
    )
    assert out_path == out
    assert manifest_path.exists()
    assert manifest_path.name == "candidate_pool_labels_manifest.json"
    labeled = pl.read_parquet(out_path)
    assert labeled.get_column("label").to_list() == [1.0]
    assert labeled.get_column("label_source").to_list() == ["solution_rmsd_inv1"]


def test_add_labels_to_candidate_pool_rejects_missing_solution_target(tmp_path: Path) -> None:
    pool = tmp_path / "candidate_pool.parquet"
    pl.DataFrame(
        [
            {
                "target_id": "T1",
                "model_id": 1,
                "candidate_id": "tbm:T1:1",
                "source": "tbm",
                "resid_count": 1,
                "coords": [[1.0, 0.0, 0.0]],
                "resids": [1],
            }
        ]
    ).write_parquet(pool)

    solution = tmp_path / "solution.csv"
    pl.DataFrame(
        [
            {"ID": "OTHER_1", "resid": 1, "x_1": 1.0, "y_1": 0.0, "z_1": 0.0}
        ]
    ).write_csv(solution)

    with pytest.raises(PipelineError):
        add_labels_to_candidate_pool(
            candidate_pool_path=pool,
            solution_path=solution,
            out_path=tmp_path / "labeled.parquet",
            label_col="label",
            label_source_col="label_source",
            label_source_name="solution_rmsd_inv1",
            memory_budget_mb=8192,
            max_rows_in_memory=500000,
        )
