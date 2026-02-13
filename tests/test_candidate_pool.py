from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rna3d_local.candidate_pool import build_candidate_pool_from_predictions
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
