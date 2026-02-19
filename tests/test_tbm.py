from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.tbm import predict_tbm


def _write_targets(path: Path) -> None:
    pl.DataFrame(
        [
            {
                "target_id": "T1",
                "sequence": "ACG",
                "temporal_cutoff": "2024-01-01",
            }
        ]
    ).write_parquet(path)


def _write_retrieval(path: Path) -> None:
    pl.DataFrame(
        [
            {"target_id": "T1", "template_uid": "bad", "rerank_rank": 1},
            {"target_id": "T1", "template_uid": "good1", "rerank_rank": 2},
            {"target_id": "T1", "template_uid": "good2", "rerank_rank": 3},
        ]
    ).write_parquet(path)


def _write_templates(path: Path) -> None:
    rows = [
        {"template_uid": "bad", "resid": 1, "resname": "A", "x": 0.0, "y": 0.0, "z": 0.0},
        {"template_uid": "bad", "resid": 2, "resname": "C", "x": 1.0, "y": 0.0, "z": 0.0},
    ]
    for template_uid, offset in [("good1", 10.0), ("good2", 20.0)]:
        rows.extend(
            [
                {"template_uid": template_uid, "resid": 1, "resname": "A", "x": offset + 0.0, "y": 0.0, "z": 0.0},
                {"template_uid": template_uid, "resid": 2, "resname": "C", "x": offset + 1.0, "y": 0.0, "z": 0.0},
                {"template_uid": template_uid, "resid": 3, "resname": "G", "x": offset + 2.0, "y": 0.0, "z": 0.0},
            ]
        )
    pl.DataFrame(rows).write_parquet(path)


def test_predict_tbm_skips_incomplete_templates(tmp_path: Path) -> None:
    retrieval = tmp_path / "retrieval.parquet"
    templates = tmp_path / "templates.parquet"
    targets = tmp_path / "targets.parquet"
    out = tmp_path / "tbm.parquet"
    _write_retrieval(retrieval)
    _write_templates(templates)
    _write_targets(targets)

    result = predict_tbm(
        repo_root=tmp_path,
        retrieval_path=retrieval,
        templates_path=templates,
        targets_path=targets,
        out_path=out,
        n_models=2,
    )
    pred = pl.read_parquet(result.predictions_path)
    assert pred.get_column("model_id").n_unique() == 2
    used = set(pred.get_column("template_uid").unique().to_list())
    assert used == {"good1", "good2"}


def test_predict_tbm_keeps_partial_when_valid_templates_are_insufficient(tmp_path: Path) -> None:
    retrieval = tmp_path / "retrieval.parquet"
    templates = tmp_path / "templates.parquet"
    targets = tmp_path / "targets.parquet"
    out = tmp_path / "tbm.parquet"
    _write_retrieval(retrieval)
    _write_targets(targets)
    pl.DataFrame(
        [
            {"template_uid": "bad", "resid": 1, "resname": "A", "x": 0.0, "y": 0.0, "z": 0.0},
            {"template_uid": "good1", "resid": 1, "resname": "A", "x": 1.0, "y": 0.0, "z": 0.0},
            {"template_uid": "good1", "resid": 2, "resname": "C", "x": 2.0, "y": 0.0, "z": 0.0},
            {"template_uid": "good1", "resid": 3, "resname": "G", "x": 3.0, "y": 0.0, "z": 0.0},
        ]
    ).write_parquet(templates)

    result = predict_tbm(
        repo_root=tmp_path,
        retrieval_path=retrieval,
        templates_path=templates,
        targets_path=targets,
        out_path=out,
        n_models=2,
    )
    pred = pl.read_parquet(result.predictions_path)
    assert pred.height > 0
    assert pred.get_column("model_id").n_unique() == 2
    used = pred.group_by("model_id").agg(pl.col("template_uid").first().alias("t")).sort("model_id").get_column("t").to_list()
    assert used == ["good1", "good1"]


def test_predict_tbm_fails_when_no_valid_template_is_available(tmp_path: Path) -> None:
    retrieval = tmp_path / "retrieval.parquet"
    templates = tmp_path / "templates.parquet"
    targets = tmp_path / "targets.parquet"
    out = tmp_path / "tbm.parquet"
    _write_retrieval(retrieval)
    _write_targets(targets)
    pl.DataFrame(
        [
            {"template_uid": "bad", "resid": 1, "resname": "A", "x": 0.0, "y": 0.0, "z": 0.0},
            {"template_uid": "bad", "resid": 2, "resname": "C", "x": 1.0, "y": 0.0, "z": 0.0},
        ]
    ).write_parquet(templates)

    with pytest.raises(PipelineError, match="alvos sem templates validos para TBM"):
        predict_tbm(
            repo_root=tmp_path,
            retrieval_path=retrieval,
            templates_path=templates,
            targets_path=targets,
            out_path=out,
            n_models=2,
        )


def test_predict_tbm_exports_chain_index_for_multichain_target(tmp_path: Path) -> None:
    retrieval = tmp_path / "retrieval.parquet"
    templates = tmp_path / "templates.parquet"
    targets = tmp_path / "targets.parquet"
    out = tmp_path / "tbm.parquet"
    pl.DataFrame([{"target_id": "TM", "template_uid": "good1", "rerank_rank": 1}]).write_parquet(retrieval)
    pl.DataFrame([{"target_id": "TM", "sequence": "AC|GU", "temporal_cutoff": "2024-01-01"}]).write_parquet(targets)
    pl.DataFrame(
        [
            {"template_uid": "good1", "resid": 1, "resname": "A", "x": 1.0, "y": 0.0, "z": 0.0},
            {"template_uid": "good1", "resid": 2, "resname": "C", "x": 2.0, "y": 0.0, "z": 0.0},
            {"template_uid": "good1", "resid": 3, "resname": "G", "x": 3.0, "y": 0.0, "z": 0.0},
            {"template_uid": "good1", "resid": 4, "resname": "U", "x": 4.0, "y": 0.0, "z": 0.0},
        ]
    ).write_parquet(templates)

    result = predict_tbm(
        repo_root=tmp_path,
        retrieval_path=retrieval,
        templates_path=templates,
        targets_path=targets,
        out_path=out,
        n_models=1,
    )
    pred = pl.read_parquet(result.predictions_path).sort(["target_id", "model_id", "resid"])
    assert "chain_index" in pred.columns
    assert "residue_index_1d" in pred.columns
    assert pred.get_column("chain_index").to_list() == [0, 0, 1, 1]
    assert pred.get_column("residue_index_1d").to_list() == [0, 1, 1002, 1003]
