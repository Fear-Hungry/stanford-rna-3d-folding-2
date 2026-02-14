from __future__ import annotations

from pathlib import Path

import polars as pl

from rna3d_local.tbm_predictor import predict_tbm


def _write_csv(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_csv(path)


def test_predict_tbm_selects_next_candidate_when_first_fails_coverage(tmp_path: Path) -> None:
    retrieval = tmp_path / "retrieval.parquet"
    templates = tmp_path / "templates.parquet"
    template_index = tmp_path / "template_index.parquet"
    targets = tmp_path / "targets.csv"
    out = tmp_path / "tbm.parquet"

    pl.DataFrame(
        [
            {"target_id": "Q1", "template_uid": "ext:T1", "rank": 1, "similarity": 0.99},
            {"target_id": "Q1", "template_uid": "ext:T2", "rank": 2, "similarity": 0.98},
        ]
    ).write_parquet(retrieval)

    template_rows: list[dict] = []
    for resid in range(1, 5):
        template_rows.append({"template_uid": "ext:T1", "resid": resid, "x": float(resid), "y": 0.0, "z": 0.0})
        template_rows.append({"template_uid": "ext:T2", "resid": resid, "x": float(resid), "y": 1.0, "z": 0.0})
    pl.DataFrame(template_rows).write_parquet(templates)

    pl.DataFrame(
        [
            {"template_uid": "ext:T1", "sequence": "T"},
            {"template_uid": "ext:T2", "sequence": "AAAA"},
        ]
    ).write_parquet(template_index)

    _write_csv(targets, [{"target_id": "Q1", "sequence": "AAAA"}])

    predict_tbm(
        repo_root=tmp_path,
        retrieval_candidates_path=retrieval,
        templates_path=templates,
        target_sequences_path=targets,
        out_path=out,
        n_models=1,
        min_coverage=0.50,
        chunk_size=100,
    )

    pred = pl.read_parquet(out).sort("resid")
    assert pred.height == 4
    assert pred.get_column("template_uid").unique().to_list() == ["ext:T2"]
    assert pred.get_column("model_id").unique().to_list() == [1]


def test_predict_tbm_reranks_by_coverage_before_similarity(tmp_path: Path) -> None:
    retrieval = tmp_path / "retrieval.parquet"
    templates = tmp_path / "templates.parquet"
    template_index = tmp_path / "template_index.parquet"
    targets = tmp_path / "targets.csv"
    out = tmp_path / "tbm.parquet"

    pl.DataFrame(
        [
            {"target_id": "Q1", "template_uid": "ext:LOW_COV", "rank": 1, "similarity": 0.99},
            {"target_id": "Q1", "template_uid": "ext:HIGH_COV", "rank": 2, "similarity": 0.70},
        ]
    ).write_parquet(retrieval)

    template_rows: list[dict] = []
    for resid in range(1, 5):
        template_rows.append({"template_uid": "ext:LOW_COV", "resid": resid, "x": float(resid), "y": 0.0, "z": 0.0})
        template_rows.append({"template_uid": "ext:HIGH_COV", "resid": resid, "x": float(resid), "y": 1.0, "z": 0.0})
    pl.DataFrame(template_rows).write_parquet(templates)

    pl.DataFrame(
        [
            {"template_uid": "ext:LOW_COV", "sequence": "AAA"},
            {"template_uid": "ext:HIGH_COV", "sequence": "AAAA"},
        ]
    ).write_parquet(template_index)

    _write_csv(targets, [{"target_id": "Q1", "sequence": "AAAA"}])

    predict_tbm(
        repo_root=tmp_path,
        retrieval_candidates_path=retrieval,
        templates_path=templates,
        target_sequences_path=targets,
        out_path=out,
        n_models=1,
        min_coverage=0.50,
        rerank_pool_size=8,
        chunk_size=100,
    )

    pred = pl.read_parquet(out).sort("resid")
    assert pred.height == 4
    assert pred.get_column("template_uid").unique().to_list() == ["ext:HIGH_COV"]
    assert pred.get_column("coverage").min() > 0.99


def test_predict_tbm_can_generate_multiple_models_from_gap_variants(tmp_path: Path) -> None:
    retrieval = tmp_path / "retrieval.parquet"
    templates = tmp_path / "templates.parquet"
    template_index = tmp_path / "template_index.parquet"
    targets = tmp_path / "targets.csv"
    out = tmp_path / "tbm.parquet"

    pl.DataFrame(
        [
            {"target_id": "Q1", "template_uid": "ext:T1", "rank": 1, "similarity": 0.99},
        ]
    ).write_parquet(retrieval)

    pl.DataFrame(
        [
            {"template_uid": "ext:T1", "resid": 1, "x": 0.0, "y": 0.0, "z": 0.0},
            {"template_uid": "ext:T1", "resid": 2, "x": 1.0, "y": 0.0, "z": 0.0},
            {"template_uid": "ext:T1", "resid": 3, "x": 2.0, "y": 0.0, "z": 0.0},
            {"template_uid": "ext:T1", "resid": 4, "x": 3.0, "y": 0.0, "z": 0.0},
        ]
    ).write_parquet(templates)

    pl.DataFrame(
        [
            {"template_uid": "ext:T1", "sequence": "AAAA"},
        ]
    ).write_parquet(template_index)

    _write_csv(targets, [{"target_id": "Q1", "sequence": "AAAA"}])

    predict_tbm(
        repo_root=tmp_path,
        retrieval_candidates_path=retrieval,
        templates_path=templates,
        target_sequences_path=targets,
        out_path=out,
        n_models=3,
        min_coverage=0.80,
        rerank_pool_size=8,
        gap_open_scores=(-3.0, -5.0, -7.0),
        gap_extend_scores=(-1.0,),
        max_variants_per_template=3,
        perturbation_scale=0.0,
        chunk_size=100,
    )

    pred = pl.read_parquet(out)
    assert pred.select(pl.col("model_id").n_unique()).item() == 3
    assert pred.select(pl.col("gap_open_score").n_unique()).item() == 3
    assert pred.select(pl.col("variant_id").n_unique()).item() == 3


def test_predict_tbm_perturbation_is_deterministic(tmp_path: Path) -> None:
    retrieval = tmp_path / "retrieval.parquet"
    templates = tmp_path / "templates.parquet"
    template_index = tmp_path / "template_index.parquet"
    targets = tmp_path / "targets.csv"
    out_a = tmp_path / "tbm_a.parquet"
    out_b = tmp_path / "tbm_b.parquet"

    pl.DataFrame(
        [
            {"target_id": "Q1", "template_uid": "ext:T1", "rank": 1, "similarity": 0.99},
        ]
    ).write_parquet(retrieval)

    pl.DataFrame(
        [
            {"template_uid": "ext:T1", "resid": 1, "x": 0.0, "y": 0.0, "z": 0.0},
            {"template_uid": "ext:T1", "resid": 2, "x": 1.0, "y": 0.0, "z": 0.0},
            {"template_uid": "ext:T1", "resid": 3, "x": 2.0, "y": 0.0, "z": 0.0},
            {"template_uid": "ext:T1", "resid": 4, "x": 3.0, "y": 0.0, "z": 0.0},
        ]
    ).write_parquet(templates)

    pl.DataFrame(
        [
            {"template_uid": "ext:T1", "sequence": "AAAA"},
        ]
    ).write_parquet(template_index)

    _write_csv(targets, [{"target_id": "Q1", "sequence": "AAAA"}])

    kwargs = dict(
        repo_root=tmp_path,
        retrieval_candidates_path=retrieval,
        templates_path=templates,
        target_sequences_path=targets,
        n_models=2,
        min_coverage=0.80,
        rerank_pool_size=8,
        gap_open_scores=(-5.0,),
        gap_extend_scores=(-1.0, -2.0),
        max_variants_per_template=2,
        perturbation_scale=0.01,
        chunk_size=100,
    )
    predict_tbm(out_path=out_a, **kwargs)
    predict_tbm(out_path=out_b, **kwargs)

    a = pl.read_parquet(out_a).sort(["model_id", "resid"])
    b = pl.read_parquet(out_b).sort(["model_id", "resid"])
    assert a.equals(b)


def test_predict_tbm_filters_high_mismatch_ratio(tmp_path: Path) -> None:
    retrieval = tmp_path / "retrieval.parquet"
    templates = tmp_path / "templates.parquet"
    template_index = tmp_path / "template_index.parquet"
    targets = tmp_path / "targets.csv"
    out = tmp_path / "tbm.parquet"

    pl.DataFrame(
        [
            {"target_id": "Q1", "template_uid": "ext:BAD", "rank": 1, "similarity": 0.99},
            {"target_id": "Q1", "template_uid": "ext:GOOD", "rank": 2, "similarity": 0.98},
        ]
    ).write_parquet(retrieval)

    template_rows: list[dict] = []
    for resid in range(1, 5):
        template_rows.append({"template_uid": "ext:BAD", "resid": resid, "x": float(resid), "y": 0.0, "z": 0.0})
        template_rows.append({"template_uid": "ext:GOOD", "resid": resid, "x": float(resid), "y": 1.0, "z": 0.0})
    pl.DataFrame(template_rows).write_parquet(templates)

    pl.DataFrame(
        [
            {"template_uid": "ext:BAD", "sequence": "UUUU"},
            {"template_uid": "ext:GOOD", "sequence": "AAAA"},
        ]
    ).write_parquet(template_index)

    _write_csv(targets, [{"target_id": "Q1", "sequence": "AAAA"}])

    predict_tbm(
        repo_root=tmp_path,
        retrieval_candidates_path=retrieval,
        templates_path=templates,
        target_sequences_path=targets,
        out_path=out,
        n_models=1,
        min_coverage=0.50,
        max_mismatch_ratio=0.0,
        chunk_size=100,
    )

    pred = pl.read_parquet(out).sort("resid")
    assert pred.height == 4
    assert pred.get_column("template_uid").unique().to_list() == ["ext:GOOD"]
    assert pred.get_column("coverage").min() > 0.99
