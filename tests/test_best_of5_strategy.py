from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import torch

from rna3d_local.ensemble.select_top5 import select_top5_se3
from rna3d_local.errors import PipelineError
from rna3d_local.generative.diffusion_se3 import Se3Diffusion
from rna3d_local.generative.flow_matching_se3 import Se3FlowMatching
from rna3d_local.generative.sampler import sample_methods_for_target


def _build_ranked_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scores = {
        "A1": 1.00,
        "A2": 0.99,
        "A3": 0.98,
        "A4": 0.97,
        "A5": 0.96,
        "A6": 0.95,
        "A7": 0.94,
        "A8": 0.93,
        "B1": 0.965,
        "B2": 0.91,
        "B3": 0.90,
        "B4": 0.89,
    }
    for sample_id, score in scores.items():
        cluster_shift = 0.0 if sample_id.startswith("A") else 20.0
        sample_shift = float((ord(sample_id[-1]) - ord("0")) * 0.1)
        for resid, base in enumerate(["A", "C", "G", "U"], start=1):
            rows.append(
                {
                    "target_id": "T1",
                    "sample_id": sample_id,
                    "resid": resid,
                    "resname": base,
                    "x": float(resid * 3.0),
                    "y": float(cluster_shift + sample_shift),
                    "z": float(sample_shift),
                    "qa_score": float(score),
                    "final_score": float(score),
                }
            )
    return rows


def test_select_top5_se3_uses_cluster_medoids_after_prune(tmp_path: Path) -> None:
    ranked_path = tmp_path / "ranked.parquet"
    out_path = tmp_path / "top5.parquet"
    pl.DataFrame(_build_ranked_rows()).write_parquet(ranked_path)
    result = select_top5_se3(
        repo_root=tmp_path,
        ranked_path=ranked_path,
        out_path=out_path,
        n_models=5,
        diversity_lambda=0.35,
    )
    out = pl.read_parquet(result.predictions_path)
    chosen = sorted(set(out.get_column("sample_id").to_list()))
    assert len(chosen) == 5
    assert any(sample_id.startswith("B") for sample_id in chosen)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert int(len(manifest["stats"]["selection"])) == 1


def test_select_top5_se3_fails_on_missing_qa_or_final_score(tmp_path: Path) -> None:
    ranked_path = tmp_path / "ranked_missing.parquet"
    out_path = tmp_path / "top5.parquet"
    rows = _build_ranked_rows()
    rows[0]["qa_score"] = None
    pl.DataFrame(rows).write_parquet(ranked_path)
    with pytest.raises(PipelineError, match="qa_score/final_score ausente"):
        select_top5_se3(
            repo_root=tmp_path,
            ranked_path=ranked_path,
            out_path=out_path,
            n_models=5,
            diversity_lambda=0.35,
        )


def test_select_top5_se3_fails_on_negative_diversity_lambda(tmp_path: Path) -> None:
    ranked_path = tmp_path / "ranked.parquet"
    out_path = tmp_path / "top5.parquet"
    pl.DataFrame(_build_ranked_rows()).write_parquet(ranked_path)
    with pytest.raises(PipelineError, match="diversity_lambda invalido"):
        select_top5_se3(
            repo_root=tmp_path,
            ranked_path=ranked_path,
            out_path=out_path,
            n_models=5,
            diversity_lambda=-0.01,
        )


def test_sampler_generates_24_fast_candidates_with_finite_coords() -> None:
    torch.manual_seed(7)
    h = torch.randn(6, 8)
    x_cond = torch.randn(6, 3)
    diffusion = Se3Diffusion(hidden_dim=8, num_steps=12)
    flow = Se3FlowMatching(hidden_dim=8, num_steps=12)
    outputs = sample_methods_for_target(
        target_id="T1",
        h=h,
        x_cond=x_cond,
        method="both",
        n_samples=24,
        base_seed=123,
        diffusion=diffusion,
        flow=flow,
        stage="TEST",
        location="tests/test_best_of5_strategy.py:test_sampler_generates_24_fast_candidates_with_finite_coords",
    )
    assert len(outputs) == 24
    methods = [item[0] for item in outputs]
    assert methods.count("diffusion") == 12
    assert methods.count("flow") == 12
    for _name, _rank, coords in outputs:
        assert coords.shape == x_cond.shape
        assert torch.isfinite(coords).all()
