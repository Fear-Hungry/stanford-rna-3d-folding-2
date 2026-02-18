from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.ensemble.qa_ranker_se3 import rank_se3_ensemble
from rna3d_local.errors import PipelineError


def _write_candidates(path: Path) -> None:
    rows: list[dict[str, object]] = []
    good = {
        1: (-5.0, 0.0, 0.0),
        2: (0.2, 0.0, 0.0),
        3: (-0.2, 0.0, 0.0),
        4: (5.0, 0.0, 0.0),
    }
    bad = {
        1: (0.1, 0.0, 0.0),
        2: (4.0, 0.0, 0.0),
        3: (-4.0, 0.0, 0.0),
        4: (-0.1, 0.0, 0.0),
    }
    for sample_id, coords in [("s_good", good), ("s_bad", bad)]:
        for resid in [1, 2, 3, 4]:
            x, y, z = coords[resid]
            rows.append(
                {
                    "target_id": "T1",
                    "sample_id": sample_id,
                    "resid": resid,
                    "resname": "A",
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                }
            )
    pl.DataFrame(rows).write_parquet(path)


def _write_chemical(path: Path) -> None:
    pl.DataFrame(
        [
            {"target_id": "T1", "resid": 1, "p_open": 0.95, "p_paired": 0.05},
            {"target_id": "T1", "resid": 2, "p_open": 0.05, "p_paired": 0.95},
            {"target_id": "T1", "resid": 3, "p_open": 0.05, "p_paired": 0.95},
            {"target_id": "T1", "resid": 4, "p_open": 0.95, "p_paired": 0.05},
        ]
    ).write_parquet(path)


def test_rank_se3_ensemble_uses_chemical_exposure_consistency(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.parquet"
    chemical = tmp_path / "chemical.parquet"
    qa_config = tmp_path / "qa_config.json"
    out = tmp_path / "ranked.parquet"
    _write_candidates(candidates)
    _write_chemical(chemical)
    qa_config.write_text(
        json.dumps(
            {
                "compactness": 0.0,
                "smoothness": 0.0,
                "chem_exposure_consistency": 1.0,
            }
        ),
        encoding="utf-8",
    )
    ranked = rank_se3_ensemble(
        repo_root=tmp_path,
        candidates_path=candidates,
        out_path=out,
        qa_config_path=qa_config,
        chemical_features_path=chemical,
        diversity_lambda=0.0,
    )
    df = pl.read_parquet(ranked.ranked_path)
    first = (
        df.select("sample_id", "rank", "qa_chem_exposure_consistency")
        .unique()
        .sort("rank")
        .row(0, named=True)
    )
    assert str(first["sample_id"]) == "s_good"
    assert float(first["qa_chem_exposure_consistency"]) > 0.90


def test_rank_se3_ensemble_fails_when_chem_weight_without_chemical_features(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.parquet"
    qa_config = tmp_path / "qa_config.json"
    _write_candidates(candidates)
    qa_config.write_text(
        json.dumps(
            {
                "compactness": 0.5,
                "smoothness": 0.0,
                "chem_exposure_consistency": 0.5,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PipelineError, match="qa_config exige sinal quimico"):
        rank_se3_ensemble(
            repo_root=tmp_path,
            candidates_path=candidates,
            out_path=tmp_path / "ranked.parquet",
            qa_config_path=qa_config,
            chemical_features_path=None,
            diversity_lambda=0.0,
        )
