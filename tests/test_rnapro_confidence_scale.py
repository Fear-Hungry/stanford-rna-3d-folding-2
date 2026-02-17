from __future__ import annotations

import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.runners.rnapro import _normalize_rnapro_ranking_score


def test_normalize_rnapro_ranking_score_accepts_unit_interval() -> None:
    out = _normalize_rnapro_ranking_score(0.85, stage="TEST", location="x", target_id="T1")
    assert out == 0.85


def test_normalize_rnapro_ranking_score_scales_percent() -> None:
    out = _normalize_rnapro_ranking_score(85.0, stage="TEST", location="x", target_id="T1")
    assert out == pytest.approx(0.85, rel=0, abs=1e-12)


def test_normalize_rnapro_ranking_score_rejects_out_of_range() -> None:
    with pytest.raises(PipelineError, match="fora do intervalo esperado"):
        _normalize_rnapro_ranking_score(1000.0, stage="TEST", location="x", target_id="T1")

