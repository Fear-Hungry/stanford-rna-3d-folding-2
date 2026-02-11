from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.bigdata import assert_row_budget
from rna3d_local.errors import PipelineError
from rna3d_local.retrieval import retrieve_template_candidates


def test_row_budget_fail_fast() -> None:
    with pytest.raises(PipelineError) as e:
        assert_row_budget(
            stage="DATA",
            location="tests:test_row_budget_fail_fast",
            rows=101,
            max_rows_in_memory=100,
            label="synthetic_table",
        )
    msg = str(e.value)
    assert msg.startswith("[DATA]")
    assert "linhas em memoria acima do limite" in msg


def test_retrieval_rejects_invalid_chunk_size(tmp_path: Path) -> None:
    idx = tmp_path / "template_index.parquet"
    targets = tmp_path / "targets.csv"
    out = tmp_path / "retrieval.parquet"

    pl.DataFrame(
        [
            {
                "template_uid": "ext:T1",
                "template_id": "T1",
                "source": "ext",
                "sequence": "ACGU",
                "release_date": date(2020, 1, 1),
            }
        ]
    ).write_parquet(idx)
    pl.DataFrame(
        [
            {"target_id": "Q1", "sequence": "ACGU", "temporal_cutoff": "2021-01-01"},
        ]
    ).write_csv(targets)

    with pytest.raises(PipelineError) as e:
        retrieve_template_candidates(
            repo_root=tmp_path,
            template_index_path=idx,
            target_sequences_path=targets,
            out_path=out,
            top_k=1,
            kmer_size=2,
            chunk_size=0,
        )
    msg = str(e.value)
    assert msg.startswith("[RETRIEVAL]")
    assert "chunk_size deve ser > 0" in msg
