from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.retrieval import retrieve_template_candidates
from rna3d_local.errors import PipelineError


def _write_csv(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_csv(path)


def test_retrieval_length_weight_can_promote_length_compatible_template(tmp_path: Path) -> None:
    template_index = tmp_path / "template_index.parquet"
    targets = tmp_path / "targets.csv"
    out = tmp_path / "retrieval.parquet"

    pl.DataFrame(
        [
            {
                "template_uid": "ext:SHORT",
                "template_id": "SHORT",
                "source": "ext",
                "sequence": "AAAA",
                "release_date": "2019-01-01",
            },
            {
                "template_uid": "ext:LONG",
                "template_id": "LONG",
                "source": "ext",
                "sequence": "AAAAGGGGTT",
                "release_date": "2019-01-01",
            },
        ]
    ).write_parquet(template_index)

    _write_csv(
        targets,
        [
            {
                "target_id": "Q1",
                "sequence": "AAAAAACCCC",
                "temporal_cutoff": "2022-01-01",
            }
        ],
    )

    retrieve_template_candidates(
        repo_root=tmp_path,
        template_index_path=template_index,
        target_sequences_path=targets,
        out_path=out,
        top_k=2,
        kmer_size=2,
        length_weight=0.80,
    )

    ranked = pl.read_parquet(out).sort("rank")
    assert ranked.height == 2
    assert ranked.row(0, named=True)["template_uid"] == "ext:LONG"
    assert ranked.row(0, named=True)["length_compatibility"] > ranked.row(1, named=True)["length_compatibility"]


def test_retrieval_alignment_refinement_can_rerank_coarse_top(tmp_path: Path) -> None:
    template_index = tmp_path / "template_index.parquet"
    targets = tmp_path / "targets.csv"
    out = tmp_path / "retrieval.parquet"

    pl.DataFrame(
        [
            {
                "template_uid": "ext:BAD_LEN_ONLY",
                "template_id": "BAD_LEN_ONLY",
                "source": "ext",
                "sequence": "CCCCCC",
                "release_date": "2019-01-01",
            },
            {
                "template_uid": "ext:GOOD_ALIGN",
                "template_id": "GOOD_ALIGN",
                "source": "ext",
                "sequence": "AAAA",
                "release_date": "2019-01-01",
            },
        ]
    ).write_parquet(template_index)

    _write_csv(
        targets,
        [
            {
                "target_id": "Q1",
                "sequence": "AAAAAA",
                "temporal_cutoff": "2022-01-01",
            }
        ],
    )

    retrieve_template_candidates(
        repo_root=tmp_path,
        template_index_path=template_index,
        target_sequences_path=targets,
        out_path=out,
        top_k=2,
        kmer_size=2,
        length_weight=0.90,
        refine_pool_size=2,
        refine_alignment_weight=0.95,
        refine_open_gap_score=-5.0,
        refine_extend_gap_score=-1.0,
    )

    ranked = pl.read_parquet(out).sort("rank")
    assert ranked.height == 2
    assert ranked.row(0, named=True)["template_uid"] == "ext:GOOD_ALIGN"
    assert ranked.row(0, named=True)["alignment_similarity"] > ranked.row(1, named=True)["alignment_similarity"]


def test_retrieval_cache_hit_reuses_cached_candidates(tmp_path: Path) -> None:
    template_index = tmp_path / "template_index.parquet"
    targets = tmp_path / "targets.csv"
    cache_dir = tmp_path / "cache"
    out_first = tmp_path / "run1" / "retrieval.parquet"
    out_second = tmp_path / "run2" / "retrieval.parquet"

    pl.DataFrame(
        [
            {
                "template_uid": "ext:T1",
                "template_id": "T1",
                "source": "ext",
                "sequence": "ACGU",
                "release_date": "2019-01-01",
            }
        ]
    ).write_parquet(template_index)
    _write_csv(
        targets,
        [
            {
                "target_id": "Q1",
                "sequence": "ACGU",
                "temporal_cutoff": "2022-01-01",
            }
        ],
    )

    retrieve_template_candidates(
        repo_root=tmp_path,
        template_index_path=template_index,
        target_sequences_path=targets,
        out_path=out_first,
        top_k=1,
        kmer_size=2,
        cache_dir=cache_dir,
    )
    manifest_first = json.loads((out_first.parent / "retrieval_manifest.json").read_text(encoding="utf-8"))
    assert manifest_first["cache"]["cache_hit"] is False

    retrieve_template_candidates(
        repo_root=tmp_path,
        template_index_path=template_index,
        target_sequences_path=targets,
        out_path=out_second,
        top_k=1,
        kmer_size=2,
        cache_dir=cache_dir,
    )
    manifest_second = json.loads((out_second.parent / "retrieval_manifest.json").read_text(encoding="utf-8"))
    assert manifest_second["cache"]["cache_hit"] is True

    first_df = pl.read_parquet(out_first).sort(["target_id", "rank"])
    second_df = pl.read_parquet(out_second).sort(["target_id", "rank"])
    assert first_df.equals(second_df)


def test_retrieval_cache_inconsistent_files_fail_fast(tmp_path: Path) -> None:
    template_index = tmp_path / "template_index.parquet"
    targets = tmp_path / "targets.csv"
    cache_dir = tmp_path / "cache"
    out_first = tmp_path / "run1" / "retrieval.parquet"
    out_second = tmp_path / "run2" / "retrieval.parquet"

    pl.DataFrame(
        [
            {
                "template_uid": "ext:T1",
                "template_id": "T1",
                "source": "ext",
                "sequence": "ACGU",
                "release_date": "2019-01-01",
            }
        ]
    ).write_parquet(template_index)
    _write_csv(
        targets,
        [
            {
                "target_id": "Q1",
                "sequence": "ACGU",
                "temporal_cutoff": "2022-01-01",
            }
        ],
    )

    retrieve_template_candidates(
        repo_root=tmp_path,
        template_index_path=template_index,
        target_sequences_path=targets,
        out_path=out_first,
        top_k=1,
        kmer_size=2,
        cache_dir=cache_dir,
    )
    manifest_first = json.loads((out_first.parent / "retrieval_manifest.json").read_text(encoding="utf-8"))
    cache_meta = tmp_path / manifest_first["cache"]["cache_meta"]
    cache_meta.unlink()

    with pytest.raises(PipelineError):
        retrieve_template_candidates(
            repo_root=tmp_path,
            template_index_path=template_index,
            target_sequences_path=targets,
            out_path=out_second,
            top_k=1,
            kmer_size=2,
            cache_dir=cache_dir,
        )
