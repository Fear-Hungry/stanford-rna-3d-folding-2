from __future__ import annotations

from pathlib import Path

import polars as pl

from rna3d_local.retrieval import retrieve_template_candidates


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
