from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rna3d_local.embedding_index import build_embedding_index
from rna3d_local.retrieval_latent import retrieve_templates_latent


def _write_csv(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_csv(path)


def _write_toy_ribonanzanet2(model_path: Path, dim: int) -> None:
    torch = pytest.importorskip("torch")

    class Toy(torch.nn.Module):
        __annotations__ = {}

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(4, dim)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (B,L)
            x = self.emb(tokens)  # (B,L,D)
            return x.mean(dim=1)  # (B,D)

    scripted = torch.jit.script(Toy(dim))
    scripted.save(str(model_path))


def test_retrieval_latent_respects_temporal_filter(tmp_path: Path) -> None:
    template_index = tmp_path / "template_index.parquet"
    pl.DataFrame(
        [
            {"template_uid": "ext:T_OLD", "template_id": "T_OLD", "source": "ext", "sequence": "AAAA", "release_date": "2020-01-01", "n_residues": 4},
            {"template_uid": "ext:T_NEW", "template_id": "T_NEW", "source": "ext", "sequence": "AAAA", "release_date": "2030-01-01", "n_residues": 4},
        ]
    ).write_parquet(template_index)
    model_path = tmp_path / "toy_ribonanzanet2.pt"
    _write_toy_ribonanzanet2(model_path, dim=32)
    emb = build_embedding_index(
        repo_root=tmp_path,
        template_index_path=template_index,
        out_dir=tmp_path / "emb",
        embedding_dim=32,
        encoder="ribonanzanet2",
        model_path=model_path,
        ann_engine="none",
    )
    targets = tmp_path / "targets.csv"
    _write_csv(targets, [{"target_id": "Q1", "sequence": "AAAA", "temporal_cutoff": "2024-01-01"}])
    out = tmp_path / "retrieval.parquet"
    res = retrieve_templates_latent(
        repo_root=tmp_path,
        template_index_path=template_index,
        template_embeddings_path=emb.embeddings_path,
        targets_path=targets,
        out_path=out,
        top_k=2,
        encoder="ribonanzanet2",
        embedding_dim=32,
        model_path=model_path,
        ann_engine="numpy_bruteforce",
        faiss_index_path=None,
        family_prior_path=None,
    )
    ranked = pl.read_parquet(res.candidates_path).sort("rank")
    assert ranked.height == 1
    assert ranked.row(0, named=True)["template_uid"] == "ext:T_OLD"


def test_retrieval_latent_uses_family_prior_weight(tmp_path: Path) -> None:
    template_index = tmp_path / "template_index.parquet"
    pl.DataFrame(
        [
            {"template_uid": "ext:T1", "template_id": "T1", "source": "ext", "sequence": "AAAA", "release_date": "2020-01-01", "n_residues": 4},
            {"template_uid": "ext:T2", "template_id": "T2", "source": "ext", "sequence": "CCCC", "release_date": "2020-01-01", "n_residues": 4},
        ]
    ).write_parquet(template_index)
    model_path = tmp_path / "toy_ribonanzanet2.pt"
    _write_toy_ribonanzanet2(model_path, dim=32)
    emb = build_embedding_index(
        repo_root=tmp_path,
        template_index_path=template_index,
        out_dir=tmp_path / "emb",
        embedding_dim=32,
        encoder="ribonanzanet2",
        model_path=model_path,
        ann_engine="none",
    )
    targets = tmp_path / "targets.csv"
    _write_csv(targets, [{"target_id": "Q1", "sequence": "AAAA", "temporal_cutoff": "2024-01-01"}])
    family_prior = tmp_path / "family_prior.parquet"
    pl.DataFrame(
        [
            {"target_id": "Q1", "template_uid": "ext:T1", "family_prior_score": 0.0},
            {"target_id": "Q1", "template_uid": "ext:T2", "family_prior_score": 1.0},
        ]
    ).write_parquet(family_prior)
    out = tmp_path / "retrieval.parquet"
    retrieve_templates_latent(
        repo_root=tmp_path,
        template_index_path=template_index,
        template_embeddings_path=emb.embeddings_path,
        targets_path=targets,
        out_path=out,
        top_k=2,
        encoder="ribonanzanet2",
        embedding_dim=32,
        model_path=model_path,
        ann_engine="numpy_bruteforce",
        faiss_index_path=None,
        family_prior_path=family_prior,
        weight_embed=0.2,
        weight_llm=0.8,
        weight_seq=0.0,
    )
    ranked = pl.read_parquet(out).sort("rank")
    assert ranked.row(0, named=True)["template_uid"] == "ext:T2"


def test_retrieval_latent_accepts_targets_without_temporal_cutoff(tmp_path: Path) -> None:
    template_index = tmp_path / "template_index.parquet"
    pl.DataFrame(
        [
            {"template_uid": "ext:T_OLD", "template_id": "T_OLD", "source": "ext", "sequence": "AAAA", "release_date": "2020-01-01", "n_residues": 4},
            {"template_uid": "ext:T_NEW", "template_id": "T_NEW", "source": "ext", "sequence": "AAAA", "release_date": "2030-01-01", "n_residues": 4},
        ]
    ).write_parquet(template_index)
    emb = build_embedding_index(
        repo_root=tmp_path,
        template_index_path=template_index,
        out_dir=tmp_path / "emb",
        embedding_dim=16,
        encoder="mock",
        model_path=None,
        ann_engine="none",
    )
    targets = tmp_path / "targets.csv"
    _write_csv(targets, [{"target_id": "Q1", "sequence": "AAAA"}])
    out = tmp_path / "retrieval.parquet"
    retrieve_templates_latent(
        repo_root=tmp_path,
        template_index_path=template_index,
        template_embeddings_path=emb.embeddings_path,
        targets_path=targets,
        out_path=out,
        top_k=2,
        encoder="mock",
        embedding_dim=16,
        model_path=None,
        ann_engine="numpy_bruteforce",
        faiss_index_path=None,
        family_prior_path=None,
    )
    ranked = pl.read_parquet(out).sort("rank")
    assert ranked.height == 2
    assert set(ranked.get_column("template_uid").to_list()) == {"ext:T_OLD", "ext:T_NEW"}
