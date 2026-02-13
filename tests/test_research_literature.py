from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.research import sync_literature


def test_sync_literature_writes_manifest_and_related_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_s2(*, topic: str, limit: int, timeout_s: int):
        return [
            {
                "paper_id": "s2:p1",
                "title": "RNA test",
                "authors": "A; B",
                "year": 2025,
                "doi": "10.1000/test",
                "url": "https://example.org/p1",
                "abstract": "x",
                "pdf_url": "https://example.org/p1.pdf",
                "license": "cc-by",
                "source": "semanticscholar",
            }
        ]

    def _fake_oa(*, topic: str, limit: int, timeout_s: int):
        return []

    def _fake_ax(*, topic: str, limit: int, timeout_s: int):
        return []

    def _fake_download(**kwargs):
        out_path = kwargs["out_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"%PDF-1.4 fake")
        return ("downloaded", "abc123")

    monkeypatch.setattr("rna3d_local.research._extract_s2_papers", _fake_s2)
    monkeypatch.setattr("rna3d_local.research._extract_openalex_papers", _fake_oa)
    monkeypatch.setattr("rna3d_local.research._extract_arxiv_papers", _fake_ax)
    monkeypatch.setattr("rna3d_local.research._download_pdf", _fake_download)

    out_dir = tmp_path / "lit"
    res = sync_literature(
        topic="rna structure",
        out_dir=out_dir,
        limit_per_source=2,
        timeout_s=5,
        download_pdfs=True,
        strict_pdf_download=True,
        max_pdf_mb=5,
        strict_sources=True,
    )

    assert res.manifest_path.exists()
    assert res.papers_path.exists()
    assert res.related_work_path.exists()

    manifest = json.loads(res.manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_papers"] == 1
    assert manifest["pdf_downloaded"] == 1

    df = pl.read_parquet(res.papers_path)
    assert df.height == 1
    assert df.get_column("pdf_status").to_list() == ["downloaded"]


def test_sync_literature_strict_pdf_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_source(*, topic: str, limit: int, timeout_s: int):
        return [
            {
                "paper_id": "s2:p1",
                "title": "RNA test",
                "authors": "A",
                "year": 2025,
                "doi": "",
                "url": "https://example.org/p1",
                "abstract": "x",
                "pdf_url": "https://example.org/p1.pdf",
                "license": "cc-by",
                "source": "semanticscholar",
            }
        ]

    def _fake_download(**kwargs):
        raise PipelineError("boom")

    monkeypatch.setattr("rna3d_local.research._extract_s2_papers", _fake_source)
    monkeypatch.setattr("rna3d_local.research._extract_openalex_papers", lambda **_: [])
    monkeypatch.setattr("rna3d_local.research._extract_arxiv_papers", lambda **_: [])
    monkeypatch.setattr("rna3d_local.research._download_pdf", _fake_download)

    with pytest.raises(PipelineError):
        sync_literature(
            topic="rna",
            out_dir=tmp_path / "lit",
            limit_per_source=1,
            timeout_s=5,
            download_pdfs=True,
            strict_pdf_download=True,
            max_pdf_mb=5,
            strict_sources=True,
        )


def test_sync_literature_allows_source_failures_when_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_source(*, topic: str, limit: int, timeout_s: int):
        raise PipelineError("[RESEARCH] [x] rate limit | impacto=1 | exemplos=s2")

    def _ok_source(*, topic: str, limit: int, timeout_s: int):
        return [
            {
                "paper_id": "arxiv:1",
                "title": "Arxiv paper",
                "authors": "A",
                "year": 2026,
                "doi": "",
                "url": "https://arxiv.org/abs/1",
                "abstract": "x",
                "pdf_url": "",
                "license": "arxiv",
                "source": "arxiv",
            }
        ]

    monkeypatch.setattr("rna3d_local.research._extract_s2_papers", _fail_source)
    monkeypatch.setattr("rna3d_local.research._extract_openalex_papers", _fail_source)
    monkeypatch.setattr("rna3d_local.research._extract_arxiv_papers", _ok_source)

    res = sync_literature(
        topic="rna",
        out_dir=tmp_path / "lit",
        limit_per_source=1,
        timeout_s=5,
        download_pdfs=False,
        strict_pdf_download=False,
        max_pdf_mb=5,
        strict_sources=False,
    )
    manifest = json.loads(res.manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_papers"] == 1
    assert len(manifest["source_failures"]) == 2
