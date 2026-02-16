from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.homology_folds import build_homology_folds


def _write_train(path: Path) -> None:
    pl.DataFrame(
        [
            {"target_id": "T1", "sequence": "ACGUACGU", "description": "self-cleaving ribozyme motif", "temporal_cutoff": "2024-01-01"},
            {"target_id": "T2", "sequence": "ACGUACGA", "description": "self-cleaving ribozyme class", "temporal_cutoff": "2024-01-01"},
            {"target_id": "T3", "sequence": "UUUUCCCC", "description": "CRISPR Cas13 guide RNA", "temporal_cutoff": "2024-01-01"},
            {"target_id": "T4", "sequence": "GGGGAAAA", "description": "CRISPR Cas13 accessory RNA", "temporal_cutoff": "2024-01-01"},
        ]
    ).write_csv(path)


def _write_pdb(path: Path) -> None:
    pl.DataFrame(
        [
            {"template_id": "P1", "sequence": "ACGUACGG"},
            {"template_id": "P2", "sequence": "UUUUCCCA"},
            {"template_id": "P3", "sequence": "GGGGAAAU"},
        ]
    ).write_csv(path)


def test_build_homology_folds_mock_prevents_cluster_leakage(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    pdb = tmp_path / "pdb.csv"
    out_dir = tmp_path / "folds"
    _write_train(train)
    _write_pdb(pdb)
    out = build_homology_folds(
        repo_root=tmp_path,
        train_targets_path=train,
        pdb_sequences_path=pdb,
        out_dir=out_dir,
        backend="mock",
        identity_threshold=0.85,
        coverage_threshold=0.8,
        n_folds=2,
        chain_separator="|",
        mmseqs_bin="mmseqs",
        cdhit_bin="cd-hit-est",
        domain_labels_path=None,
        domain_column="domain_label",
        description_column="description",
        strict_domain_stratification=True,
    )
    train_folds = pl.read_parquet(out.train_folds_path)
    assert train_folds.height == 4
    leak = train_folds.group_by("cluster_id").agg(pl.col("fold_id").n_unique().alias("n_folds")).filter(pl.col("n_folds") > 1)
    assert leak.height == 0
    t1 = train_folds.filter(pl.col("target_id") == "T1").item(0, "cluster_id")
    t2 = train_folds.filter(pl.col("target_id") == "T2").item(0, "cluster_id")
    assert t1 == t2


def test_build_homology_folds_mmseqs_missing_binary_fails(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    pdb = tmp_path / "pdb.csv"
    _write_train(train)
    _write_pdb(pdb)
    with pytest.raises(PipelineError, match="mmseqs2"):
        build_homology_folds(
            repo_root=tmp_path,
            train_targets_path=train,
            pdb_sequences_path=pdb,
            out_dir=tmp_path / "folds",
            backend="mmseqs2",
            identity_threshold=0.40,
            coverage_threshold=0.80,
            n_folds=2,
            chain_separator="|",
            mmseqs_bin="/bin/nao_existe_mmseqs",
            cdhit_bin="cd-hit-est",
            domain_labels_path=None,
            domain_column="domain_label",
            description_column="description",
            strict_domain_stratification=True,
        )


def test_build_homology_folds_fails_when_nfolds_exceeds_train_count(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    pdb = tmp_path / "pdb.csv"
    _write_train(train)
    _write_pdb(pdb)
    with pytest.raises(PipelineError, match="n_folds maior"):
        build_homology_folds(
            repo_root=tmp_path,
            train_targets_path=train,
            pdb_sequences_path=pdb,
            out_dir=tmp_path / "folds",
            backend="mock",
            identity_threshold=0.40,
            coverage_threshold=0.80,
            n_folds=10,
            chain_separator="|",
            mmseqs_bin="mmseqs",
            cdhit_bin="cd-hit-est",
            domain_labels_path=None,
            domain_column="domain_label",
            description_column="description",
            strict_domain_stratification=True,
        )


def test_build_homology_folds_fails_without_domain_source_when_strict(tmp_path: Path) -> None:
    train = tmp_path / "train_no_desc.csv"
    pdb = tmp_path / "pdb.csv"
    pl.DataFrame(
        [
            {"target_id": "T1", "sequence": "ACGUACGU"},
            {"target_id": "T2", "sequence": "UUUUCCCC"},
        ]
    ).write_csv(train)
    _write_pdb(pdb)
    with pytest.raises(PipelineError, match="estratificacao por dominio"):
        build_homology_folds(
            repo_root=tmp_path,
            train_targets_path=train,
            pdb_sequences_path=pdb,
            out_dir=tmp_path / "folds",
            backend="mock",
            identity_threshold=0.40,
            coverage_threshold=0.80,
            n_folds=2,
            chain_separator="|",
            mmseqs_bin="mmseqs",
            cdhit_bin="cd-hit-est",
            domain_labels_path=None,
            domain_column="domain_label",
            description_column="description",
            strict_domain_stratification=True,
        )


def test_build_homology_folds_stratifies_domains_when_feasible(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    pdb = tmp_path / "pdb.csv"
    pl.DataFrame(
        [
            {"target_id": "T1", "sequence": "ACGUACGU", "description": "self-cleaving ribozyme motif"},
            {"target_id": "T2", "sequence": "UACGUACG", "description": "self-cleaving ribozyme class"},
            {"target_id": "T3", "sequence": "GGGGAAAA", "description": "CRISPR Cas13 guide RNA"},
            {"target_id": "T4", "sequence": "AAAAGGGG", "description": "CRISPR Cas13 accessory RNA"},
        ]
    ).write_csv(train)
    _write_pdb(pdb)
    out = build_homology_folds(
        repo_root=tmp_path,
        train_targets_path=train,
        pdb_sequences_path=pdb,
        out_dir=tmp_path / "folds",
        backend="mock",
        identity_threshold=0.95,
        coverage_threshold=0.80,
        n_folds=2,
        chain_separator="|",
        mmseqs_bin="mmseqs",
        cdhit_bin="cd-hit-est",
        domain_labels_path=None,
        domain_column="domain_label",
        description_column="description",
        strict_domain_stratification=True,
    )
    manifest = out.manifest_path.read_text(encoding="utf-8")
    assert '"domain_fold_coverage_train": {' in manifest
    train_folds = pl.read_parquet(out.train_folds_path)
    coverage = train_folds.group_by("domain_label").agg(pl.col("fold_id").n_unique().alias("n_folds"))
    assert coverage.filter(pl.col("n_folds") != 2).height == 0
