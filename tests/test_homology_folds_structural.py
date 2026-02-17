from __future__ import annotations

import os
import subprocess
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.homology_folds import build_homology_folds


def _write_train_targets(path: Path) -> None:
    pl.DataFrame(
        [
            {"target_id": "T1", "sequence": "ACGU"},
            {"target_id": "T2", "sequence": "ACGU"},
            {"target_id": "T3", "sequence": "ACGU"},
            {"target_id": "T4", "sequence": "ACGU"},
        ]
    ).write_csv(path)


def _write_pdb_sequences(path: Path) -> None:
    pl.DataFrame([{"template_id": "P1", "sequence": "ACGU"}]).write_csv(path)


def _write_labels(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for target_id in ["T1", "T2", "T3", "T4"]:
        for resid in range(1, 5):
            rows.append({"target_id": target_id, "resid": resid, "x": float(resid), "y": 0.0, "z": 0.0})
    pl.DataFrame(rows).write_csv(path)


def test_build_homology_folds_usalign_tm_clusters_train_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train_targets = tmp_path / "train_targets.csv"
    pdb_sequences = tmp_path / "pdb_sequences.csv"
    train_labels = tmp_path / "train_labels.csv"
    out_dir = tmp_path / "out"
    usalign_bin = tmp_path / "USalign"
    _write_train_targets(train_targets)
    _write_pdb_sequences(pdb_sequences)
    _write_labels(train_labels)
    usalign_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    os.chmod(usalign_bin, 0o755)

    def _fake_run(cmd, **kwargs):  # noqa: ANN001
        cand = Path(cmd[1]).stem
        rep = Path(cmd[2]).stem
        tm = 0.10
        if {cand, rep} == {"T1", "T2"}:
            tm = 0.80
        stdout = f"header\nx x x x\nTM-score= {tm:.3f} (normalized by length)\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    out = build_homology_folds(
        repo_root=tmp_path,
        train_targets_path=train_targets,
        pdb_sequences_path=pdb_sequences,
        train_labels_path=train_labels,
        usalign_bin=str(usalign_bin),
        tm_threshold=0.50,
        usalign_timeout_seconds=5,
        out_dir=out_dir,
        backend="usalign_tm",
        identity_threshold=0.40,
        coverage_threshold=0.80,
        n_folds=2,
        chain_separator="|",
        mmseqs_bin="mmseqs",
        cdhit_bin="cd-hit-est",
        domain_labels_path=None,
        domain_column="domain_label",
        description_column="description",
        strict_domain_stratification=False,
    )
    folds = pl.read_parquet(out.train_folds_path)
    t1 = folds.filter(pl.col("target_id") == "T1").row(0, named=True)
    t2 = folds.filter(pl.col("target_id") == "T2").row(0, named=True)
    assert t1["cluster_id"] == t2["cluster_id"]
    assert int(t1["fold_id"]) == int(t2["fold_id"])


def test_build_homology_folds_usalign_tm_requires_train_labels(tmp_path: Path) -> None:
    train_targets = tmp_path / "train_targets.csv"
    pdb_sequences = tmp_path / "pdb_sequences.csv"
    out_dir = tmp_path / "out"
    _write_train_targets(train_targets)
    _write_pdb_sequences(pdb_sequences)
    with pytest.raises(PipelineError, match="train_labels_path"):
        build_homology_folds(
            repo_root=tmp_path,
            train_targets_path=train_targets,
            pdb_sequences_path=pdb_sequences,
            train_labels_path=None,
            usalign_bin="USalign",
            tm_threshold=0.50,
            usalign_timeout_seconds=5,
            out_dir=out_dir,
            backend="usalign_tm",
            identity_threshold=0.40,
            coverage_threshold=0.80,
            n_folds=2,
            chain_separator="|",
            mmseqs_bin="mmseqs",
            cdhit_bin="cd-hit-est",
            domain_labels_path=None,
            domain_column="domain_label",
            description_column="description",
            strict_domain_stratification=False,
        )

