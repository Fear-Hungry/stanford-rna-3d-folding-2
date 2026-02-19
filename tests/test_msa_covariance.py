from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from rna3d_local.errors import PipelineError
from rna3d_local.se3.sparse_graph import compute_chain_relative_features
from rna3d_local.training import msa_covariance
from rna3d_local.training.msa_covariance import compute_msa_covariance


def _encode_seq(seq: str) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    return np.array([mapping[ch] for ch in seq.strip().upper().replace("T", "U")], dtype=np.int16)


def _choose_canonical_pair(exclude_left: int, exclude_right: int) -> tuple[int, int]:
    candidates = [(0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    for left, right in candidates:
        if left != exclude_left and right != exclude_right:
            return left, right
    left = (exclude_left + 1) % 4
    right = (exclude_right + 1) % 4
    return left, right


def test_compute_msa_covariance_mmseqs_backend_shapes_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_mmseqs_chain_alignments(
        *,
        mmseqs_bin: str,
        mmseqs_db: str,
        chain_sequence: str,
        query_id: str,
        max_msa_sequences: int,
        stage: str,
        location: str,
    ) -> np.ndarray:
        row0 = _encode_seq(chain_sequence)
        row1 = row0.copy()
        if row0.size >= 2:
            left, right = _choose_canonical_pair(int(row0[0]), int(row0[-1]))
            row1[0] = np.int16(left)
            row1[-1] = np.int16(right)
        return np.stack([row0, row1], axis=0)

    monkeypatch.setattr(msa_covariance, "_run_mmseqs_chain_alignments", _fake_run_mmseqs_chain_alignments)
    targets = pl.DataFrame(
        [
            {"target_id": "T1", "sequence": "ACGUAC", "temporal_cutoff": "2024-01-01"},
            {"target_id": "T2", "sequence": "GG|AAUU", "temporal_cutoff": "2024-01-01"},
        ]
    )
    out = compute_msa_covariance(
        targets=targets,
        backend="mmseqs2",
        mmseqs_bin="mmseqs",
        mmseqs_db="/tmp/fake_db",
        cache_dir=None,
        chain_separator="|",
        max_msa_sequences=16,
        max_cov_positions=64,
        max_cov_pairs=512,
        stage="TEST",
        location="tests/test_msa_covariance.py:test_compute_msa_covariance_mmseqs_backend_shapes_with_stub",
    )
    assert set(out.keys()) == {"T1", "T2"}
    assert int(out["T1"].cov_marginal.numel()) == 6
    assert int(out["T2"].cov_marginal.numel()) == 6
    assert int(out["T2"].pair_src.numel()) == int(out["T2"].pair_dst.numel())


def test_compute_msa_covariance_mmseqs_missing_binary_fails() -> None:
    targets = pl.DataFrame([{"target_id": "T1", "sequence": "ACGU", "temporal_cutoff": "2024-01-01"}])
    with pytest.raises(PipelineError, match="mmseqs2"):
        compute_msa_covariance(
            targets=targets,
            backend="mmseqs2",
            mmseqs_bin="/bin/nao_existe_mmseqs",
            mmseqs_db="/tmp/db_nao_existe",
            cache_dir=None,
            chain_separator="|",
            max_msa_sequences=16,
            max_cov_positions=64,
            max_cov_pairs=512,
            stage="TEST",
            location="tests/test_msa_covariance.py:test_compute_msa_covariance_mmseqs_missing_binary_fails",
        )


def test_compute_chain_relative_features_applies_break_offset() -> None:
    src = torch.tensor([0, 1, 1], dtype=torch.long)
    dst = torch.tensor([1, 2, 3], dtype=torch.long)
    residue_index = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    chain_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    rel, break_mask = compute_chain_relative_features(
        src=src,
        dst=dst,
        residue_index=residue_index,
        chain_index=chain_index,
        chain_break_offset=1000,
    )
    assert float(rel[0].item()) < 0.01
    assert float(break_mask[0].item()) == 0.0
    assert float(break_mask[1].item()) == 1.0


def test_compute_msa_covariance_mmseqs_backend_parallel_consistent(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_mmseqs_chain_alignments(
        *,
        mmseqs_bin: str,
        mmseqs_db: str,
        chain_sequence: str,
        query_id: str,
        max_msa_sequences: int,
        stage: str,
        location: str,
    ) -> np.ndarray:
        row0 = _encode_seq(chain_sequence)
        row1 = row0.copy()
        if row0.size >= 2:
            left, right = _choose_canonical_pair(int(row0[0]), int(row0[-1]))
            row1[0] = np.int16(left)
            row1[-1] = np.int16(right)
        return np.stack([row0, row1], axis=0)

    monkeypatch.setattr(msa_covariance, "_run_mmseqs_chain_alignments", _fake_run_mmseqs_chain_alignments)
    targets = pl.DataFrame(
        [
            {"target_id": "T1", "sequence": "ACGUAC", "temporal_cutoff": "2024-01-01"},
            {"target_id": "T2", "sequence": "GGAAUU", "temporal_cutoff": "2024-01-01"},
            {"target_id": "T3", "sequence": "AUGCAU", "temporal_cutoff": "2024-01-01"},
        ]
    )
    serial = compute_msa_covariance(
        targets=targets,
        backend="mmseqs2",
        mmseqs_bin="mmseqs",
        mmseqs_db="/tmp/fake_db",
        cache_dir=None,
        chain_separator="|",
        max_msa_sequences=16,
        max_cov_positions=64,
        max_cov_pairs=512,
        stage="TEST",
        location="tests/test_msa_covariance.py:test_compute_msa_covariance_mmseqs_backend_parallel_consistent:serial",
        num_workers=1,
    )
    parallel = compute_msa_covariance(
        targets=targets,
        backend="mmseqs2",
        mmseqs_bin="mmseqs",
        mmseqs_db="/tmp/fake_db",
        cache_dir=None,
        chain_separator="|",
        max_msa_sequences=16,
        max_cov_positions=64,
        max_cov_pairs=512,
        stage="TEST",
        location="tests/test_msa_covariance.py:test_compute_msa_covariance_mmseqs_backend_parallel_consistent:parallel",
        num_workers=4,
    )
    assert sorted(serial.keys()) == sorted(parallel.keys())
    for target_id in serial:
        assert int(serial[target_id].pair_src.numel()) == int(parallel[target_id].pair_src.numel())


def test_compute_msa_covariance_applies_dynamic_cap_for_medium_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_depths: list[int] = []

    def _fake_run_mmseqs_chain_alignments(
        *,
        mmseqs_bin: str,
        mmseqs_db: str,
        chain_sequence: str,
        query_id: str,
        max_msa_sequences: int,
        stage: str,
        location: str,
    ) -> np.ndarray:
        base = _encode_seq(chain_sequence)
        rows: list[np.ndarray] = [base]
        for idx in range(1, int(max_msa_sequences)):
            row = base.copy()
            pos = idx % max(1, row.size)
            row[pos] = np.int16((int(row[pos]) + int((idx % 3) + 1)) % 4)
            rows.append(row)
        return np.stack(rows, axis=0)

    def _fake_covariance_pairs_from_alignment(*, aligned: np.ndarray, max_cov_positions: int, max_cov_pairs: int) -> list[tuple[int, int, float]]:
        captured_depths.append(int(aligned.shape[0]))
        return [(1, 2, 0.9)]

    monkeypatch.setattr(msa_covariance, "_run_mmseqs_chain_alignments", _fake_run_mmseqs_chain_alignments)
    monkeypatch.setattr(msa_covariance, "_covariance_pairs_from_alignment", _fake_covariance_pairs_from_alignment)
    targets = pl.DataFrame([{"target_id": "TMED", "sequence": ("A" * 500), "temporal_cutoff": "2024-01-01"}])
    out = compute_msa_covariance(
        targets=targets,
        backend="mmseqs2",
        mmseqs_bin="mmseqs",
        mmseqs_db="/tmp/fake_db",
        cache_dir=None,
        chain_separator="|",
        max_msa_sequences=96,
        max_cov_positions=64,
        max_cov_pairs=512,
        stage="TEST",
        location="tests/test_msa_covariance.py:test_compute_msa_covariance_applies_dynamic_cap_for_medium_targets",
    )
    assert "TMED" in out
    assert captured_depths == [64]


def test_compute_msa_covariance_applies_dynamic_cap_for_long_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_depths: list[int] = []

    def _fake_run_mmseqs_chain_alignments(
        *,
        mmseqs_bin: str,
        mmseqs_db: str,
        chain_sequence: str,
        query_id: str,
        max_msa_sequences: int,
        stage: str,
        location: str,
    ) -> np.ndarray:
        base = _encode_seq(chain_sequence)
        rows: list[np.ndarray] = [base]
        for idx in range(1, int(max_msa_sequences)):
            row = base.copy()
            pos = idx % max(1, row.size)
            row[pos] = np.int16((int(row[pos]) + int((idx % 3) + 1)) % 4)
            rows.append(row)
        return np.stack(rows, axis=0)

    def _fake_covariance_pairs_from_alignment(*, aligned: np.ndarray, max_cov_positions: int, max_cov_pairs: int) -> list[tuple[int, int, float]]:
        captured_depths.append(int(aligned.shape[0]))
        return [(1, 2, 0.9)]

    monkeypatch.setattr(msa_covariance, "_run_mmseqs_chain_alignments", _fake_run_mmseqs_chain_alignments)
    monkeypatch.setattr(msa_covariance, "_covariance_pairs_from_alignment", _fake_covariance_pairs_from_alignment)
    targets = pl.DataFrame([{"target_id": "TLONG", "sequence": ("A" * 700), "temporal_cutoff": "2024-01-01"}])
    out = compute_msa_covariance(
        targets=targets,
        backend="mmseqs2",
        mmseqs_bin="mmseqs",
        mmseqs_db="/tmp/fake_db",
        cache_dir=None,
        chain_separator="|",
        max_msa_sequences=96,
        max_cov_positions=64,
        max_cov_pairs=512,
        stage="TEST",
        location="tests/test_msa_covariance.py:test_compute_msa_covariance_applies_dynamic_cap_for_long_targets",
    )
    assert "TLONG" in out
    assert captured_depths == [32]
