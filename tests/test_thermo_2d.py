from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.training import thermo_2d
from rna3d_local.training.thermo_2d import compute_thermo_bpp


def _stub_pairs(sequence: str) -> list[tuple[int, int, float]]:
    length = len(sequence)
    pairs: list[tuple[int, int, float]] = []
    for left in range(1, (length // 2) + 1):
        right = length - left + 1
        if left < right:
            pairs.append((left, right, 0.60))
    return pairs


def test_compute_thermo_bpp_rnafold_backend_shapes_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_rnafold_pairs(*, sequence: str, target_id: str, rnafold_bin: str, stage: str, location: str) -> list[tuple[int, int, float]]:
        return _stub_pairs(sequence)

    monkeypatch.setattr(thermo_2d, "_run_rnafold_pairs", _fake_run_rnafold_pairs)
    targets = pl.DataFrame(
        [
            {"target_id": "T1", "sequence": "ACGUAC", "temporal_cutoff": "2024-01-01"},
            {"target_id": "T2", "sequence": "GGAAUU", "temporal_cutoff": "2024-01-01"},
        ]
    )
    out = compute_thermo_bpp(
        targets=targets,
        backend="rnafold",
        rnafold_bin="RNAfold",
        linearfold_bin="linearfold",
        cache_dir=None,
        chain_separator="|",
        stage="TEST",
        location="tests/test_thermo_2d.py:test_compute_thermo_bpp_rnafold_backend_shapes_with_stub",
    )
    assert set(out.keys()) == {"T1", "T2"}
    assert int(out["T1"].paired_marginal.numel()) == 6
    assert int(out["T2"].pair_src.numel()) > 0
    assert int(out["T2"].pair_src.numel()) == int(out["T2"].pair_dst.numel())
    assert int(out["T2"].pair_src.numel()) == int(out["T2"].pair_prob.numel())


def test_compute_thermo_bpp_rnafold_missing_binary_fails() -> None:
    targets = pl.DataFrame([{"target_id": "T1", "sequence": "ACGU", "temporal_cutoff": "2024-01-01"}])
    with pytest.raises(PipelineError, match="RNAfold"):
        compute_thermo_bpp(
            targets=targets,
            backend="rnafold",
            rnafold_bin="/bin/nao_existe_rnafold",
            linearfold_bin="linearfold",
            cache_dir=None,
            chain_separator="|",
            stage="TEST",
            location="tests/test_thermo_2d.py:test_compute_thermo_bpp_rnafold_missing_binary_fails",
        )


def test_run_rnafold_pairs_timeout_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0] if args else ["RNAfold", "-p"], timeout=kwargs.get("timeout", 300))

    monkeypatch.setattr(thermo_2d.subprocess, "run", _fake_run)
    with pytest.raises(PipelineError, match="timeout no RNAfold"):
        thermo_2d._run_rnafold_pairs(
            sequence="ACGU" * 100,
            target_id="T_timeout",
            rnafold_bin="RNAfold",
            stage="TEST",
            location="tests/test_thermo_2d.py:test_run_rnafold_pairs_timeout_fails_fast",
        )


def test_compute_thermo_bpp_cache_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_rnafold_pairs(*, sequence: str, target_id: str, rnafold_bin: str, stage: str, location: str) -> list[tuple[int, int, float]]:
        return _stub_pairs(sequence)

    monkeypatch.setattr(thermo_2d, "_run_rnafold_pairs", _fake_run_rnafold_pairs)
    targets = pl.DataFrame([{"target_id": "T1", "sequence": "ACGUACGU", "temporal_cutoff": "2024-01-01"}])
    cache_dir = tmp_path / "thermo_cache"
    first = compute_thermo_bpp(
        targets=targets,
        backend="rnafold",
        rnafold_bin="RNAfold",
        linearfold_bin="linearfold",
        cache_dir=cache_dir,
        chain_separator="|",
        stage="TEST",
        location="tests/test_thermo_2d.py:test_compute_thermo_bpp_cache_roundtrip:first",
    )
    second = compute_thermo_bpp(
        targets=targets,
        backend="rnafold",
        rnafold_bin="RNAfold",
        linearfold_bin="linearfold",
        cache_dir=cache_dir,
        chain_separator="|",
        stage="TEST",
        location="tests/test_thermo_2d.py:test_compute_thermo_bpp_cache_roundtrip:second",
    )
    assert float(first["T1"].paired_marginal.sum().item()) == float(second["T1"].paired_marginal.sum().item())


def test_compute_thermo_bpp_pruning_limits_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    def _dense_pairs(sequence: str) -> list[tuple[int, int, float]]:
        length = len(sequence)
        pairs: list[tuple[int, int, float]] = []
        for i in range(1, length):
            for k in range(1, min(5, (length - i) + 1)):
                j = i + k
                pairs.append((i, j, 0.08 / float(k)))
        return pairs

    def _fake_run_rnafold_pairs(*, sequence: str, target_id: str, rnafold_bin: str, stage: str, location: str) -> list[tuple[int, int, float]]:
        return _dense_pairs(sequence)

    monkeypatch.setattr(thermo_2d, "_run_rnafold_pairs", _fake_run_rnafold_pairs)
    targets = pl.DataFrame([{"target_id": "T1", "sequence": "ACGUACGUACGU", "temporal_cutoff": "2024-01-01"}])
    full = compute_thermo_bpp(
        targets=targets,
        backend="rnafold",
        rnafold_bin="RNAfold",
        linearfold_bin="linearfold",
        cache_dir=None,
        chain_separator="|",
        stage="TEST",
        location="tests/test_thermo_2d.py:test_compute_thermo_bpp_pruning_limits_pairs:full",
        min_pair_prob=0.0,
        max_pairs_per_node=0,
    )
    pruned = compute_thermo_bpp(
        targets=targets,
        backend="rnafold",
        rnafold_bin="RNAfold",
        linearfold_bin="linearfold",
        cache_dir=None,
        chain_separator="|",
        stage="TEST",
        location="tests/test_thermo_2d.py:test_compute_thermo_bpp_pruning_limits_pairs:pruned",
        min_pair_prob=0.0,
        max_pairs_per_node=1,
    )
    assert int(pruned["T1"].pair_src.numel()) < int(full["T1"].pair_src.numel())
    assert int(pruned["T1"].pair_src.numel()) == int(pruned["T1"].pair_dst.numel())
    assert int(pruned["T1"].pair_src.numel()) == int(pruned["T1"].pair_prob.numel())


def test_compute_thermo_bpp_multichain_respects_separator(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_rnafold_pairs(*, sequence: str, target_id: str, rnafold_bin: str, stage: str, location: str) -> list[tuple[int, int, float]]:
        return _stub_pairs(sequence)

    monkeypatch.setattr(thermo_2d, "_run_rnafold_pairs", _fake_run_rnafold_pairs)
    targets = pl.DataFrame([{"target_id": "T1", "sequence": "AC|GU", "temporal_cutoff": "2024-01-01"}])
    out = compute_thermo_bpp(
        targets=targets,
        backend="rnafold",
        rnafold_bin="RNAfold",
        linearfold_bin="linearfold",
        cache_dir=None,
        chain_separator="|",
        stage="TEST",
        location="tests/test_thermo_2d.py:test_compute_thermo_bpp_multichain_respects_separator",
    )
    target = out["T1"]
    assert target.sequence == "ACGU"
    assert int(target.paired_marginal.numel()) == 4
    assert int(target.pair_src.numel()) == 4


def test_compute_thermo_bpp_rnafold_backend_parallel_consistent(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_rnafold_pairs(*, sequence: str, target_id: str, rnafold_bin: str, stage: str, location: str) -> list[tuple[int, int, float]]:
        return _stub_pairs(sequence)

    monkeypatch.setattr(thermo_2d, "_run_rnafold_pairs", _fake_run_rnafold_pairs)
    targets = pl.DataFrame(
        [
            {"target_id": "T1", "sequence": "ACGUAC", "temporal_cutoff": "2024-01-01"},
            {"target_id": "T2", "sequence": "GGAAUU", "temporal_cutoff": "2024-01-01"},
            {"target_id": "T3", "sequence": "AUGCAU", "temporal_cutoff": "2024-01-01"},
        ]
    )
    serial = compute_thermo_bpp(
        targets=targets,
        backend="rnafold",
        rnafold_bin="RNAfold",
        linearfold_bin="linearfold",
        cache_dir=None,
        chain_separator="|",
        stage="TEST",
        location="tests/test_thermo_2d.py:test_compute_thermo_bpp_rnafold_backend_parallel_consistent:serial",
        num_workers=1,
    )
    parallel = compute_thermo_bpp(
        targets=targets,
        backend="rnafold",
        rnafold_bin="RNAfold",
        linearfold_bin="linearfold",
        cache_dir=None,
        chain_separator="|",
        stage="TEST",
        location="tests/test_thermo_2d.py:test_compute_thermo_bpp_rnafold_backend_parallel_consistent:parallel",
        num_workers=4,
    )
    assert sorted(serial.keys()) == sorted(parallel.keys())
    for target_id in serial:
        assert float(serial[target_id].paired_marginal.sum().item()) == float(parallel[target_id].paired_marginal.sum().item())


def test_compute_thermo_bpp_viennarna_backend_shapes() -> None:
    pytest.importorskip("RNA")
    targets = pl.DataFrame([{"target_id": "T1", "sequence": "GGGAAACCC", "temporal_cutoff": "2024-01-01"}])
    out = compute_thermo_bpp(
        targets=targets,
        backend="viennarna",
        rnafold_bin="RNAfold",
        linearfold_bin="linearfold",
        cache_dir=None,
        chain_separator="|",
        stage="TEST",
        location="tests/test_thermo_2d.py:test_compute_thermo_bpp_viennarna_backend_shapes",
    )
    assert set(out.keys()) == {"T1"}
    assert int(out["T1"].paired_marginal.numel()) == 9
    assert int(out["T1"].pair_src.numel()) > 0


def test_compute_thermo_bpp_soft_constraints_reweight_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_rnafold_pairs(*, sequence: str, target_id: str, rnafold_bin: str, stage: str, location: str) -> list[tuple[int, int, float]]:
        return [(1, 6, 0.80), (2, 5, 0.70), (3, 4, 0.60)]

    monkeypatch.setattr(thermo_2d, "_run_rnafold_pairs", _fake_run_rnafold_pairs)
    targets = pl.DataFrame([{"target_id": "T1", "sequence": "ACGUAC", "temporal_cutoff": "2024-01-01"}])
    chemical = pl.DataFrame(
        [
            {"target_id": "T1", "resid": 1, "p_open": 0.9, "p_paired": 0.1},
            {"target_id": "T1", "resid": 2, "p_open": 0.9, "p_paired": 0.1},
            {"target_id": "T1", "resid": 3, "p_open": 0.9, "p_paired": 0.1},
            {"target_id": "T1", "resid": 4, "p_open": 0.9, "p_paired": 0.1},
            {"target_id": "T1", "resid": 5, "p_open": 0.9, "p_paired": 0.1},
            {"target_id": "T1", "resid": 6, "p_open": 0.9, "p_paired": 0.1},
        ]
    )
    unconstrained = compute_thermo_bpp(
        targets=targets,
        backend="rnafold",
        rnafold_bin="RNAfold",
        linearfold_bin="linearfold",
        cache_dir=None,
        chain_separator="|",
        stage="TEST",
        location="tests/test_thermo_2d.py:test_compute_thermo_bpp_soft_constraints_reweight_pairs:unconstrained",
    )
    constrained = compute_thermo_bpp(
        targets=targets,
        backend="rnafold",
        rnafold_bin="RNAfold",
        linearfold_bin="linearfold",
        cache_dir=None,
        chain_separator="|",
        stage="TEST",
        location="tests/test_thermo_2d.py:test_compute_thermo_bpp_soft_constraints_reweight_pairs:constrained",
        chemical_features=chemical,
        soft_constraint_strength=1.0,
    )
    unconstrained_total = float(unconstrained["T1"].pair_prob.sum().item())
    constrained_total = float(constrained["T1"].pair_prob.sum().item())
    assert constrained_total < unconstrained_total


def test_compute_thermo_bpp_soft_constraints_fail_without_chemical_features() -> None:
    targets = pl.DataFrame([{"target_id": "T1", "sequence": "ACGU", "temporal_cutoff": "2024-01-01"}])
    with pytest.raises(PipelineError, match="soft_constraint_strength > 0 exige chemical_features"):
        compute_thermo_bpp(
            targets=targets,
            backend="rnafold",
            rnafold_bin="RNAfold",
            linearfold_bin="linearfold",
            cache_dir=None,
            chain_separator="|",
            stage="TEST",
            location="tests/test_thermo_2d.py:test_compute_thermo_bpp_soft_constraints_fail_without_chemical_features",
            soft_constraint_strength=0.5,
        )
