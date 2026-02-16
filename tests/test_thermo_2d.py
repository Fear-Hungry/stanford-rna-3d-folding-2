from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.training.thermo_2d import compute_thermo_bpp


def test_compute_thermo_bpp_mock_backend_shapes() -> None:
    targets = pl.DataFrame(
        [
            {"target_id": "T1", "sequence": "ACGUAC", "temporal_cutoff": "2024-01-01"},
            {"target_id": "T2", "sequence": "GGAAUU", "temporal_cutoff": "2024-01-01"},
        ]
    )
    out = compute_thermo_bpp(
        targets=targets,
        backend="mock",
        rnafold_bin="RNAfold",
        linearfold_bin="linearfold",
        cache_dir=None,
        chain_separator="|",
        stage="TEST",
        location="tests/test_thermo_2d.py:test_compute_thermo_bpp_mock_backend_shapes",
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


def test_compute_thermo_bpp_cache_roundtrip(tmp_path: Path) -> None:
    targets = pl.DataFrame([{"target_id": "T1", "sequence": "ACGUACGU", "temporal_cutoff": "2024-01-01"}])
    cache_dir = tmp_path / "thermo_cache"
    first = compute_thermo_bpp(
        targets=targets,
        backend="mock",
        rnafold_bin="RNAfold",
        linearfold_bin="linearfold",
        cache_dir=cache_dir,
        chain_separator="|",
        stage="TEST",
        location="tests/test_thermo_2d.py:test_compute_thermo_bpp_cache_roundtrip:first",
    )
    second = compute_thermo_bpp(
        targets=targets,
        backend="mock",
        rnafold_bin="RNAfold",
        linearfold_bin="linearfold",
        cache_dir=cache_dir,
        chain_separator="|",
        stage="TEST",
        location="tests/test_thermo_2d.py:test_compute_thermo_bpp_cache_roundtrip:second",
    )
    assert float(first["T1"].paired_marginal.sum().item()) == float(second["T1"].paired_marginal.sum().item())


def test_compute_thermo_bpp_multichain_respects_separator() -> None:
    targets = pl.DataFrame([{"target_id": "T1", "sequence": "AC|GU", "temporal_cutoff": "2024-01-01"}])
    out = compute_thermo_bpp(
        targets=targets,
        backend="mock",
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
