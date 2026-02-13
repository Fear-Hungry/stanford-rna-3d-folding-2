from __future__ import annotations

import pytest

from rna3d_local.compute_backend import resolve_compute_backend
from rna3d_local.errors import PipelineError


def test_resolve_compute_backend_cpu_mode() -> None:
    backend = resolve_compute_backend(
        requested="cpu",
        precision="fp32",
        gpu_memory_budget_mb=1024,
        stage="TEST",
        location="tests/test_compute_backend.py:test_resolve_compute_backend_cpu_mode",
    )
    assert backend.backend == "cpu"
    assert backend.device == "cpu"
    assert backend.precision == "fp32"


def test_resolve_compute_backend_invalid_backend_fails() -> None:
    with pytest.raises(PipelineError):
        resolve_compute_backend(
            requested="metal",
            precision="fp32",
            gpu_memory_budget_mb=1024,
            stage="TEST",
            location="tests/test_compute_backend.py:test_resolve_compute_backend_invalid_backend_fails",
        )


def test_resolve_compute_backend_invalid_precision_fails() -> None:
    with pytest.raises(PipelineError):
        resolve_compute_backend(
            requested="cpu",
            precision="bf16",
            gpu_memory_budget_mb=1024,
            stage="TEST",
            location="tests/test_compute_backend.py:test_resolve_compute_backend_invalid_precision_fails",
        )

