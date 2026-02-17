from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from rna3d_local.ensemble.diversity import build_sample_vectors, cosine_similarity
from rna3d_local.errors import PipelineError


def _rotz(angle_rad: float) -> np.ndarray:
    c = float(math.cos(float(angle_rad)))
    s = float(math.sin(float(angle_rad)))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def test_build_sample_vectors_kabsch_is_rotation_and_translation_invariant() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.5, 0.2],
            [4.0, 1.1, 0.6],
            [6.0, 1.5, 1.0],
        ],
        dtype=np.float64,
    )
    r = _rotz(0.9)
    shift = np.array([3.0, -2.0, 1.0], dtype=np.float64)
    coords_rt = (coords @ r.T) + shift

    rows: list[dict[str, object]] = []
    for resid, xyz in enumerate(coords, start=1):
        rows.append({"sample_id": "a", "resid": resid, "x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])})
    for resid, xyz in enumerate(coords_rt, start=1):
        rows.append({"sample_id": "b", "resid": resid, "x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])})
    df = pl.DataFrame(rows)

    vecs = build_sample_vectors(df, stage="TEST", location="tests/test_diversity_rotation_invariance.py:test_build_sample_vectors_kabsch_is_rotation_and_translation_invariant")
    sim = cosine_similarity(vecs["a"], vecs["b"])
    assert sim > 0.999


def test_build_sample_vectors_fails_on_length_mismatch() -> None:
    df = pl.DataFrame(
        [
            {"sample_id": "a", "resid": 1, "x": 0.0, "y": 0.0, "z": 0.0},
            {"sample_id": "a", "resid": 2, "x": 1.0, "y": 0.0, "z": 0.0},
            {"sample_id": "b", "resid": 1, "x": 0.0, "y": 0.0, "z": 0.0},
        ]
    )
    with pytest.raises(PipelineError, match="comprimentos divergentes"):
        build_sample_vectors(df, stage="TEST", location="tests/test_diversity_rotation_invariance.py:test_build_sample_vectors_fails_on_length_mismatch")

