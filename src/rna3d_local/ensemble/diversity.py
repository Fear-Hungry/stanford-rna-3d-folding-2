from __future__ import annotations

import math

import numpy as np
import polars as pl


def _vector_for_sample(sample_df: pl.DataFrame) -> np.ndarray:
    coords = sample_df.select("x", "y", "z").to_numpy().astype(np.float64)
    coords = coords - coords.mean(axis=0, keepdims=True)
    vec = coords.reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.zeros_like(vec)
    return vec / norm


def build_sample_vectors(target_df: pl.DataFrame) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for sample_id, part in target_df.group_by("sample_id", maintain_order=True):
        key = str(sample_id[0]) if isinstance(sample_id, tuple) else str(sample_id)
        vectors[key] = _vector_for_sample(part.sort("resid"))
    return vectors


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        min_len = min(int(a.shape[0]), int(b.shape[0]))
        a = a[:min_len]
        b = b[:min_len]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def average_similarity(sample_id: str, vectors: dict[str, np.ndarray]) -> float:
    sims: list[float] = []
    anchor = vectors[sample_id]
    for other_id, other in vectors.items():
        if other_id == sample_id:
            continue
        sims.append(max(0.0, cosine_similarity(anchor, other)))
    if not sims:
        return 0.0
    return float(sum(sims) / len(sims))


def greedy_diverse_selection(
    *,
    sample_scores: list[tuple[str, float]],
    vectors: dict[str, np.ndarray],
    n_select: int,
    lambda_diversity: float,
) -> list[str]:
    selected: list[str] = []
    available = {sample_id: float(score) for sample_id, score in sample_scores}
    while len(selected) < int(n_select) and available:
        best_id = None
        best_value = -math.inf
        for sample_id, score in available.items():
            penalty = 0.0
            if selected:
                penalty = max(max(0.0, cosine_similarity(vectors[sample_id], vectors[chosen])) for chosen in selected)
            objective = float(score) - (float(lambda_diversity) * float(penalty))
            if objective > best_value:
                best_value = objective
                best_id = sample_id
        if best_id is None:
            break
        selected.append(best_id)
        available.pop(best_id, None)
    return selected
