from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..errors import raise_error


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


def approx_tm_distance(a: np.ndarray, b: np.ndarray) -> float:
    sim = cosine_similarity(a, b)
    return float(max(0.0, 1.0 - sim))


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


def _pairwise_distance_matrix(*, sample_ids: list[str], vectors: dict[str, np.ndarray]) -> np.ndarray:
    size = int(len(sample_ids))
    dist = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(i + 1, size):
            dij = approx_tm_distance(vectors[sample_ids[i]], vectors[sample_ids[j]])
            dist[i, j] = dij
            dist[j, i] = dij
    return dist


def _neighbor_offsets() -> list[tuple[int, int, int]]:
    offsets: list[tuple[int, int, int]] = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                offsets.append((dx, dy, dz))
    return offsets


def estimate_clash_ratio(sample_df: pl.DataFrame, *, min_distance: float = 2.1, covalent_skip: int = 1) -> float:
    coords = sample_df.sort("resid").select("x", "y", "z").to_numpy().astype(np.float64)
    count = int(coords.shape[0])
    if count <= 2:
        return 0.0
    cell_size = float(min_distance)
    if cell_size <= 0.0:
        return 0.0
    cell_index = np.floor(coords / cell_size).astype(np.int64)
    cells: dict[tuple[int, int, int], list[int]] = {}
    for idx in range(count):
        key = (int(cell_index[idx, 0]), int(cell_index[idx, 1]), int(cell_index[idx, 2]))
        cells.setdefault(key, []).append(idx)

    offsets = _neighbor_offsets()
    min_sq = float(min_distance) * float(min_distance)
    clash_count = 0
    pair_count = 0
    for key, members in cells.items():
        for idx in members:
            for dx, dy, dz in offsets:
                neighbor_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                neigh = cells.get(neighbor_key)
                if neigh is None:
                    continue
                for jdx in neigh:
                    if jdx <= idx:
                        continue
                    if abs(jdx - idx) <= int(covalent_skip):
                        continue
                    pair_count += 1
                    delta = coords[idx] - coords[jdx]
                    if float(np.dot(delta, delta)) < min_sq:
                        clash_count += 1
    if pair_count <= 0:
        return 0.0
    return float(clash_count) / float(pair_count)


@dataclass(frozen=True)
class SampleCandidate:
    sample_id: str
    score: float
    clash_ratio: float
    adjusted_score: float


def prune_low_quality_half(
    *,
    candidates: list[SampleCandidate],
    keep_fraction: float,
    min_keep: int,
    stage: str,
    location: str,
) -> list[SampleCandidate]:
    if not candidates:
        raise_error(stage, location, "nenhum candidato disponivel para pre-filtro", impact="1", examples=[])
    if not (0.0 < float(keep_fraction) <= 1.0):
        raise_error(stage, location, "keep_fraction invalido para pre-filtro", impact="1", examples=[str(keep_fraction)])
    keep_count = max(int(min_keep), int(math.ceil(len(candidates) * float(keep_fraction))))
    ranked = sorted(candidates, key=lambda item: float(item.adjusted_score), reverse=True)
    if len(ranked) < int(min_keep):
        raise_error(
            stage,
            location,
            "candidatos insuficientes para manter minimo apos pre-filtro",
            impact=str(int(min_keep) - len(ranked)),
            examples=[str(item.sample_id) for item in ranked[:5]],
        )
    kept = ranked[:keep_count]
    if len(kept) < int(min_keep):
        raise_error(
            stage,
            location,
            "pre-filtro removeu amostras abaixo do minimo exigido",
            impact=str(int(min_keep) - len(kept)),
            examples=[str(item.sample_id) for item in kept[:5]],
        )
    return kept


def select_cluster_medoids(
    *,
    sample_scores: list[tuple[str, float]],
    vectors: dict[str, np.ndarray],
    n_select: int,
    lambda_diversity: float,
    stage: str,
    location: str,
) -> tuple[list[str], int]:
    if int(n_select) <= 0:
        raise_error(stage, location, "n_select invalido para clustering", impact="1", examples=[str(n_select)])
    if float(lambda_diversity) < 0.0:
        raise_error(stage, location, "lambda_diversity invalido para clustering", impact="1", examples=[str(lambda_diversity)])
    if not sample_scores:
        raise_error(stage, location, "sample_scores vazio para clustering", impact="1", examples=[])
    sample_ids = [str(sample_id) for sample_id, _score in sample_scores]
    if len(sample_ids) < int(n_select):
        raise_error(
            stage,
            location,
            "samples insuficientes para selecionar medoides",
            impact=str(int(n_select) - len(sample_ids)),
            examples=sample_ids[:5],
        )
    for sample_id in sample_ids:
        if sample_id not in vectors:
            raise_error(stage, location, "vetor latente ausente para sample", impact="1", examples=[sample_id])
    score_map = {str(sample_id): float(score) for sample_id, score in sample_scores}
    dist = _pairwise_distance_matrix(sample_ids=sample_ids, vectors=vectors)
    n_items = int(len(sample_ids))
    n_clusters = min(int(n_select), n_items)

    first_seed = max(range(n_items), key=lambda idx: score_map[sample_ids[idx]])
    seeds: list[int] = [int(first_seed)]
    while len(seeds) < n_clusters:
        best_idx = None
        best_value = -math.inf
        for idx in range(n_items):
            if idx in seeds:
                continue
            min_dist = min(float(dist[idx, seed]) for seed in seeds)
            score = score_map[sample_ids[idx]]
            objective = float(min_dist) + (1e-6 * float(score))
            if objective > best_value:
                best_value = objective
                best_idx = idx
        if best_idx is None:
            break
        seeds.append(int(best_idx))
    if len(seeds) != n_clusters:
        raise_error(stage, location, "falha na semeadura max-min de clusters", impact=str(int(n_clusters) - len(seeds)), examples=sample_ids[:5])

    assignment = np.zeros((n_items,), dtype=np.int64)
    for idx in range(n_items):
        nearest_cluster = int(np.argmin(np.array([dist[idx, seed] for seed in seeds], dtype=np.float64)))
        assignment[idx] = nearest_cluster

    selected_ids: list[str] = []
    for cluster_idx in range(n_clusters):
        members = [idx for idx in range(n_items) if int(assignment[idx]) == cluster_idx]
        if not members:
            continue
        if len(members) == 1:
            selected_ids.append(sample_ids[members[0]])
            continue
        intra = dist[np.ix_(members, members)].mean(axis=1)
        best_member = None
        best_value = -math.inf
        for local_idx, member_idx in enumerate(members):
            sid = sample_ids[member_idx]
            score = score_map[sid]
            objective = float(score) - (float(lambda_diversity) * float(intra[local_idx]))
            if objective > best_value:
                best_value = objective
                best_member = sid
        if best_member is None:
            raise_error(stage, location, "falha ao escolher medoide do cluster", impact="1", examples=[str(cluster_idx)])
        selected_ids.append(str(best_member))

    if len(selected_ids) < int(n_select):
        remaining = [sid for sid in sample_ids if sid not in set(selected_ids)]
        while len(selected_ids) < int(n_select) and remaining:
            best_sid = None
            best_value = -math.inf
            for sid in remaining:
                score = score_map[sid]
                min_dist = min(approx_tm_distance(vectors[sid], vectors[chosen]) for chosen in selected_ids)
                objective = float(score) + (float(lambda_diversity) * float(min_dist))
                if objective > best_value:
                    best_value = objective
                    best_sid = sid
            if best_sid is None:
                break
            selected_ids.append(str(best_sid))
            remaining = [sid for sid in remaining if sid != best_sid]

    if len(selected_ids) < int(n_select):
        raise_error(
            stage,
            location,
            "falha ao completar selecao de medoides",
            impact=str(int(n_select) - len(selected_ids)),
            examples=selected_ids[:5],
        )
    return selected_ids[: int(n_select)], int(n_clusters)


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
