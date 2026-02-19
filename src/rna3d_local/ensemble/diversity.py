from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..errors import raise_error

_DIVERSITY_STAGE = "DIVERSITY"
_DIVERSITY_FILE = "src/rna3d_local/ensemble/diversity.py"
_VECTOR_EPS = 1e-12
_REQUIRED_DIVERSITY_COLUMNS = ("sample_id", "resid", "x", "y", "z")
_NEIGHBOR_OFFSETS = tuple((dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1))


def _normalize_group_key(group_key: object) -> str:
    return str(group_key[0]) if isinstance(group_key, tuple) else str(group_key)


def _raise_diversity_contract(function: str, cause: str, *, impact: str, examples: list[str]) -> None:
    raise_error(_DIVERSITY_STAGE, f"{_DIVERSITY_FILE}:{function}", cause, impact=impact, examples=examples)


def _kabsch_align_centered(mobile_centered: np.ndarray, target_centered: np.ndarray) -> np.ndarray:
    if mobile_centered.ndim != 2:
        raise ValueError(f"kabsch coords shape invalido: mobile={mobile_centered.shape} target={target_centered.shape}")
    if target_centered.ndim != 2:
        raise ValueError(f"kabsch coords shape invalido: mobile={mobile_centered.shape} target={target_centered.shape}")
    if int(mobile_centered.shape[1]) != 3:
        raise ValueError(f"kabsch coords shape invalido: mobile={mobile_centered.shape} target={target_centered.shape}")
    if int(target_centered.shape[1]) != 3:
        raise ValueError(f"kabsch coords shape invalido: mobile={mobile_centered.shape} target={target_centered.shape}")
    if mobile_centered.shape != target_centered.shape:
        raise ValueError(f"kabsch coords shape invalido: mobile={mobile_centered.shape} target={target_centered.shape}")
    cov = mobile_centered.T @ target_centered
    u, _s, vt = np.linalg.svd(cov, full_matrices=False)
    v = vt.T
    d = float(np.linalg.det(v @ u.T))
    correction = np.eye(3, dtype=np.float64)
    correction[2, 2] = 1.0 if d >= 0.0 else -1.0
    rotation = v @ correction @ u.T
    return mobile_centered @ rotation.T


def _vector_from_centered_coords(coords_centered: np.ndarray) -> np.ndarray:
    vec = coords_centered.reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= _VECTOR_EPS:
        return np.zeros_like(vec)
    return vec / norm


def _coords_for_sample(sample_df: pl.DataFrame) -> np.ndarray:
    return sample_df.select("x", "y", "z").to_numpy().astype(np.float64, copy=False)


def _extract_sample_parts(target_df: pl.DataFrame, *, stage: str, location: str) -> dict[str, pl.DataFrame]:
    missing = [col for col in _REQUIRED_DIVERSITY_COLUMNS if col not in target_df.columns]
    if missing:
        raise_error(
            stage,
            location,
            "target_df sem colunas obrigatorias para diversidade",
            impact=str(int(len(missing))),
            examples=[str(item) for item in missing[:8]],
        )
    parts: dict[str, pl.DataFrame] = {}
    for raw_sample_id, part in target_df.group_by("sample_id", maintain_order=True):
        sample_id = _normalize_group_key(raw_sample_id)
        if sample_id in parts:
            raise_error(stage, location, "sample_id duplicado apos agrupamento para diversidade", impact="1", examples=[sample_id])
        parts[sample_id] = part.sort("resid")
    if not parts:
        raise_error(stage, location, "nenhum sample_id para diversidade", impact="1", examples=[])
    return parts


def _extract_resids(sample_part: pl.DataFrame, *, sample_id: str, stage: str, location: str) -> np.ndarray:
    resid_col = sample_part.get_column("resid")
    total = int(sample_part.height)
    unique = int(resid_col.n_unique())
    if unique != total:
        dup_examples = (
            sample_part.filter(pl.col("resid").is_duplicated()).select("resid").get_column("resid").head(5).to_list()
        )
        raise_error(
            stage,
            location,
            "resid duplicado dentro do sample para diversidade",
            impact=str(int(total - unique)),
            examples=[f"{sample_id}:{item}" for item in dup_examples],
        )
    return resid_col.to_numpy()


def build_sample_vectors(target_df: pl.DataFrame, *, stage: str, location: str, anchor_sample_id: str | None = None) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    sample_parts = _extract_sample_parts(target_df, stage=stage, location=location)

    anchor_id = str(anchor_sample_id) if anchor_sample_id is not None else next(iter(sample_parts.keys()))
    anchor_part = sample_parts.get(anchor_id)
    if anchor_part is None:
        raise_error(stage, location, "anchor_sample_id ausente no target_df", impact="1", examples=[anchor_id])

    anchor_resids = _extract_resids(anchor_part, sample_id=anchor_id, stage=stage, location=location)
    anchor_coords = _coords_for_sample(anchor_part)
    if not np.isfinite(anchor_coords).all():
        raise_error(stage, location, "coordenadas nao-finitas no anchor para diversidade", impact="1", examples=[anchor_id])
    anchor_center = anchor_coords.mean(axis=0, keepdims=True)
    anchor_centered = anchor_coords - anchor_center
    anchor_len = int(anchor_centered.shape[0])
    if anchor_len <= 1:
        raise_error(stage, location, "anchor com residuos insuficientes para diversidade", impact="1", examples=[anchor_id, f"n={anchor_len}"])

    vectors[anchor_id] = _vector_from_centered_coords(anchor_centered)
    for sample_id, sample_part in sample_parts.items():
        if sample_id == anchor_id:
            continue
        coords = _coords_for_sample(sample_part)
        if int(coords.shape[0]) != anchor_len:
            raise_error(
                stage,
                location,
                "samples com comprimentos divergentes para diversidade (resid mismatch)",
                impact="1",
                examples=[f"anchor={anchor_id}:n={anchor_len}", f"{sample_id}:n={int(coords.shape[0])}"],
            )
        if not np.isfinite(coords).all():
            raise_error(stage, location, "coordenadas nao-finitas para diversidade", impact="1", examples=[sample_id])

        sample_resids = _extract_resids(sample_part, sample_id=sample_id, stage=stage, location=location)
        if sample_resids.shape != anchor_resids.shape:
            raise_error(
                stage,
                location,
                "samples com comprimentos divergentes para diversidade (resid mismatch)",
                impact="1",
                examples=[f"anchor={anchor_id}:n={anchor_len}", f"{sample_id}:n={int(sample_resids.shape[0])}"],
            )
        if not np.array_equal(sample_resids, anchor_resids):
            mismatch_mask = sample_resids != anchor_resids
            mismatch_idx = int(np.argmax(mismatch_mask))
            raise_error(
                stage,
                location,
                "samples com resid divergente para diversidade (order/value mismatch)",
                impact="1",
                examples=[
                    f"anchor={anchor_id}:resid={anchor_resids[mismatch_idx]}",
                    f"{sample_id}:resid={sample_resids[mismatch_idx]}",
                    f"pos={mismatch_idx}",
                ],
            )

        coords_centered = coords - coords.mean(axis=0, keepdims=True)
        aligned = _kabsch_align_centered(coords_centered, anchor_centered)
        vectors[sample_id] = _vector_from_centered_coords(aligned)
    return vectors


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 1:
        _raise_diversity_contract(
            "cosine_similarity",
            "vetores devem ser 1D para similaridade",
            impact="1",
            examples=[f"a.ndim={a.ndim}"],
        )
    if b.ndim != 1:
        _raise_diversity_contract(
            "cosine_similarity",
            "vetores devem ser 1D para similaridade",
            impact="1",
            examples=[f"b.ndim={b.ndim}"],
        )
    if a.shape != b.shape:
        _raise_diversity_contract(
            "cosine_similarity",
            "vetores com shape divergente para similaridade",
            impact="1",
            examples=[f"a={a.shape}", f"b={b.shape}"],
        )
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= _VECTOR_EPS:
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


def estimate_clash_ratio(sample_df: pl.DataFrame, *, min_distance: float = 2.1, covalent_skip: int = 1) -> float:
    if float(min_distance) <= 0.0:
        _raise_diversity_contract(
            "estimate_clash_ratio",
            "min_distance invalido para estimativa de clash",
            impact="1",
            examples=[str(min_distance)],
        )
    if int(covalent_skip) < 0:
        _raise_diversity_contract(
            "estimate_clash_ratio",
            "covalent_skip invalido para estimativa de clash",
            impact="1",
            examples=[str(covalent_skip)],
        )

    coords = sample_df.sort("resid").select("x", "y", "z").to_numpy().astype(np.float64)
    count = int(coords.shape[0])
    if count <= 2:
        return 0.0
    if not np.isfinite(coords).all():
        _raise_diversity_contract(
            "estimate_clash_ratio",
            "coordenadas nao-finitas na estimativa de clash",
            impact="1",
            examples=[],
        )

    cell_size = float(min_distance)
    cell_index = np.floor(coords / cell_size).astype(np.int64)
    cells: dict[tuple[int, int, int], list[int]] = {}
    for idx in range(count):
        key = (int(cell_index[idx, 0]), int(cell_index[idx, 1]), int(cell_index[idx, 2]))
        cells.setdefault(key, []).append(idx)

    min_sq = float(min_distance) * float(min_distance)
    clash_count = 0
    pair_count = 0
    for key, members in cells.items():
        for idx in members:
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
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
    keep_fraction_value = float(keep_fraction)
    if keep_fraction_value <= 0.0:
        raise_error(stage, location, "keep_fraction invalido para pre-filtro", impact="1", examples=[str(keep_fraction)])
    if keep_fraction_value > 1.0:
        raise_error(stage, location, "keep_fraction invalido para pre-filtro", impact="1", examples=[str(keep_fraction)])
    keep_count = max(int(min_keep), int(math.ceil(len(candidates) * keep_fraction_value)))
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
    requested_select = int(n_select)
    if requested_select <= 0:
        raise_error(stage, location, "n_select invalido para clustering", impact="1", examples=[str(requested_select)])
    if float(lambda_diversity) < 0.0:
        raise_error(stage, location, "lambda_diversity invalido para clustering", impact="1", examples=[str(lambda_diversity)])
    if not sample_scores:
        raise_error(stage, location, "sample_scores vazio para clustering", impact="1", examples=[])

    sample_ids = [str(sample_id) for sample_id, _score in sample_scores]
    if len(sample_ids) < requested_select:
        raise_error(
            stage,
            location,
            "samples insuficientes para selecionar medoides",
            impact=str(requested_select - len(sample_ids)),
            examples=sample_ids[:5],
        )
    for sample_id in sample_ids:
        if sample_id not in vectors:
            raise_error(stage, location, "vetor latente ausente para sample", impact="1", examples=[sample_id])

    score_map = {str(sample_id): float(score) for sample_id, score in sample_scores}
    dist = _pairwise_distance_matrix(sample_ids=sample_ids, vectors=vectors)
    n_items = int(len(sample_ids))
    n_clusters = min(requested_select, n_items)

    first_seed = max(range(n_items), key=lambda idx: score_map[sample_ids[idx]])
    seeds: list[int] = [int(first_seed)]
    seed_set = {int(first_seed)}
    while len(seeds) < n_clusters:
        best_idx = None
        best_value = -math.inf
        for idx in range(n_items):
            if idx in seed_set:
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
        seed_set.add(int(best_idx))
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

    selected_set = set(selected_ids)
    if len(selected_ids) < requested_select:
        remaining = [sid for sid in sample_ids if sid not in selected_set]
        while len(selected_ids) < requested_select:
            if not remaining:
                break
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
            selected_set.add(str(best_sid))
            remaining = [sid for sid in remaining if sid != best_sid]

    if len(selected_ids) < requested_select:
        raise_error(
            stage,
            location,
            "falha ao completar selecao de medoides",
            impact=str(requested_select - len(selected_ids)),
            examples=selected_ids[:5],
        )
    return selected_ids[:requested_select], int(n_clusters)


def greedy_diverse_selection(
    *,
    sample_scores: list[tuple[str, float]],
    vectors: dict[str, np.ndarray],
    n_select: int,
    lambda_diversity: float,
) -> list[str]:
    requested_select = int(n_select)
    if requested_select <= 0:
        _raise_diversity_contract(
            "greedy_diverse_selection",
            "n_select invalido para selecao greedy",
            impact="1",
            examples=[str(requested_select)],
        )
    if float(lambda_diversity) < 0.0:
        _raise_diversity_contract(
            "greedy_diverse_selection",
            "lambda_diversity invalido para selecao greedy",
            impact="1",
            examples=[str(lambda_diversity)],
        )
    if not sample_scores:
        _raise_diversity_contract("greedy_diverse_selection", "sample_scores vazio para selecao greedy", impact="1", examples=[])

    selected: list[str] = []
    available = {str(sample_id): float(score) for sample_id, score in sample_scores}
    for sample_id in available:
        if sample_id not in vectors:
            _raise_diversity_contract(
                "greedy_diverse_selection",
                "vetor latente ausente para sample",
                impact="1",
                examples=[sample_id],
            )
    while len(selected) < requested_select:
        if not available:
            break
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
