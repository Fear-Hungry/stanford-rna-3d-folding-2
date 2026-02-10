from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import xxhash


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _kmer_iter(seq: str, k: int) -> Iterable[str]:
    s = seq.strip().upper()
    if not s:
        yield ""
        return
    if len(s) < k:
        yield s
        return
    for i in range(0, len(s) - k + 1):
        yield s[i : i + k]


def _minhash_signature(seq: str, *, k: int, n_hashes: int) -> list[int]:
    kmers = list(_kmer_iter(seq, k))
    sig: list[int] = []
    for seed in range(n_hashes):
        mn = None
        for km in kmers:
            h = xxhash.xxh64(km, seed=seed).intdigest()
            mn = h if mn is None else min(mn, h)
        sig.append(int(mn if mn is not None else 0))
    return sig


@dataclass(frozen=True)
class ClusterResult:
    cluster_ids: list[str]
    n_clusters: int


def cluster_targets_minhash(
    *,
    target_ids: list[str],
    sequences: list[str],
    k: int = 5,
    n_hashes: int = 32,
    bands: int = 8,
) -> ClusterResult:
    """
    Deterministic, approximate sequence clustering using MinHash + LSH banding.
    This is intended for robust CV splits (anti-leakage), not for biological rigor.
    """
    assert len(target_ids) == len(sequences)
    n = len(target_ids)
    if n == 0:
        return ClusterResult(cluster_ids=[], n_clusters=0)

    if n_hashes % bands != 0:
        raise ValueError("n_hashes must be divisible by bands")
    rows_per_band = n_hashes // bands

    sigs = [_minhash_signature(sequences[i], k=k, n_hashes=n_hashes) for i in range(n)]

    uf = _UnionFind(n)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        sig = sigs[i]
        for b in range(bands):
            start = b * rows_per_band
            chunk = sig[start : start + rows_per_band]
            hb = xxhash.xxh64(seed=b)
            for v in chunk:
                hb.update(int(v).to_bytes(8, "little", signed=False))
            key = (b, hb.intdigest())
            buckets.setdefault(key, []).append(i)

    for _, idxs in buckets.items():
        if len(idxs) <= 1:
            continue
        root = idxs[0]
        for j in idxs[1:]:
            uf.union(root, j)

    comps: dict[int, list[int]] = {}
    for i in range(n):
        r = uf.find(i)
        comps.setdefault(r, []).append(i)

    # stable cluster id: min target_id in the component
    cluster_id_by_root: dict[int, str] = {}
    for r, idxs in comps.items():
        names = sorted(target_ids[i] for i in idxs)
        cluster_id_by_root[r] = names[0]

    cluster_ids = [cluster_id_by_root[uf.find(i)] for i in range(n)]
    return ClusterResult(cluster_ids=cluster_ids, n_clusters=len(set(cluster_ids)))


def assign_folds_from_clusters(*, cluster_ids: list[str], n_folds: int, seed: int) -> list[int]:
    folds: list[int] = []
    for cid in cluster_ids:
        h = xxhash.xxh64(cid, seed=seed).intdigest()
        folds.append(int(h % n_folds))
    return folds

