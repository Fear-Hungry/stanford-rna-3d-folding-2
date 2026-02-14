from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

CANDIDATE_POOL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "target_id",
    "ID",
    "resid",
    "resname",
    "model_id",
    "x",
    "y",
    "z",
    "coverage",
    "similarity",
    "template_uid",
)

CANDIDATE_POOL_DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    "coverage",
    "similarity",
    "mapped_ratio",
    "match_ratio",
    "mismatch_ratio",
    "chem_compatible_ratio",
    "path_length",
    "step_mean",
    "step_std",
    "radius_gyr",
    "gap_open_score",
    "gap_extend_score",
    "qa_score_base",
    "resid_count",
    "dist_off_1",
    "dist_off_2",
    "dist_off_4",
    "dist_off_8",
    "dist_off_16",
    "dist_off_32",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        x = float(value)
        if not (x == x):
            return float(default)
        if x in (float("inf"), float("-inf")):
            return float(default)
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _mean_distance_for_offset(*, coords: list[tuple[float, float, float]], offset: int) -> float:
    n = len(coords)
    if n <= int(offset):
        return 0.0
    total = 0.0
    count = 0
    for i in range(0, n - int(offset)):
        a = coords[i]
        b = coords[i + int(offset)]
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        dz = float(a[2]) - float(b[2])
        total += float((dx * dx + dy * dy + dz * dz) ** 0.5)
        count += 1
    if count <= 0:
        return 0.0
    return float(total / float(count))


__all__ = [
    "CANDIDATE_POOL_DEFAULT_FEATURE_NAMES",
    "CANDIDATE_POOL_REQUIRED_COLUMNS",
    "_mean_distance_for_offset",
    "_rel",
    "_safe_float",
    "_utc_now",
]
