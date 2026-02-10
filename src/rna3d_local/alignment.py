from __future__ import annotations

from difflib import SequenceMatcher

from .errors import raise_error


def map_target_to_template_positions(
    *,
    target_sequence: str,
    template_sequence: str,
    location: str,
) -> dict[int, int]:
    """
    Build a deterministic positional mapping (1-based) from target residues to template residues.
    """
    t = (target_sequence or "").strip().upper()
    s = (template_sequence or "").strip().upper()
    if not t:
        raise_error("ALIGN", location, "target_sequence vazia", impact="1", examples=[])
    if not s:
        raise_error("ALIGN", location, "template_sequence vazia", impact="1", examples=[])

    matcher = SequenceMatcher(a=t, b=s, autojunk=False)
    mapping: dict[int, int] = {}
    for m in matcher.get_matching_blocks():
        if m.size <= 0:
            continue
        for i in range(m.size):
            mapping[m.a + i + 1] = m.b + i + 1
    return mapping


def compute_coverage(*, mapped_positions: int, target_length: int, location: str) -> float:
    if target_length <= 0:
        raise_error("ALIGN", location, "target_length invalido", impact="1", examples=[str(target_length)])
    return float(mapped_positions) / float(target_length)


def project_target_coordinates(
    *,
    target_length: int,
    mapping: dict[int, int],
    template_coordinates: dict[int, tuple[float, float, float]],
    location: str,
) -> list[tuple[float, float, float]]:
    """
    Produce coordinates for each target residue (1..target_length).
    Uses exact mapped coordinates when available and deterministic geometric interpolation
    around nearest mapped residues for unmatched positions.
    """
    if target_length <= 0:
        raise_error("ALIGN", location, "target_length invalido", impact="1", examples=[str(target_length)])
    if not mapping:
        raise_error("ALIGN", location, "mapping vazio", impact="1", examples=[])
    if not template_coordinates:
        raise_error("ALIGN", location, "template_coordinates vazio", impact="1", examples=[])

    exact_coords: dict[int, tuple[float, float, float]] = {}
    for target_pos, template_pos in mapping.items():
        coord = template_coordinates.get(template_pos)
        if coord is not None:
            exact_coords[target_pos] = coord
    if not exact_coords:
        raise_error("ALIGN", location, "mapping sem coordenadas validas", impact="1", examples=[])

    mapped_positions = sorted(exact_coords.keys())
    out: list[tuple[float, float, float]] = []
    for i in range(1, target_length + 1):
        if i in exact_coords:
            out.append(exact_coords[i])
            continue
        nearest = min(mapped_positions, key=lambda p: (abs(i - p), p))
        x0, y0, z0 = exact_coords[nearest]
        delta = float(i - nearest)
        out.append((x0 + 0.25 * delta, y0 - 0.10 * delta, z0 + 0.35 * delta))
    return out

