from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

try:
    from Bio.Align import PairwiseAligner
except Exception as exc:  # pragma: no cover - runtime validation below emits fail-fast message
    PairwiseAligner = None
    _BIO_IMPORT_ERROR = exc
else:
    _BIO_IMPORT_ERROR = None

from .errors import raise_error


@dataclass(frozen=True)
class AlignmentMapping:
    target_to_template: dict[int, int]
    pair_types: dict[int, str]
    mapped_count: int
    match_count: int
    mismatch_count: int
    chem_compatible_count: int


def _build_global_aligner(
    *,
    location: str,
    match_score: float = 2.0,
    mismatch_score: float = -1.0,
    open_gap_score: float = -5.0,
    extend_gap_score: float = -1.0,
):
    if PairwiseAligner is None:
        raise_error(
            "ALIGN",
            location,
            "biopython indisponivel para alinhamento global",
            impact="1",
            examples=[f"{type(_BIO_IMPORT_ERROR).__name__}:{_BIO_IMPORT_ERROR}"],
        )
    aligner = PairwiseAligner(mode="global")
    aligner.match_score = float(match_score)
    aligner.mismatch_score = float(mismatch_score)
    aligner.open_gap_score = float(open_gap_score)
    aligner.extend_gap_score = float(extend_gap_score)
    return aligner


def _validate_mapping_mode(*, mapping_mode: str, location: str) -> str:
    mode = str(mapping_mode).strip().lower()
    if mode not in {"strict_match", "hybrid", "chemical_class"}:
        raise_error(
            "ALIGN",
            location,
            "mapping_mode invalido",
            impact="1",
            examples=[str(mapping_mode)],
        )
    return mode


def _nucleotide_class(base: str) -> str:
    b = str(base).upper()
    if b in {"A", "G"}:
        return "purine"
    if b in {"C", "U", "T"}:
        return "pyrimidine"
    return "other"


def _is_chemically_compatible(*, a: str, b: str) -> bool:
    ca = _nucleotide_class(a)
    cb = _nucleotide_class(b)
    return ca in {"purine", "pyrimidine"} and ca == cb


def map_target_to_template_alignment(
    *,
    target_sequence: str,
    template_sequence: str,
    location: str,
    open_gap_score: float = -5.0,
    extend_gap_score: float = -1.0,
    mapping_mode: str = "hybrid",
) -> AlignmentMapping:
    t = (target_sequence or "").strip().upper()
    s = (template_sequence or "").strip().upper()
    if not t:
        raise_error("ALIGN", location, "target_sequence vazia", impact="1", examples=[])
    if not s:
        raise_error("ALIGN", location, "template_sequence vazia", impact="1", examples=[])
    mode = _validate_mapping_mode(mapping_mode=mapping_mode, location=location)

    aligner = _build_global_aligner(
        location=location,
        open_gap_score=float(open_gap_score),
        extend_gap_score=float(extend_gap_score),
    )
    alignments = aligner.align(t, s)
    it = iter(alignments)
    best = next(it, None)
    if best is None:
        raise_error("ALIGN", location, "alinhamento global sem resultado", impact="1", examples=[])

    mapping: dict[int, int] = {}
    pair_types: dict[int, str] = {}
    match_count = 0
    mismatch_count = 0
    chem_compatible_count = 0

    coords = best.coordinates
    if coords.shape[0] != 2:
        raise_error("ALIGN", location, "coordinates de alinhamento invalido", impact="1", examples=[str(coords.shape)])
    for i in range(coords.shape[1] - 1):
        t0 = int(coords[0, i])
        t1 = int(coords[0, i + 1])
        s0 = int(coords[1, i])
        s1 = int(coords[1, i + 1])
        dt = t1 - t0
        ds = s1 - s0
        if dt <= 0 or ds <= 0:
            continue
        span = min(dt, ds)
        for k in range(span):
            tp = t0 + k
            sp = s0 + k
            if tp < 0 or tp >= len(t) or sp < 0 or sp >= len(s):
                raise_error(
                    "ALIGN",
                    location,
                    "coordenadas de alinhamento fora de faixa",
                    impact="1",
                    examples=[f"tp={tp} sp={sp} len_t={len(t)} len_s={len(s)}"],
                )
            tb = t[tp]
            sb = s[sp]
            accepted = False
            pair_type = "mismatch"
            if tb == sb:
                accepted = True
                pair_type = "match"
            elif mode == "hybrid":
                accepted = True
                if _is_chemically_compatible(a=tb, b=sb):
                    pair_type = "chem_compatible_mismatch"
            elif mode == "chemical_class":
                if _is_chemically_compatible(a=tb, b=sb):
                    accepted = True
                    pair_type = "chem_compatible_mismatch"
            if not accepted:
                continue
            mapping[tp + 1] = sp + 1
            pair_types[tp + 1] = pair_type
            if pair_type == "match":
                match_count += 1
            elif pair_type == "chem_compatible_mismatch":
                mismatch_count += 1
                chem_compatible_count += 1
            else:
                mismatch_count += 1

    return AlignmentMapping(
        target_to_template=mapping,
        pair_types=pair_types,
        mapped_count=int(len(mapping)),
        match_count=int(match_count),
        mismatch_count=int(mismatch_count),
        chem_compatible_count=int(chem_compatible_count),
    )


def map_target_to_template_positions(
    *,
    target_sequence: str,
    template_sequence: str,
    location: str,
    open_gap_score: float = -5.0,
    extend_gap_score: float = -1.0,
    mapping_mode: str = "hybrid",
) -> dict[int, int]:
    """
    Build a deterministic positional mapping (1-based) from target residues to template residues.
    """
    mapping = map_target_to_template_alignment(
        target_sequence=target_sequence,
        template_sequence=template_sequence,
        location=location,
        open_gap_score=float(open_gap_score),
        extend_gap_score=float(extend_gap_score),
        mapping_mode=mapping_mode,
    )
    return mapping.target_to_template


def normalized_global_alignment_similarity(
    *,
    target_sequence: str,
    template_sequence: str,
    location: str,
    open_gap_score: float = -5.0,
    extend_gap_score: float = -1.0,
) -> float:
    """
    Compute a normalized global alignment similarity in [0,1].
    """
    t = (target_sequence or "").strip().upper()
    s = (template_sequence or "").strip().upper()
    if not t:
        raise_error("ALIGN", location, "target_sequence vazia", impact="1", examples=[])
    if not s:
        raise_error("ALIGN", location, "template_sequence vazia", impact="1", examples=[])

    aligner = _build_global_aligner(
        location=location,
        open_gap_score=float(open_gap_score),
        extend_gap_score=float(extend_gap_score),
    )
    raw = float(aligner.score(t, s))
    den = 2.0 * float(min(len(t), len(s)))
    if den <= 0.0:
        raise_error("ALIGN", location, "denominador invalido para normalizacao", impact="1", examples=[f"len_t={len(t)} len_s={len(s)}"])
    ratio = raw / den
    if ratio < 0.0:
        return 0.0
    if ratio > 1.0:
        return 1.0
    return float(ratio)


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
    projection_mode: str = "template_warped",
) -> list[tuple[float, float, float]]:
    """
    Produce coordinates for each target residue (1..target_length).
    Uses exact mapped coordinates when available and deterministic linear interpolation/
    extrapolation around nearest aligned anchors for unmatched positions.
    """
    mode = str(projection_mode).strip().lower()
    if mode not in {"target_linear", "template_warped"}:
        raise_error("ALIGN", location, "projection_mode invalido", impact="1", examples=[str(projection_mode)])
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

    def _interp_coords(c0: tuple[float, float, float], c1: tuple[float, float, float], alpha: float) -> tuple[float, float, float]:
        return (
            float(c0[0] + (c1[0] - c0[0]) * float(alpha)),
            float(c0[1] + (c1[1] - c0[1]) * float(alpha)),
            float(c0[2] + (c1[2] - c0[2]) * float(alpha)),
        )

    if mode == "target_linear":
        anchors = sorted(exact_coords.keys())
        anchor_count = len(anchors)
        out: list[tuple[float, float, float]] = []
        for i in range(1, target_length + 1):
            if i in exact_coords:
                out.append(exact_coords[i])
                continue
            if anchor_count == 1:
                out.append(exact_coords[anchors[0]])
                continue
            idx = bisect_left(anchors, i)
            if idx <= 0:
                p0, p1 = anchors[0], anchors[1]
            elif idx >= anchor_count:
                p0, p1 = anchors[-2], anchors[-1]
            else:
                p0, p1 = anchors[idx - 1], anchors[idx]
            den = float(p1 - p0)
            alpha = 0.0 if den == 0.0 else float(i - p0) / den
            out.append(_interp_coords(exact_coords[p0], exact_coords[p1], alpha))
        return out

    anchor_pairs: list[tuple[int, int]] = []
    for tpos, spos in sorted(mapping.items()):
        if spos in template_coordinates:
            anchor_pairs.append((int(tpos), int(spos)))
    if not anchor_pairs:
        raise_error("ALIGN", location, "mapping sem pares ancora validos", impact="1", examples=[])

    tpl_positions = sorted(int(p) for p in template_coordinates.keys())
    if not tpl_positions:
        raise_error("ALIGN", location, "template_coordinates vazio apos normalizacao", impact="1", examples=[])

    def _coord_at_template_pos(sval: float) -> tuple[float, float, float]:
        if len(tpl_positions) == 1:
            return template_coordinates[tpl_positions[0]]
        idx = bisect_left(tpl_positions, float(sval))
        if idx <= 0:
            s0, s1 = tpl_positions[0], tpl_positions[1]
        elif idx >= len(tpl_positions):
            s0, s1 = tpl_positions[-2], tpl_positions[-1]
        else:
            s0, s1 = tpl_positions[idx - 1], tpl_positions[idx]
        c0 = template_coordinates[s0]
        c1 = template_coordinates[s1]
        den = float(s1 - s0)
        alpha = 0.0 if den == 0.0 else float(sval - s0) / den
        return _interp_coords(c0, c1, alpha)

    t_anchors = [int(t) for t, _ in anchor_pairs]
    out: list[tuple[float, float, float]] = []
    for i in range(1, target_length + 1):
        if i in exact_coords:
            out.append(exact_coords[i])
            continue
        if len(anchor_pairs) == 1:
            out.append(template_coordinates[anchor_pairs[0][1]])
            continue
        idx = bisect_left(t_anchors, i)
        if idx <= 0:
            t0, s0 = anchor_pairs[0]
            t1, s1 = anchor_pairs[1]
        elif idx >= len(anchor_pairs):
            t0, s0 = anchor_pairs[-2]
            t1, s1 = anchor_pairs[-1]
        else:
            t0, s0 = anchor_pairs[idx - 1]
            t1, s1 = anchor_pairs[idx]
        den = float(t1 - t0)
        alpha = 0.0 if den == 0.0 else float(i - t0) / den
        s_est = float(s0 + (s1 - s0) * alpha)
        out.append(_coord_at_template_pos(s_est))
    return out
