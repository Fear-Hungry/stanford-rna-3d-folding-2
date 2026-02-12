from __future__ import annotations

from bisect import bisect_left

try:
    from Bio.Align import PairwiseAligner
except Exception as exc:  # pragma: no cover - runtime validation below emits fail-fast message
    PairwiseAligner = None
    _BIO_IMPORT_ERROR = exc
else:
    _BIO_IMPORT_ERROR = None

from .errors import raise_error


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


def map_target_to_template_positions(
    *,
    target_sequence: str,
    template_sequence: str,
    location: str,
    open_gap_score: float = -5.0,
    extend_gap_score: float = -1.0,
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
    # Use the best-ranked alignment deterministically.

    mapping: dict[int, int] = {}
    # Build mapping from alignment coordinates and keep only strict nucleotide matches.
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
            # Gap segment in one of the sequences.
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
            if t[tp] == s[sp]:
                mapping[tp + 1] = sp + 1
    return mapping


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
) -> list[tuple[float, float, float]]:
    """
    Produce coordinates for each target residue (1..target_length).
    Uses exact mapped coordinates when available and deterministic linear interpolation/
    extrapolation around nearest aligned anchors for unmatched positions.
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

    anchors = sorted(exact_coords.keys())
    anchor_count = len(anchors)

    def _interp(p0: int, p1: int, p: int) -> tuple[float, float, float]:
        c0 = exact_coords[p0]
        c1 = exact_coords[p1]
        den = float(p1 - p0)
        if den == 0.0:
            return c0
        alpha = float(p - p0) / den
        return (
            float(c0[0] + (c1[0] - c0[0]) * alpha),
            float(c0[1] + (c1[1] - c0[1]) * alpha),
            float(c0[2] + (c1[2] - c0[2]) * alpha),
        )

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
            # Extrapolate left using first segment slope.
            out.append(_interp(anchors[0], anchors[1], i))
            continue
        if idx >= anchor_count:
            # Extrapolate right using last segment slope.
            out.append(_interp(anchors[-2], anchors[-1], i))
            continue
        # Interpolate between nearest left and right anchors.
        out.append(_interp(anchors[idx - 1], anchors[idx], i))
    return out
