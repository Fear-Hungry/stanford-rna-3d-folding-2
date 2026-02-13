from __future__ import annotations

from rna3d_local.alignment import (
    map_target_to_template_alignment,
    map_target_to_template_positions,
    project_target_coordinates,
)


def test_map_target_to_template_positions_handles_template_insertion() -> None:
    mapping = map_target_to_template_positions(
        target_sequence="AACGUU",
        template_sequence="AAACGUU",
        location="tests/test_alignment_biopython.py:test_map_target_to_template_positions_handles_template_insertion",
    )
    assert mapping == {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}


def test_project_target_coordinates_interpolates_and_extrapolates_linearly() -> None:
    coords = project_target_coordinates(
        target_length=5,
        mapping={2: 2, 4: 4},
        template_coordinates={
            2: (0.0, 0.0, 0.0),
            4: (2.0, 2.0, 2.0),
        },
        location="tests/test_alignment_biopython.py:test_project_target_coordinates_interpolates_and_extrapolates_linearly",
        projection_mode="target_linear",
    )
    assert coords == [
        (-1.0, -1.0, -1.0),  # left extrapolation
        (0.0, 0.0, 0.0),  # anchor
        (1.0, 1.0, 1.0),  # interpolation
        (2.0, 2.0, 2.0),  # anchor
        (3.0, 3.0, 3.0),  # right extrapolation
    ]


def test_map_target_to_template_positions_ignores_mismatch_in_strict_mode() -> None:
    mapping = map_target_to_template_positions(
        target_sequence="AAAA",
        template_sequence="TTTT",
        location="tests/test_alignment_biopython.py:test_map_target_to_template_positions_ignores_mismatch_in_strict_mode",
        mapping_mode="strict_match",
    )
    assert mapping == {}


def test_map_target_to_template_positions_includes_mismatch_in_hybrid_mode() -> None:
    mapping = map_target_to_template_positions(
        target_sequence="AAAA",
        template_sequence="TTTT",
        location="tests/test_alignment_biopython.py:test_map_target_to_template_positions_includes_mismatch_in_hybrid_mode",
        mapping_mode="hybrid",
    )
    assert mapping == {1: 1, 2: 2, 3: 3, 4: 4}


def test_map_target_to_template_alignment_reports_pair_types() -> None:
    info = map_target_to_template_alignment(
        target_sequence="AGCU",
        template_sequence="GGAU",
        location="tests/test_alignment_biopython.py:test_map_target_to_template_alignment_reports_pair_types",
        mapping_mode="hybrid",
    )
    assert info.mapped_count == 4
    assert info.match_count == 2
    assert info.mismatch_count == 2
    assert info.pair_types[1] == "chem_compatible_mismatch"
    assert info.pair_types[3] == "mismatch"
