from __future__ import annotations

from rna3d_local.alignment import map_target_to_template_positions, project_target_coordinates


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
    )
    assert coords == [
        (-1.0, -1.0, -1.0),  # left extrapolation
        (0.0, 0.0, 0.0),  # anchor
        (1.0, 1.0, 1.0),  # interpolation
        (2.0, 2.0, 2.0),  # anchor
        (3.0, 3.0, 3.0),  # right extrapolation
    ]


def test_map_target_to_template_positions_ignores_mismatch() -> None:
    mapping = map_target_to_template_positions(
        target_sequence="AAAA",
        template_sequence="TTTT",
        location="tests/test_alignment_biopython.py:test_map_target_to_template_positions_ignores_mismatch",
    )
    assert mapping == {}
