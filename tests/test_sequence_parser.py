from __future__ import annotations

import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.se3.sequence_parser import parse_sequence_with_chains


def test_parse_sequence_with_chains_builds_1d_index_with_chain_jump() -> None:
    parsed = parse_sequence_with_chains(
        sequence="AC|GU",
        chain_separator="|",
        stage="TEST",
        location="tests/test_sequence_parser.py:test_parse_sequence_with_chains_builds_1d_index_with_chain_jump",
        target_id="T1",
    )
    assert parsed.residues == ["A", "C", "G", "U"]
    assert parsed.chain_index == [0, 0, 1, 1]
    assert parsed.chain_lengths == [2, 2]
    assert parsed.residue_position_index_1d == [0, 1, 1002, 1003]


def test_parse_sequence_with_chains_single_chain_index_is_contiguous() -> None:
    parsed = parse_sequence_with_chains(
        sequence="ACGU",
        chain_separator="|",
        stage="TEST",
        location="tests/test_sequence_parser.py:test_parse_sequence_with_chains_single_chain_index_is_contiguous",
        target_id="T2",
    )
    assert parsed.residue_position_index_1d == [0, 1, 2, 3]


def test_parse_sequence_with_chains_fails_for_invalid_chain_break_offset() -> None:
    with pytest.raises(PipelineError, match="chain_break_offset_1d"):
        parse_sequence_with_chains(
            sequence="AC|GU",
            chain_separator="|",
            stage="TEST",
            location="tests/test_sequence_parser.py:test_parse_sequence_with_chains_fails_for_invalid_chain_break_offset",
            target_id="T3",
            chain_break_offset_1d=0,
        )
