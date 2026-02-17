from __future__ import annotations

import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.runners.chai1 import _normalize_seq


def test_chai1_normalize_seq_accepts_dynamic_chain_separator() -> None:
    out = _normalize_seq(
        "acg:uU",
        stage="TEST",
        location="tests/test_chai1_runner_chain_separator.py:test_chai1_normalize_seq_accepts_dynamic_chain_separator",
        target_id="T1",
        chain_separator=":",
    )
    assert out == "ACG:UU"


def test_chai1_normalize_seq_rejects_unexpected_separator_char() -> None:
    with pytest.raises(PipelineError, match="simbolos invalidos"):
        _normalize_seq(
            "ACG:UU",
            stage="TEST",
            location="tests/test_chai1_runner_chain_separator.py:test_chai1_normalize_seq_rejects_unexpected_separator_char",
            target_id="T1",
            chain_separator="|",
        )

