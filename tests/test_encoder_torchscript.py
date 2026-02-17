from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rna3d_local.encoder import encode_sequences


def test_ribonanzanet2_encoder_uses_torchscript_and_matches_dim(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class Toy(torch.nn.Module):
        __annotations__ = {}

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(4, dim)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (B,L)
            x = self.emb(tokens)  # (B,L,D)
            return x.mean(dim=1)  # (B,D)

    dim = 16
    model = Toy(dim)
    scripted = torch.jit.script(model)
    model_path = tmp_path / "toy_ribonanzanet2.pt"
    scripted.save(str(model_path))

    out = encode_sequences(
        ["ACGU", "AAAA", "GGUU"],
        encoder="ribonanzanet2",
        embedding_dim=dim,
        model_path=model_path,
        stage="EMBEDDING_INDEX",
        location="tests/test_encoder_torchscript.py:test_ribonanzanet2_encoder_uses_torchscript_and_matches_dim",
    )
    assert out.shape == (3, dim)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_mock_encoder_is_deterministic_and_normalized() -> None:
    out1 = encode_sequences(
        ["ACGU", "ACGU|ACGU", "AAAA", "GGUU"],
        encoder="mock",
        embedding_dim=32,
        model_path=None,
        stage="EMBEDDING_INDEX",
        location="tests/test_encoder_torchscript.py:test_mock_encoder_is_deterministic_and_normalized",
    )
    out2 = encode_sequences(
        ["ACGU", "ACGU|ACGU", "AAAA", "GGUU"],
        encoder="mock",
        embedding_dim=32,
        model_path=None,
        stage="EMBEDDING_INDEX",
        location="tests/test_encoder_torchscript.py:test_mock_encoder_is_deterministic_and_normalized",
    )
    assert out1.shape == (4, 32)
    assert np.allclose(out1, out2, atol=0.0)
    norms = np.linalg.norm(out1, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
