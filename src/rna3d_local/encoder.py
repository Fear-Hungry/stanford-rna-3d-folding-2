from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .errors import raise_error
from .mock_policy import enforce_no_mock_backend


def _hash_embedding(text: str, *, dim: int, salt: str) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    seq = (text or "").strip().upper()
    if not seq:
        return vec
    tokens = [seq[i : i + 3] for i in range(max(1, len(seq) - 2))]
    for token in tokens:
        digest = hashlib.sha256(f"{salt}:{token}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big", signed=False) % dim
        vec[bucket] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def encode_sequences(
    sequences: list[str],
    *,
    encoder: str,
    embedding_dim: int,
    model_path: Path | None,
    stage: str,
    location: str,
) -> np.ndarray:
    if embedding_dim <= 0:
        raise_error(stage, location, "embedding_dim deve ser > 0", impact="1", examples=[str(embedding_dim)])
    mode = str(encoder).strip().lower()
    if mode not in {"ribonanzanet2", "mock"}:
        raise_error(stage, location, "encoder invalido", impact="1", examples=[encoder])
    enforce_no_mock_backend(backend=mode, field="encoder", stage=stage, location=location)
    if mode == "ribonanzanet2":
        if model_path is None:
            raise_error(stage, location, "model_path obrigatorio para ribonanzanet2", impact="1", examples=["model_path=None"])
        if not model_path.exists():
            raise_error(stage, location, "model_path nao encontrado para ribonanzanet2", impact="1", examples=[str(model_path)])
        salt = model_path.name
    else:
        salt = "mock"
    rows = [_hash_embedding(seq, dim=embedding_dim, salt=salt) for seq in sequences]
    return np.asarray(rows, dtype=np.float32)
