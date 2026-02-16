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


def _tokenize_sequence(seq: str, *, stage: str, location: str) -> list[int]:
    mapping = {"A": 0, "C": 1, "G": 2, "U": 3}
    cleaned = str(seq or "").strip().upper().replace("T", "U")
    tokens: list[int] = []
    bad: set[str] = set()
    for ch in cleaned:
        if ch in {"|", " ", "\t", "\n", "\r"}:
            continue
        if ch in mapping:
            tokens.append(mapping[ch])
        else:
            bad.add(ch)
    if not tokens:
        raise_error(stage, location, "sequencia vazia para encoder", impact="1", examples=["empty_sequence"])
    if bad:
        raise_error(stage, location, "sequencia contem simbolos invalidos para encoder", impact=str(len(bad)), examples=sorted(bad)[:8])
    return tokens


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
        try:
            import torch  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise_error(stage, location, "torch indisponivel para encoder ribonanzanet2", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
        try:
            model = torch.jit.load(str(model_path), map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            raise_error(stage, location, "falha ao carregar modelo torchscript ribonanzanet2", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
        model.eval()
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for seq in sequences:
                tokens = _tokenize_sequence(str(seq), stage=stage, location=location)
                x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
                try:
                    out = model(x)
                except Exception as exc:  # noqa: BLE001
                    raise_error(stage, location, "falha ao executar modelo ribonanzanet2", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
                if isinstance(out, (tuple, list)):
                    if not out:
                        raise_error(stage, location, "modelo ribonanzanet2 retornou output vazio", impact="1", examples=["empty_output"])
                    out = out[0]
                if not hasattr(out, "shape"):
                    raise_error(stage, location, "modelo ribonanzanet2 retornou tipo invalido", impact="1", examples=[str(type(out).__name__)])
                emb = out
                if emb.ndim == 3:
                    emb = emb.mean(dim=1)
                elif emb.ndim == 2:
                    if emb.shape[0] != 1:
                        emb = emb.mean(dim=0, keepdim=True)
                elif emb.ndim == 1:
                    emb = emb.unsqueeze(0)
                else:
                    raise_error(stage, location, "dimensao de embedding invalida do modelo", impact="1", examples=[str(tuple(int(x) for x in emb.shape))])
                if int(emb.shape[1]) != int(embedding_dim):
                    raise_error(
                        stage,
                        location,
                        "embedding_dim divergente do modelo",
                        impact="1",
                        examples=[f"expected={embedding_dim}", f"actual={int(emb.shape[1])}"],
                    )
                vec = emb[0].detach().cpu().to(dtype=torch.float32).numpy()
                norm = float(np.linalg.norm(vec))
                if not np.isfinite(norm) or norm <= 0:
                    raise_error(stage, location, "embedding invalido (norm <= 0)", impact="1", examples=[f"norm={norm}"])
                vec = vec.astype(np.float32) / float(norm)
                rows.append(vec)
        return np.asarray(rows, dtype=np.float32)
    else:
        salt = "mock"
    rows = [_hash_embedding(seq, dim=embedding_dim, salt=salt) for seq in sequences]
    return np.asarray(rows, dtype=np.float32)
