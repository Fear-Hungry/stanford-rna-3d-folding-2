from __future__ import annotations

import os

from .errors import raise_error


ALLOW_MOCK_ENV = "RNA3D_ALLOW_MOCK_BACKENDS"


def _allow_mock(stage: str) -> bool:
    text = str(stage).strip().upper()
    if text.startswith("TEST"):
        return True
    return str(os.getenv(ALLOW_MOCK_ENV, "")).strip() == "1"


def enforce_no_mock_backend(*, backend: str, field: str, stage: str, location: str) -> None:
    mode = str(backend).strip().lower()
    if mode != "mock":
        return
    if _allow_mock(stage):
        return
    raise_error(
        stage,
        location,
        f"{field}=mock bloqueado por contrato; habilite {ALLOW_MOCK_ENV}=1 apenas para testes locais",
        impact="1",
        examples=[f"{field}=mock"],
    )
