from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PipelineError(RuntimeError):
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


def _fmt_examples(examples: Iterable[str] | None, max_items: int = 8) -> str:
    if not examples:
        return "-"
    items = list(examples)
    items = [str(x) for x in items if str(x)]
    if not items:
        return "-"
    return ",".join(items[:max_items])


def raise_error(
    stage: str,
    location: str,
    cause: str,
    *,
    impact: str,
    examples: Iterable[str] | None = None,
) -> None:
    """
    Required format (AGENTS.md):
    `[ETAPA] [ARQUIVO/FUNCAO] <causa> | impacto=<qtd> | exemplos=<itens>`
    """
    msg = f"[{stage}] [{location}] {cause} | impacto={impact} | exemplos={_fmt_examples(examples)}"
    raise PipelineError(msg)

