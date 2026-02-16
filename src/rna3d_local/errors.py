from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message


def raise_error(stage: str, location: str, cause: str, *, impact: str, examples: list[str]) -> None:
    examples_str = ",".join(str(item) for item in examples) if examples else "-"
    msg = f"[{stage}] [{location}] {cause} | impacto={impact} | exemplos={examples_str}"
    raise PipelineError(msg)
