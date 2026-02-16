from __future__ import annotations

from dataclasses import dataclass

from ..errors import raise_error


@dataclass(frozen=True)
class ParsedSequence:
    residues: list[str]
    chain_index: list[int]
    chain_lengths: list[int]


def parse_sequence_with_chains(
    *,
    sequence: str,
    chain_separator: str,
    stage: str,
    location: str,
    target_id: str,
) -> ParsedSequence:
    separator = str(chain_separator)
    if len(separator) != 1:
        raise_error(stage, location, "chain_separator deve ter 1 caractere", impact="1", examples=[separator])
    raw = str(sequence).strip().upper().replace(" ", "")
    if not raw:
        raise_error(stage, location, "sequence vazia para parse multicadeia", impact="1", examples=[target_id])
    chunks = raw.split(separator)
    if any(chunk == "" for chunk in chunks):
        raise_error(
            stage,
            location,
            "sequence multicadeia invalida (cadeia vazia entre separadores)",
            impact="1",
            examples=[target_id],
        )
    residues: list[str] = []
    chain_index: list[int] = []
    chain_lengths: list[int] = []
    for cidx, chunk in enumerate(chunks):
        normalized = chunk.replace("T", "U")
        bad = sorted({base for base in normalized if base not in {"A", "C", "G", "U"}})
        if bad:
            raise_error(
                stage,
                location,
                "base invalida na sequencia multicadeia",
                impact=str(len(bad)),
                examples=[f"{target_id}:{item}" for item in bad[:8]],
            )
        chain_lengths.append(len(normalized))
        for base in normalized:
            residues.append(base)
            chain_index.append(int(cidx))
    if not residues:
        raise_error(stage, location, "sequence sem residuos apos parse multicadeia", impact="1", examples=[target_id])
    return ParsedSequence(residues=residues, chain_index=chain_index, chain_lengths=chain_lengths)
