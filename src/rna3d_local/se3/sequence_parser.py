from __future__ import annotations

from dataclasses import dataclass

from ..errors import raise_error


@dataclass(frozen=True)
class ParsedSequence:
    residues: list[str]
    chain_index: list[int]
    chain_lengths: list[int]
    residue_position_index_1d: list[int]


def parse_sequence_with_chains(
    *,
    sequence: str,
    chain_separator: str,
    stage: str,
    location: str,
    target_id: str,
    chain_break_offset_1d: int = 1000,
) -> ParsedSequence:
    separator = str(chain_separator)
    if len(separator) != 1:
        raise_error(stage, location, "chain_separator deve ter 1 caractere", impact="1", examples=[separator])
    raw = str(sequence).strip().upper().replace(" ", "")
    if not raw:
        raise_error(stage, location, "sequence vazia para parse multicadeia", impact="1", examples=[target_id])
    if int(chain_break_offset_1d) <= 0:
        raise_error(stage, location, "chain_break_offset_1d deve ser > 0", impact="1", examples=[str(chain_break_offset_1d)])
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
    residue_position_index_1d: list[int] = []
    position_cursor = 0
    for cidx, chunk in enumerate(chunks):
        normalized = chunk.replace("T", "U")
        bad = sorted({base for base in normalized if base not in {"A", "C", "G", "U", "N"}})
        if bad:
            raise_error(
                stage,
                location,
                "base invalida na sequencia multicadeia",
                impact=str(len(bad)),
                examples=[f"{target_id}:{item}" for item in bad[:8]],
            )
        chain_lengths.append(len(normalized))
        if cidx > 0:
            position_cursor += int(chain_break_offset_1d)
        for base in normalized:
            residues.append(base)
            chain_index.append(int(cidx))
            residue_position_index_1d.append(int(position_cursor))
            position_cursor += 1
    if not residues:
        raise_error(stage, location, "sequence sem residuos apos parse multicadeia", impact="1", examples=[target_id])
    return ParsedSequence(
        residues=residues,
        chain_index=chain_index,
        chain_lengths=chain_lengths,
        residue_position_index_1d=residue_position_index_1d,
    )
