from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import torch

from ..contracts import require_columns
from ..errors import raise_error
from ..se3.sequence_parser import parse_sequence_with_chains

_NUC_TO_INT = {"A": 0, "C": 1, "G": 2, "U": 3}
_SHORT_MAX_LEN = 350
_MEDIUM_MAX_LEN = 600
_MSA_CAP_MEDIUM = 64
_MSA_CAP_LONG = 32


@dataclass(frozen=True)
class MsaCovTarget:
    target_id: str
    sequence: str
    cov_marginal: torch.Tensor
    pair_src: torch.Tensor
    pair_dst: torch.Tensor
    pair_prob: torch.Tensor


def _cache_path(*, cache_dir: Path, backend: str, sequence: str) -> Path:
    digest = hashlib.sha256(f"{backend}:{sequence}".encode("utf-8")).hexdigest()
    return cache_dir / f"{backend}_{digest}.json"


def _load_cache(path: Path) -> list[tuple[int, int, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[tuple[int, int, float]] = []
    for item in payload.get("pairs", []):
        rows.append((int(item["i"]), int(item["j"]), float(item["p"])))
    return rows


def _save_cache(path: Path, pairs: list[tuple[int, int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pairs": [{"i": int(i_1), "j": int(j_1), "p": float(prob)} for i_1, j_1, prob in pairs]}
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def _map_target_alignment_to_query(qaln: str, taln: str, query_length: int) -> np.ndarray:
    mapped = np.full((query_length,), -1, dtype=np.int16)
    q_index = 0
    for q_char, t_char in zip(qaln, taln):
        if q_char == "-":
            continue
        if q_index >= query_length:
            break
        if t_char in {"A", "C", "G", "U", "T"}:
            mapped[q_index] = _NUC_TO_INT[t_char.replace("T", "U")]
        q_index += 1
    return mapped


def _dynamic_msa_cap(*, sequence_length: int, configured_max_msa_sequences: int) -> int:
    seq_len = int(sequence_length)
    if seq_len > _MEDIUM_MAX_LEN:
        return int(min(int(configured_max_msa_sequences), int(_MSA_CAP_LONG)))
    if seq_len > _SHORT_MAX_LEN:
        return int(min(int(configured_max_msa_sequences), int(_MSA_CAP_MEDIUM)))
    return int(configured_max_msa_sequences)


def _normalized_hamming_distance(left: np.ndarray, right: np.ndarray) -> float:
    mask = (left >= 0) & (right >= 0)
    if not np.any(mask):
        return 0.0
    return float(np.mean(left[mask] != right[mask]))


def _select_hamming_diverse_alignment(*, aligned: np.ndarray, max_sequences: int) -> np.ndarray:
    if aligned.ndim != 2:
        return aligned
    n_rows = int(aligned.shape[0])
    limit = int(max_sequences)
    if limit >= n_rows:
        return aligned
    if limit <= 0:
        return aligned[:0, :]
    if limit == 1:
        return aligned[:1, :]

    selected: list[int] = [0]
    remaining: list[int] = list(range(1, n_rows))

    seed_idx = max(remaining, key=lambda idx: (_normalized_hamming_distance(aligned[idx], aligned[0]), -idx))
    selected.append(int(seed_idx))
    remaining.remove(int(seed_idx))

    while len(selected) < limit and remaining:
        best_idx = remaining[0]
        best_key = (-1.0, -1.0, -best_idx)
        for idx in remaining:
            distances = [_normalized_hamming_distance(aligned[idx], aligned[sel]) for sel in selected]
            min_dist = float(min(distances))
            mean_dist = float(sum(distances) / max(1, len(distances)))
            key = (min_dist, mean_dist, -idx)
            if key > best_key:
                best_key = key
                best_idx = idx
        selected.append(int(best_idx))
        remaining.remove(int(best_idx))
    keep = np.array(selected, dtype=np.int64)
    return aligned[keep, :]


def _run_mmseqs_chain_alignments(
    *,
    mmseqs_bin: str,
    mmseqs_db: str,
    chain_sequence: str,
    query_id: str,
    max_msa_sequences: int,
    stage: str,
    location: str,
) -> np.ndarray:
    if not mmseqs_db:
        raise_error(stage, location, "msa_backend=mmseqs2 exige mmseqs_db", impact="1", examples=[mmseqs_db])
    with TemporaryDirectory(prefix="rna3d_msa_") as tmp_dir:
        tmp = Path(tmp_dir)
        query_fasta = tmp / "query.fasta"
        result_tsv = tmp / "result.tsv"
        query_fasta.write_text(f">{query_id}\n{chain_sequence}\n", encoding="utf-8")
        cmd = [
            str(mmseqs_bin),
            "easy-search",
            str(query_fasta),
            str(mmseqs_db),
            str(result_tsv),
            str(tmp / "mmseqs_tmp"),
            "--format-output",
            "query,target,qaln,taln",
            "--max-seqs",
            str(int(max_msa_sequences)),
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except FileNotFoundError:
            raise_error(stage, location, "binario mmseqs2 nao encontrado", impact="1", examples=[str(mmseqs_bin)])
        except Exception as exc:
            raise_error(
                stage,
                location,
                "falha ao executar mmseqs2 easy-search",
                impact="1",
                examples=[f"{type(exc).__name__}:{exc}"],
            )
        if proc.returncode != 0:
            stderr_txt = proc.stderr.decode("utf-8", errors="replace").strip()
            raise_error(
                stage,
                location,
                "mmseqs2 retornou erro ao gerar MSA",
                impact="1",
                examples=[stderr_txt[:240] if stderr_txt else f"returncode={proc.returncode}"],
            )
        if not result_tsv.exists():
            raise_error(stage, location, "mmseqs2 nao gerou arquivo de alinhamento", impact="1", examples=[str(result_tsv)])
        rows = [line.strip() for line in result_tsv.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        if not rows:
            raise_error(stage, location, "mmseqs2 sem alinhamentos para o alvo", impact="1", examples=[query_id])
        aligned: list[np.ndarray] = [np.array([_NUC_TO_INT[item] for item in chain_sequence], dtype=np.int16)]
        for line in rows:
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            qaln = parts[2].strip().upper().replace("T", "U")
            taln = parts[3].strip().upper().replace("T", "U")
            aligned.append(_map_target_alignment_to_query(qaln=qaln, taln=taln, query_length=len(chain_sequence)))
            if len(aligned) >= int(max_msa_sequences):
                break
        if len(aligned) < 2:
            raise_error(stage, location, "mmseqs2 com alinhamentos insuficientes para covariancia", impact="1", examples=[query_id])
        return np.stack(aligned, axis=0)


def _entropy_from_counts(counts: np.ndarray) -> np.ndarray:
    probs = counts / np.clip(counts.sum(axis=1, keepdims=True), a_min=1e-9, a_max=None)
    log_probs = np.zeros_like(probs)
    mask = probs > 0
    if np.any(mask):
        log_probs[mask] = np.log(probs[mask])
    return -np.sum(probs * log_probs, axis=1)


def _covariance_pairs_from_alignment(
    *,
    aligned: np.ndarray,
    max_cov_positions: int,
    max_cov_pairs: int,
) -> list[tuple[int, int, float]]:
    if aligned.ndim != 2:
        return []
    n_seq, length = aligned.shape
    if n_seq < 2 or length < 2:
        return []
    valid = aligned >= 0
    counts = np.zeros((length, 4), dtype=np.float64)
    for nuc in range(4):
        counts[:, nuc] = (aligned == nuc).sum(axis=0)
    coverage = valid.sum(axis=0)
    entropy = _entropy_from_counts(counts)
    candidate = np.where(coverage >= 2)[0]
    if candidate.size == 0:
        return []
    order = np.argsort(-entropy[candidate])
    keep = candidate[order][: int(max_cov_positions)]
    pair_scores: list[tuple[int, int, float]] = []
    for idx_i in range(len(keep)):
        i = int(keep[idx_i])
        vi = aligned[:, i]
        for idx_j in range(idx_i + 1, len(keep)):
            j = int(keep[idx_j])
            vj = aligned[:, j]
            mask = (vi >= 0) & (vj >= 0)
            valid_count = int(mask.sum())
            if valid_count < 2:
                continue
            pair_ids = (vi[mask] * 4) + vj[mask]
            hist = np.bincount(pair_ids, minlength=16).astype(np.float64)
            pxy = hist / float(valid_count)
            px = pxy.reshape(4, 4).sum(axis=1, keepdims=True)
            py = pxy.reshape(4, 4).sum(axis=0, keepdims=True)
            denom = np.clip(px * py, a_min=1e-12, a_max=None)
            ratio = np.where(pxy.reshape(4, 4) > 0, pxy.reshape(4, 4) / denom, 1.0)
            mi = float(np.sum(np.where(pxy.reshape(4, 4) > 0, pxy.reshape(4, 4) * np.log(ratio), 0.0)))
            # Keep tertiary/non-canonical couplings by using MI directly.
            score = mi
            score_norm = float(score / (1.0 + score))
            if score_norm > 0.0:
                pair_scores.append((i + 1, j + 1, score_norm))
    pair_scores.sort(key=lambda item: item[2], reverse=True)
    return pair_scores[: int(max_cov_pairs)]


def _pairs_to_target(
    *,
    target_id: str,
    sequence: str,
    pairs_1based: list[tuple[int, int, float]],
    stage: str,
    location: str,
) -> MsaCovTarget:
    length = len(sequence)
    marginal = torch.zeros((length,), dtype=torch.float32)
    directed_src: list[int] = []
    directed_dst: list[int] = []
    directed_prob: list[float] = []
    for i_1, j_1, prob in pairs_1based:
        if i_1 < 1 or j_1 < 1 or i_1 > length or j_1 > length or i_1 == j_1:
            raise_error(stage, location, "par de covariancia fora de intervalo", impact="1", examples=[f"{target_id}:{i_1}-{j_1}"])
        score = float(prob)
        if score < 0 or score > 1:
            raise_error(stage, location, "score de covariancia fora de [0,1]", impact="1", examples=[f"{target_id}:{i_1}-{j_1}:{score}"])
        i = int(i_1 - 1)
        j = int(j_1 - 1)
        marginal[i] = torch.maximum(marginal[i], torch.tensor(score, dtype=torch.float32))
        marginal[j] = torch.maximum(marginal[j], torch.tensor(score, dtype=torch.float32))
        directed_src.extend([i, j])
        directed_dst.extend([j, i])
        directed_prob.extend([score, score])
    if directed_src:
        pair_src = torch.tensor(directed_src, dtype=torch.long)
        pair_dst = torch.tensor(directed_dst, dtype=torch.long)
        pair_prob = torch.tensor(directed_prob, dtype=torch.float32)
    else:
        pair_src = torch.zeros((0,), dtype=torch.long)
        pair_dst = torch.zeros((0,), dtype=torch.long)
        pair_prob = torch.zeros((0,), dtype=torch.float32)
    return MsaCovTarget(
        target_id=target_id,
        sequence=sequence,
        cov_marginal=marginal,
        pair_src=pair_src,
        pair_dst=pair_dst,
        pair_prob=pair_prob,
    )


def _compute_single_target(
    *,
    target_id: str,
    sequence: str,
    backend_name: str,
    mmseqs_bin: str,
    mmseqs_db: str,
    cache_root: Path | None,
    chain_separator: str,
    max_msa_sequences: int,
    max_cov_positions: int,
    max_cov_pairs: int,
    stage: str,
    location: str,
) -> tuple[str, MsaCovTarget]:
    tid = str(target_id)
    parsed = parse_sequence_with_chains(
        sequence=str(sequence),
        chain_separator=chain_separator,
        stage=stage,
        location=location,
        target_id=tid,
    )
    joined_sequence = "".join(parsed.residues)
    effective_max_msa_sequences = _dynamic_msa_cap(
        sequence_length=len(joined_sequence),
        configured_max_msa_sequences=int(max_msa_sequences),
    )
    global_pairs: list[tuple[int, int, float]] = []
    offset = 0
    for chain_idx, chain_length in enumerate(parsed.chain_lengths):
        chain_seq = joined_sequence[offset : offset + chain_length]
        if cache_root is not None:
            cpath = _cache_path(cache_dir=cache_root, backend=f"{backend_name}_chain", sequence=chain_seq)
        else:
            cpath = None
        if cpath is not None and cpath.exists():
            chain_pairs = _load_cache(cpath)
        else:
            aligned = _run_mmseqs_chain_alignments(
                mmseqs_bin=mmseqs_bin,
                mmseqs_db=mmseqs_db,
                chain_sequence=chain_seq,
                query_id=f"{tid}_c{chain_idx}",
                max_msa_sequences=max_msa_sequences,
                stage=stage,
                location=location,
            )
            aligned_diverse = _select_hamming_diverse_alignment(
                aligned=aligned,
                max_sequences=effective_max_msa_sequences,
            )
            if int(aligned_diverse.shape[0]) < int(aligned.shape[0]):
                print(
                    f"[{stage}] [{location}] cap dinamico de MSA aplicado | impacto={int(aligned.shape[0] - aligned_diverse.shape[0])} | exemplos={tid}:L={len(joined_sequence)}:chain={chain_idx}:depth={int(aligned.shape[0])}->{int(aligned_diverse.shape[0])}",
                    file=sys.stderr,
                )
            chain_pairs = _covariance_pairs_from_alignment(
                aligned=aligned_diverse,
                max_cov_positions=max_cov_positions,
                max_cov_pairs=max_cov_pairs,
            )
            if not chain_pairs:
                raise_error(
                    stage,
                    location,
                    "mmseqs2 sem pares de covariancia utilizaveis",
                    impact="1",
                    examples=[f"{tid}:chain={chain_idx}"],
                )
            if cpath is not None:
                _save_cache(cpath, chain_pairs)
        for i_1, j_1, prob in chain_pairs:
            global_pairs.append((i_1 + offset, j_1 + offset, float(prob)))
        offset += chain_length
    target = _pairs_to_target(
        target_id=tid,
        sequence=joined_sequence,
        pairs_1based=global_pairs,
        stage=stage,
        location=location,
    )
    return tid, target


def compute_msa_covariance(
    *,
    targets: pl.DataFrame,
    backend: str,
    mmseqs_bin: str,
    mmseqs_db: str,
    cache_dir: Path | None,
    chain_separator: str,
    max_msa_sequences: int,
    max_cov_positions: int,
    max_cov_pairs: int,
    stage: str,
    location: str,
    num_workers: int = 1,
) -> dict[str, MsaCovTarget]:
    require_columns(targets, ["target_id", "sequence"], stage=stage, location=location, label="targets")
    if max_msa_sequences <= 1:
        raise_error(stage, location, "max_msa_sequences deve ser > 1", impact="1", examples=[str(max_msa_sequences)])
    if int(num_workers) <= 0:
        raise_error(stage, location, "num_workers invalido para extracao de MSA/covariancia", impact="1", examples=[str(num_workers)])
    if max_cov_positions <= 0 or max_cov_pairs <= 0:
        raise_error(
            stage,
            location,
            "max_cov_positions/max_cov_pairs invalidos",
            impact="2",
            examples=[str(max_cov_positions), str(max_cov_pairs)],
        )
    backend_name = str(backend).strip().lower()
    if backend_name not in {"mmseqs2"}:
        raise_error(stage, location, "msa_backend invalido", impact="1", examples=[backend_name])
    cache_root = None if cache_dir is None else Path(cache_dir)
    rows = [(str(target_id), str(sequence)) for target_id, sequence in targets.select("target_id", "sequence").iter_rows()]
    outputs: dict[str, MsaCovTarget] = {}
    if int(num_workers) == 1 or len(rows) <= 1:
        for tid, sequence in rows:
            key, value = _compute_single_target(
                target_id=tid,
                sequence=sequence,
                backend_name=backend_name,
                mmseqs_bin=mmseqs_bin,
                mmseqs_db=mmseqs_db,
                cache_root=cache_root,
                chain_separator=chain_separator,
                max_msa_sequences=max_msa_sequences,
                max_cov_positions=max_cov_positions,
                max_cov_pairs=max_cov_pairs,
                stage=stage,
                location=location,
            )
            outputs[key] = value
    else:
        completed: dict[str, MsaCovTarget] = {}
        with ThreadPoolExecutor(max_workers=int(num_workers), thread_name_prefix="rna3d_msa") as executor:
            futures = [
                executor.submit(
                    _compute_single_target,
                    target_id=tid,
                    sequence=sequence,
                    backend_name=backend_name,
                    mmseqs_bin=mmseqs_bin,
                    mmseqs_db=mmseqs_db,
                    cache_root=cache_root,
                    chain_separator=chain_separator,
                    max_msa_sequences=max_msa_sequences,
                    max_cov_positions=max_cov_positions,
                    max_cov_pairs=max_cov_pairs,
                    stage=stage,
                    location=location,
                )
                for tid, sequence in rows
            ]
            for future in as_completed(futures):
                key, value = future.result()
                completed[key] = value
        for tid, _ in rows:
            outputs[tid] = completed[tid]
    if not outputs:
        raise_error(stage, location, "nenhum alvo para extracao de MSA/covariancia", impact="0", examples=[])
    return outputs
