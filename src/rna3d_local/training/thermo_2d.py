from __future__ import annotations

import hashlib
import json
import re
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import torch

from ..contracts import require_columns
from ..errors import raise_error
from ..se3.sequence_parser import parse_sequence_with_chains


_BPP_LINE = re.compile(r"^\s*(\d+)\s+(\d+)\s+([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s+ubox\s*$")


@dataclass(frozen=True)
class ThermoBppTarget:
    target_id: str
    sequence: str
    paired_marginal: torch.Tensor
    pair_src: torch.Tensor
    pair_dst: torch.Tensor
    pair_prob: torch.Tensor


def _prune_chain_pairs(
    *,
    pairs_1based: list[tuple[int, int, float]],
    chain_length: int,
    min_pair_prob: float,
    max_pairs_per_node: int,
    stage: str,
    location: str,
    target_id: str,
) -> list[tuple[int, int, float]]:
    if int(chain_length) <= 0 or not pairs_1based:
        return []
    if float(min_pair_prob) <= 0.0 and int(max_pairs_per_node) <= 0:
        return pairs_1based
    if not (0.0 <= float(min_pair_prob) <= 1.0):
        raise_error(stage, location, "min_pair_prob fora de [0,1] no pruning BPP", impact="1", examples=[str(min_pair_prob)])
    if int(max_pairs_per_node) < 0:
        raise_error(stage, location, "max_pairs_per_node invalido (<0) no pruning BPP", impact="1", examples=[str(max_pairs_per_node)])
    if int(max_pairs_per_node) == 0:
        return [(i, j, float(p)) for i, j, p in pairs_1based if float(p) >= float(min_pair_prob)]

    edge_prob: dict[tuple[int, int], float] = {}
    neighbors: list[list[tuple[float, int]]] = [[] for _ in range(int(chain_length) + 1)]
    for i_1, j_1, prob in pairs_1based:
        p = float(prob)
        if p < float(min_pair_prob):
            continue
        i = int(i_1)
        j = int(j_1)
        if i < 1 or j < 1 or i > int(chain_length) or j > int(chain_length) or i == j:
            raise_error(stage, location, "par BPP fora do intervalo no pruning", impact="1", examples=[f"{target_id}:{i_1}-{j_1}"])
        key = (i, j) if i < j else (j, i)
        prev = edge_prob.get(key)
        if prev is None or p > float(prev):
            edge_prob[key] = p
        neighbors[i].append((p, j))
        neighbors[j].append((p, i))

    selected: set[tuple[int, int]] = set()
    for i in range(1, int(chain_length) + 1):
        cand = neighbors[i]
        if not cand:
            continue
        cand.sort(key=lambda item: float(item[0]), reverse=True)
        for p, j in cand[: int(max_pairs_per_node)]:
            key = (i, int(j)) if i < int(j) else (int(j), i)
            if key in edge_prob:
                selected.add(key)

    out = [(int(i), int(j), float(edge_prob[(int(i), int(j))])) for i, j in selected]
    out.sort(key=lambda item: (int(item[0]), int(item[1])))
    return out


def _target_pairs_to_tensors(
    *,
    target_id: str,
    sequence: str,
    pairs_1based: list[tuple[int, int, float]],
    stage: str,
    location: str,
) -> ThermoBppTarget:
    length = len(sequence)
    marg = torch.zeros((length,), dtype=torch.float32)
    directed_src: list[int] = []
    directed_dst: list[int] = []
    directed_prob: list[float] = []
    for i_1, j_1, prob in pairs_1based:
        if i_1 < 1 or j_1 < 1 or i_1 > length or j_1 > length or i_1 == j_1:
            raise_error(
                stage,
                location,
                "par BPP fora do intervalo valido",
                impact="1",
                examples=[f"{target_id}:{i_1}-{j_1}"],
            )
        p = float(prob)
        if p < 0 or p > 1:
            raise_error(
                stage,
                location,
                "probabilidade BPP fora de [0,1]",
                impact="1",
                examples=[f"{target_id}:{i_1}-{j_1}:{p}"],
            )
        i = int(i_1 - 1)
        j = int(j_1 - 1)
        marg[i] += p
        marg[j] += p
        directed_src.extend([i, j])
        directed_dst.extend([j, i])
        directed_prob.extend([p, p])
    max_marg = float(marg.max().item()) if length > 0 else 0.0
    if max_marg > 1.0001:
        raise_error(
            stage,
            location,
            "marginal BPP excede 1.0; entrada termodinamica inconsistente",
            impact="1",
            examples=[f"{target_id}:max_marginal={max_marg:.4f}"],
        )
    if directed_src:
        pair_src = torch.tensor(directed_src, dtype=torch.long)
        pair_dst = torch.tensor(directed_dst, dtype=torch.long)
        pair_prob = torch.tensor(directed_prob, dtype=torch.float32)
    else:
        pair_src = torch.zeros((0,), dtype=torch.long)
        pair_dst = torch.zeros((0,), dtype=torch.long)
        pair_prob = torch.zeros((0,), dtype=torch.float32)
    return ThermoBppTarget(
        target_id=target_id,
        sequence=sequence,
        paired_marginal=marg,
        pair_src=pair_src,
        pair_dst=pair_dst,
        pair_prob=pair_prob,
    )


def _run_rnafold_pairs(
    *,
    sequence: str,
    target_id: str,
    rnafold_bin: str,
    stage: str,
    location: str,
) -> list[tuple[int, int, float]]:
    with TemporaryDirectory(prefix="rna3d_bpp_") as tmp_dir:
        workdir = Path(tmp_dir)
        cmd = [str(rnafold_bin), "-p"]
        payload = f">{target_id}\n{sequence}\n"
        try:
            proc = subprocess.run(
                cmd,
                input=payload.encode("utf-8"),
                cwd=str(workdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except FileNotFoundError:
            raise_error(stage, location, "binario RNAfold nao encontrado", impact="1", examples=[str(rnafold_bin)])
        except Exception as exc:
            raise_error(
                stage,
                location,
                "falha ao executar RNAfold",
                impact="1",
                examples=[f"{type(exc).__name__}:{exc}"],
            )
        if proc.returncode != 0:
            stderr_txt = proc.stderr.decode("utf-8", errors="replace").strip()
            raise_error(
                stage,
                location,
                "RNAfold retornou erro ao gerar BPP",
                impact="1",
                examples=[stderr_txt[:240] if stderr_txt else f"returncode={proc.returncode}"],
            )
        dot_plot = workdir / "dot.ps"
        if not dot_plot.exists():
            raise_error(
                stage,
                location,
                "RNAfold nao gerou dot.ps com probabilidades BPP",
                impact="1",
                examples=[str(dot_plot)],
            )
        pairs: list[tuple[int, int, float]] = []
        for line in dot_plot.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _BPP_LINE.match(line)
            if match is None:
                continue
            i_1 = int(match.group(1))
            j_1 = int(match.group(2))
            sqrt_p = float(match.group(3))
            prob = float(sqrt_p * sqrt_p)
            pairs.append((i_1, j_1, prob))
        return pairs


def _run_linearfold_pairs(
    *,
    sequence: str,
    linearfold_bin: str,
    stage: str,
    location: str,
) -> list[tuple[int, int, float]]:
    cmd = [str(linearfold_bin), "--bpp"]
    try:
        proc = subprocess.run(
            cmd,
            input=(sequence + "\n").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        raise_error(stage, location, "binario linearfold nao encontrado", impact="1", examples=[str(linearfold_bin)])
    except Exception as exc:
        raise_error(
            stage,
            location,
            "falha ao executar linearfold --bpp",
            impact="1",
            examples=[f"{type(exc).__name__}:{exc}"],
        )
    if proc.returncode != 0:
        stderr_txt = proc.stderr.decode("utf-8", errors="replace").strip()
        raise_error(
            stage,
            location,
            "linearfold retornou erro ao gerar BPP",
            impact="1",
            examples=[stderr_txt[:240] if stderr_txt else f"returncode={proc.returncode}"],
        )
    pairs: list[tuple[int, int, float]] = []
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        parts = clean.split()
        if len(parts) < 3:
            continue
        try:
            i_1 = int(parts[0])
            j_1 = int(parts[1])
            prob = float(parts[2])
        except Exception:
            continue
        pairs.append((i_1, j_1, prob))
    return pairs


def _run_viennarna_pairs(
    *,
    sequence: str,
    stage: str,
    location: str,
) -> list[tuple[int, int, float]]:
    try:
        import RNA  # type: ignore
    except Exception as exc:
        raise_error(
            stage,
            location,
            "backend viennarna indisponivel (modulo RNA ausente)",
            impact="1",
            examples=[f"{type(exc).__name__}:{exc}"],
        )
    try:
        fc = RNA.fold_compound(sequence)
        fc.pf()
        bpp = fc.bpp()
    except Exception as exc:
        raise_error(
            stage,
            location,
            "falha ao executar ViennaRNA (fold_compound/pf)",
            impact="1",
            examples=[f"{type(exc).__name__}:{exc}"],
        )
    pairs: list[tuple[int, int, float]] = []
    length = int(len(sequence))
    for i_1 in range(1, length + 1):
        for j_1 in range(i_1 + 1, length + 1):
            prob = float(bpp[i_1][j_1])
            if prob <= 0.0:
                continue
            pairs.append((int(i_1), int(j_1), prob))
    return pairs


def _cache_path(*, cache_dir: Path, backend: str, sequence: str) -> Path:
    digest = hashlib.sha256(f"{backend}:{sequence}".encode("utf-8")).hexdigest()
    return cache_dir / f"{backend}_{digest}.json"


def _load_cache(path: Path) -> list[tuple[int, int, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[tuple[int, int, float]] = []
    for item in payload.get("pairs", []):
        out.append((int(item["i"]), int(item["j"]), float(item["p"])))
    return out


def _save_cache(path: Path, pairs: list[tuple[int, int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pairs": [{"i": int(i_1), "j": int(j_1), "p": float(prob)} for i_1, j_1, prob in pairs]}
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def _compute_single_target(
    *,
    target_id: str,
    sequence: str,
    backend_name: str,
    rnafold_bin: str,
    linearfold_bin: str,
    cache_root: Path | None,
    chain_separator: str,
    min_pair_prob: float,
    max_pairs_per_node: int,
    stage: str,
    location: str,
) -> tuple[str, ThermoBppTarget]:
    tid = str(target_id)
    parsed = parse_sequence_with_chains(
        sequence=str(sequence),
        chain_separator=chain_separator,
        stage=stage,
        location=location,
        target_id=tid,
    )
    seq = "".join(parsed.residues)
    pairs: list[tuple[int, int, float]] = []
    offset = 0
    for chain_idx, chain_length in enumerate(parsed.chain_lengths):
        chain_seq = seq[offset : offset + chain_length]
        cache_tag = f"{backend_name}_chain_pmin{float(min_pair_prob):.4f}_k{int(max_pairs_per_node)}"
        cpath = None if cache_root is None else _cache_path(cache_dir=cache_root, backend=cache_tag, sequence=chain_seq)
        chain_pairs: list[tuple[int, int, float]]
        if cpath is not None and cpath.exists():
            chain_pairs = _load_cache(cpath)
        else:
            if backend_name == "rnafold":
                chain_pairs = _run_rnafold_pairs(
                    sequence=chain_seq,
                    target_id=f"{tid}_c{chain_idx}",
                    rnafold_bin=rnafold_bin,
                    stage=stage,
                    location=location,
                )
            elif backend_name == "linearfold":
                chain_pairs = _run_linearfold_pairs(
                    sequence=chain_seq,
                    linearfold_bin=linearfold_bin,
                    stage=stage,
                    location=location,
                )
            elif backend_name == "viennarna":
                chain_pairs = _run_viennarna_pairs(
                    sequence=chain_seq,
                    stage=stage,
                    location=location,
                )
            else:
                raise_error(stage, location, "backend termoquimico BPP invalido", impact="1", examples=[backend_name])
            if float(min_pair_prob) > 0.0 or int(max_pairs_per_node) > 0:
                chain_pairs = _prune_chain_pairs(
                    pairs_1based=chain_pairs,
                    chain_length=int(chain_length),
                    min_pair_prob=float(min_pair_prob),
                    max_pairs_per_node=int(max_pairs_per_node),
                    stage=stage,
                    location=location,
                    target_id=tid,
                )
            if cpath is not None:
                _save_cache(cpath, chain_pairs)
        for i_1, j_1, prob in chain_pairs:
            pairs.append((int(i_1 + offset), int(j_1 + offset), float(prob)))
        offset += chain_length
    target = _target_pairs_to_tensors(
        target_id=tid,
        sequence=seq,
        pairs_1based=pairs,
        stage=stage,
        location=location,
    )
    return tid, target


def compute_thermo_bpp(
    *,
    targets: pl.DataFrame,
    backend: str,
    rnafold_bin: str,
    linearfold_bin: str,
    cache_dir: Path | None,
    chain_separator: str,
    stage: str,
    location: str,
    num_workers: int = 1,
    min_pair_prob: float = 0.0,
    max_pairs_per_node: int = 0,
) -> dict[str, ThermoBppTarget]:
    require_columns(targets, ["target_id", "sequence"], stage=stage, location=location, label="targets")
    backend_name = str(backend).strip().lower()
    if backend_name not in {"rnafold", "linearfold", "viennarna"}:
        raise_error(stage, location, "backend termoquimico BPP invalido", impact="1", examples=[backend_name])
    if int(num_workers) <= 0:
        raise_error(stage, location, "num_workers invalido para extracao BPP", impact="1", examples=[str(num_workers)])
    if not (0.0 <= float(min_pair_prob) <= 1.0):
        raise_error(stage, location, "min_pair_prob fora de [0,1] para pruning BPP", impact="1", examples=[str(min_pair_prob)])
    if int(max_pairs_per_node) < 0:
        raise_error(stage, location, "max_pairs_per_node invalido (<0) para pruning BPP", impact="1", examples=[str(max_pairs_per_node)])
    cache_root = None if cache_dir is None else Path(cache_dir)
    rows = [(str(target_id), str(sequence)) for target_id, sequence in targets.select("target_id", "sequence").iter_rows()]
    out: dict[str, ThermoBppTarget] = {}
    if int(num_workers) == 1 or len(rows) <= 1:
        for tid, sequence in rows:
            key, value = _compute_single_target(
                target_id=tid,
                sequence=sequence,
                backend_name=backend_name,
                rnafold_bin=rnafold_bin,
                linearfold_bin=linearfold_bin,
                cache_root=cache_root,
                chain_separator=chain_separator,
                min_pair_prob=float(min_pair_prob),
                max_pairs_per_node=int(max_pairs_per_node),
                stage=stage,
                location=location,
            )
            out[key] = value
    else:
        completed: dict[str, ThermoBppTarget] = {}
        with ThreadPoolExecutor(max_workers=int(num_workers), thread_name_prefix="rna3d_thermo") as executor:
            futures = [
                executor.submit(
                    _compute_single_target,
                    target_id=tid,
                    sequence=sequence,
                    backend_name=backend_name,
                    rnafold_bin=rnafold_bin,
                    linearfold_bin=linearfold_bin,
                    cache_root=cache_root,
                    chain_separator=chain_separator,
                    min_pair_prob=float(min_pair_prob),
                    max_pairs_per_node=int(max_pairs_per_node),
                    stage=stage,
                    location=location,
                )
                for tid, sequence in rows
            ]
            for future in as_completed(futures):
                key, value = future.result()
                completed[key] = value
        for tid, _ in rows:
            out[tid] = completed[tid]
    if not out:
        raise_error(stage, location, "nenhum alvo disponivel para extracao BPP", impact="0", examples=[])
    return out
