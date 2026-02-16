from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import polars as pl

from ..errors import raise_error
from ..io_tables import read_table, write_table


def _normalize_seq(seq: str, *, stage: str, location: str, target_id: str) -> str:
    raw = str(seq or "").strip().upper().replace("T", "U")
    cleaned = "".join(ch for ch in raw if ch not in {" ", "\t", "\n", "\r"})
    if not cleaned:
        raise_error(stage, location, "sequencia vazia", impact="1", examples=[target_id])
    bad = sorted({ch for ch in cleaned if ch not in {"A", "C", "G", "U", "|"}})
    if bad:
        raise_error(stage, location, "sequencia contem simbolos invalidos", impact=str(len(bad)), examples=[f"{target_id}:{''.join(bad[:8])}"])
    return cleaned


def _write_fasta_for_target(*, fasta_path: Path, target_id: str, sequence: str, chain_separator: str, stage: str, location: str) -> list[str]:
    seq = _normalize_seq(sequence, stage=stage, location=location, target_id=target_id)
    parts = [p for p in seq.split(chain_separator) if p]
    if not parts:
        raise_error(stage, location, "sequencia sem nucleotideos apos split de cadeias", impact="1", examples=[target_id])
    chain_ids: list[str] = []
    lines: list[str] = []
    for idx, part in enumerate(parts):
        chain_id = chr(ord("A") + idx)
        chain_ids.append(chain_id)
        header = f">rna|name={target_id}_{chain_id}"
        lines.append(header)
        lines.append(part)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return chain_ids


def _base_from_resname(resname: str) -> str:
    name = str(resname or "").strip().upper()
    mapping = {
        "A": "A",
        "C": "C",
        "G": "G",
        "U": "U",
        "DA": "A",
        "DC": "C",
        "DG": "G",
        "DT": "U",
        "DU": "U",
    }
    return mapping.get(name, name[:1] if name else "")


def _extract_c1_from_cif(
    *,
    cif_path: Path,
    expected_chains: list[str],
    expected_seq_by_chain: list[str],
    stage: str,
    location: str,
    target_id: str,
) -> list[tuple[str, float, float, float]]:
    try:
        import gemmi  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "dependencia gemmi ausente para ler mmcif do chai1", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    try:
        st = gemmi.read_structure(str(cif_path))
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "falha ao ler mmcif do chai1", impact="1", examples=[f"{cif_path}", f"{type(exc).__name__}:{exc}"])
    if len(st) == 0:
        raise_error(stage, location, "mmcif sem models", impact="1", examples=[str(cif_path)])
    model = st[0]
    chain_map = {str(ch.name): ch for ch in model}
    missing = [c for c in expected_chains if c not in chain_map]
    if missing:
        raise_error(stage, location, "cadeias esperadas ausentes no mmcif do chai1", impact=str(len(missing)), examples=[target_id, *missing[:6]])

    out: list[tuple[str, float, float, float]] = []
    for chain_id, expected_seq in zip(expected_chains, expected_seq_by_chain, strict=True):
        chain = chain_map[chain_id]
        residues = [res for res in chain if res is not None]
        if len(residues) != len(expected_seq):
            raise_error(
                stage,
                location,
                "numero de residuos do chai1 diverge do esperado para a cadeia",
                impact="1",
                examples=[f"{target_id}:{chain_id}", f"expected={len(expected_seq)}", f"actual={len(residues)}"],
            )
        for idx, (res, expected_base) in enumerate(zip(residues, expected_seq, strict=True), start=1):
            base = _base_from_resname(res.name)
            if base != expected_base:
                raise_error(
                    stage,
                    location,
                    "resname do chai1 nao bate com sequencia esperada",
                    impact="1",
                    examples=[f"{target_id}:{chain_id}:{idx}", f"expected={expected_base}", f"actual={res.name}"],
                )
            atom = None
            for cand in ("C1'", "C1*", "C1"):
                a = res.find_atom(cand, "\0")
                if a is not None:
                    atom = a
                    break
            if atom is None:
                raise_error(stage, location, "atomo C1' ausente no chai1", impact="1", examples=[f"{target_id}:{chain_id}:{idx}", res.name])
            pos = atom.pos
            out.append((base, float(pos.x), float(pos.y), float(pos.z)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="chai1_runner")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-models", required=True, type=int)
    ap.add_argument("--chain-separator", default="|")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default=None)
    ap.add_argument("--use-esm", action="store_true")
    ap.add_argument("--no-esm", dest="use_esm", action="store_false")
    ap.set_defaults(use_esm=True)
    args = ap.parse_args(argv)

    stage = "CHAI1_RUNNER"
    location = "src/rna3d_local/runners/chai1.py:main"

    model_dir = Path(args.model_dir).resolve()
    targets_path = Path(args.targets).resolve()
    out_path = Path(args.out).resolve()
    n_models = int(args.n_models)
    if n_models <= 0:
        raise_error(stage, location, "n_models invalido", impact="1", examples=[str(args.n_models)])

    # Point chai-lab downloads resolver to the offline artifacts directory.
    os.environ["CHAI_DOWNLOADS_DIR"] = str(model_dir)

    try:
        from chai_lab.chai1 import run_inference  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "dependencia chai_lab ausente/invalid para runner chai1", impact="1", examples=[f"{type(exc).__name__}:{exc}"])

    targets = read_table(targets_path, stage=stage, location=location)
    if "target_id" not in targets.columns or "sequence" not in targets.columns:
        raise_error(stage, location, "targets schema invalido (faltam colunas)", impact="1", examples=["target_id", "sequence"])
    rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="chai1_infer_") as tmpdir:
        tmp_root = Path(tmpdir)
        for target_id, sequence in targets.select("target_id", "sequence").iter_rows():
            tid = str(target_id)
            seq = _normalize_seq(str(sequence), stage=stage, location=location, target_id=tid)
            parts = [p for p in seq.split(str(args.chain_separator)) if p]
            expected_chains = [chr(ord("A") + i) for i in range(len(parts))]
            expected_seq_by_chain = parts

            fasta = tmp_root / f"{tid}.fasta"
            _write_fasta_for_target(
                fasta_path=fasta,
                target_id=tid,
                sequence=seq,
                chain_separator=str(args.chain_separator),
                stage=stage,
                location=location,
            )
            out_dir = tmp_root / f"{tid}_out"
            if out_dir.exists():
                raise_error(stage, location, "output_dir temporario ja existe", impact="1", examples=[str(out_dir)])
            try:
                cand = run_inference(
                    fasta_file=fasta,
                    output_dir=out_dir,
                    use_esm_embeddings=bool(args.use_esm),
                    use_msa_server=False,
                    num_trunk_samples=1,
                    num_diffn_samples=n_models,
                    seed=int(args.seed),
                    device=None if args.device is None else str(args.device),
                    low_memory=True,
                )
            except Exception as exc:  # noqa: BLE001
                raise_error(stage, location, "chai1 falhou na inferencia", impact="1", examples=[tid, f"{type(exc).__name__}:{exc}"])

            cif_paths = [Path(p) for p in getattr(cand, "cif_paths", [])]
            if len(cif_paths) < n_models:
                raise_error(stage, location, "chai1 nao gerou candidatos suficientes", impact="1", examples=[tid, f"expected={n_models}", f"actual={len(cif_paths)}"])

            # Use the first N candidates (already includes diffusion samples).
            for model_id, cif_path in enumerate(cif_paths[:n_models], start=1):
                coords = _extract_c1_from_cif(
                    cif_path=cif_path,
                    expected_chains=expected_chains,
                    expected_seq_by_chain=expected_seq_by_chain,
                    stage=stage,
                    location=location,
                    target_id=tid,
                )
                for resid, (base, x, y, z) in enumerate(coords, start=1):
                    rows.append(
                        {
                            "target_id": tid,
                            "model_id": int(model_id),
                            "resid": int(resid),
                            "resname": base,
                            "x": float(x),
                            "y": float(y),
                            "z": float(z),
                            "source": "chai1",
                            "confidence": 0.78,
                        }
                    )

    df = pl.DataFrame(rows)
    write_table(df, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

