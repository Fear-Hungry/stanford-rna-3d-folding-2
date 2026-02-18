from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import polars as pl

from ..errors import raise_error
from ..io_tables import read_table, write_table


def _normalize_seq(seq: str, *, stage: str, location: str, target_id: str, chain_separator: str) -> list[str]:
    raw = str(seq or "").strip().upper().replace("T", "U")
    cleaned = "".join(ch for ch in raw if ch not in {" ", "\t", "\n", "\r"})
    if not cleaned:
        raise_error(stage, location, "sequencia vazia", impact="1", examples=[target_id])
    bad = sorted({ch for ch in cleaned if ch not in {"A", "C", "G", "U", chain_separator}})
    if bad:
        raise_error(stage, location, "sequencia contem simbolos invalidos", impact=str(len(bad)), examples=[f"{target_id}:{''.join(bad[:8])}"])
    parts = [p for p in cleaned.split(chain_separator) if p]
    if not parts:
        raise_error(stage, location, "sequencia sem nucleotideos apos split de cadeias", impact="1", examples=[target_id])
    return parts


def _write_boltz_fasta(
    *,
    path: Path,
    target_id: str,
    seq_parts: list[str],
    ligand_smiles: str,
) -> list[str]:
    chain_ids: list[str] = []
    lines: list[str] = []
    for idx, part in enumerate(seq_parts):
        chain_id = chr(ord("A") + idx)
        chain_ids.append(chain_id)
        lines.append(f">{chain_id}|rna")
        lines.append(part)
    if ligand_smiles.strip():
        lines.append(">L|smiles")
        lines.append(ligand_smiles.strip())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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


def _normalize_plddt(raw: float, *, stage: str, location: str, target_id: str) -> float:
    value = float(raw)
    if not value == value:  # NaN
        raise_error(stage, location, "pLDDT do boltz1 e NaN", impact="1", examples=[target_id])
    if value < 0.0:
        raise_error(stage, location, "pLDDT do boltz1 negativo", impact="1", examples=[target_id, str(raw)])
    if value > 1.0:
        if value <= 100.0:
            value = value / 100.0
        else:
            raise_error(stage, location, "pLDDT do boltz1 fora do intervalo esperado", impact="1", examples=[target_id, str(raw)])
    if value < 0.0 or value > 1.0:
        raise_error(stage, location, "pLDDT do boltz1 apos normalizacao fora de [0,1]", impact="1", examples=[target_id, str(value)])
    return float(value)


def _extract_c1_from_pdb(
    *,
    pdb_path: Path,
    chain_order: list[str],
    expected_seq_by_chain: list[str],
    stage: str,
    location: str,
    target_id: str,
) -> tuple[list[tuple[str, float, float, float]], float]:
    text = pdb_path.read_text(encoding="utf-8", errors="replace").splitlines()
    residue_order: list[tuple[str, str, str]] = []
    residue_info: dict[tuple[str, str, str], dict[str, object]] = {}
    for line in text:
        if not line.startswith("ATOM"):
            continue
        if len(line) < 54:
            continue
        atom_name = line[12:16].strip()
        if atom_name not in {"C1'", "C1*", "C1"}:
            continue
        resname = line[17:20].strip()
        chain_id = line[21:22].strip() or "?"
        resseq = line[22:26].strip()
        icode = line[26:27].strip()
        key = (chain_id, resseq, icode)
        if key not in residue_info:
            residue_order.append(key)
            residue_info[key] = {"resname": resname, "c1": None}
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            plddt = float(line[60:66].strip())
        except Exception:
            continue
        residue_info[key]["c1"] = (x, y, z)
        residue_info[key]["plddt"] = float(plddt)
        residue_info[key]["resname"] = resname

    out: list[tuple[str, float, float, float]] = []
    plddt_values: list[float] = []
    for chain_id, expected_seq in zip(chain_order, expected_seq_by_chain, strict=True):
        keys = [k for k in residue_order if k[0] == chain_id]
        if len(keys) != len(expected_seq):
            raise_error(
                stage,
                location,
                "numero de residuos do boltz diverge do esperado para a cadeia",
                impact="1",
                examples=[f"{target_id}:{chain_id}", f"expected={len(expected_seq)}", f"actual={len(keys)}", str(pdb_path)],
            )
        for idx, (k, expected_base) in enumerate(zip(keys, expected_seq, strict=True), start=1):
            info = residue_info.get(k)
            if info is None or info.get("c1") is None:
                raise_error(stage, location, "atomo C1' ausente no boltz", impact="1", examples=[f"{target_id}:{chain_id}:{idx}", str(pdb_path)])
            if "plddt" not in info:
                raise_error(stage, location, "pLDDT ausente no C1' do boltz", impact="1", examples=[f"{target_id}:{chain_id}:{idx}", str(pdb_path)])
            base = _base_from_resname(str(info.get("resname", "")))
            if base != expected_base:
                raise_error(
                    stage,
                    location,
                    "resname do boltz nao bate com sequencia esperada",
                    impact="1",
                    examples=[f"{target_id}:{chain_id}:{idx}", f"expected={expected_base}", f"actual={info.get('resname')}"],
                )
            x, y, z = info["c1"]  # type: ignore[assignment]
            plddt_values.append(float(info["plddt"]))
            out.append((base, float(x), float(y), float(z)))
    if not plddt_values:
        raise_error(stage, location, "boltz1 sem pLDDT para calcular confidence", impact="1", examples=[target_id, str(pdb_path)])
    confidence = _normalize_plddt(
        sum(plddt_values) / float(len(plddt_values)),
        stage=stage,
        location=location,
        target_id=target_id,
    )
    return out, confidence


def _ensure_boltz_cache(
    *,
    model_dir: Path,
    cache_dir: Path,
    stage: str,
    location: str,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    required = ["boltz1_conf.ckpt", "ccd.pkl"]
    missing = [str(model_dir / rel) for rel in required if not (model_dir / rel).exists()]
    if missing:
        raise_error(stage, location, "artefatos boltz1 ausentes no model_dir", impact=str(len(missing)), examples=missing[:8])
    for rel in required:
        src = model_dir / rel
        dst = cache_dir / rel
        if dst.exists():
            continue
        try:
            shutil.copy2(src, dst)
        except Exception as exc:  # noqa: BLE001
            raise_error(stage, location, "falha ao copiar artefato boltz para cache", impact="1", examples=[rel, f"{type(exc).__name__}:{exc}"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="boltz1_runner")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-models", required=True, type=int)
    ap.add_argument("--chain-separator", default="|")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--sampling-steps", type=int, default=120)
    ap.add_argument("--recycling-steps", type=int, default=2)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args(argv)

    stage = "BOLTZ1_RUNNER"
    location = "src/rna3d_local/runners/boltz1.py:main"

    model_dir = Path(args.model_dir).resolve()
    targets_path = Path(args.targets).resolve()
    out_path = Path(args.out).resolve()
    n_models = int(args.n_models)
    if n_models <= 0:
        raise_error(stage, location, "n_models invalido", impact="1", examples=[str(args.n_models)])

    try:
        from boltz.main import predict as boltz_predict  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "dependencia boltz ausente/invalid para runner boltz1", impact="1", examples=[f"{type(exc).__name__}:{exc}"])

    targets = read_table(targets_path, stage=stage, location=location)
    if "target_id" not in targets.columns or "sequence" not in targets.columns:
        raise_error(stage, location, "targets schema invalido (faltam colunas)", impact="1", examples=["target_id", "sequence"])
    if "ligand_SMILES" not in targets.columns:
        targets = targets.with_columns(pl.lit("").alias("ligand_SMILES"))

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="boltz1_infer_") as tmpdir:
        tmp_root = Path(tmpdir)
        cache_dir = tmp_root / "boltz_cache"
        _ensure_boltz_cache(model_dir=model_dir, cache_dir=cache_dir, stage=stage, location=location)
        os.environ["BOLTZ_CACHE"] = str(cache_dir)

        for tid, seq, smiles in targets.select("target_id", "sequence", "ligand_SMILES").iter_rows():
            target_id = str(tid)
            seq_parts = _normalize_seq(str(seq), stage=stage, location=location, target_id=target_id, chain_separator=str(args.chain_separator))
            chain_order = [chr(ord("A") + i) for i in range(len(seq_parts))]
            fasta_path = tmp_root / f"{target_id}.fasta"
            _write_boltz_fasta(path=fasta_path, target_id=target_id, seq_parts=seq_parts, ligand_smiles=str(smiles))

            out_dir = tmp_root / f"{target_id}_out"
            if out_dir.exists():
                raise_error(stage, location, "output_dir temporario ja existe", impact="1", examples=[str(out_dir)])
            try:
                boltz_predict(
                    data=str(fasta_path),
                    out_dir=str(out_dir),
                    cache=str(cache_dir),
                    checkpoint=str(cache_dir / "boltz1_conf.ckpt"),
                    devices=1,
                    accelerator=str(args.device),
                    recycling_steps=int(args.recycling_steps),
                    sampling_steps=int(args.sampling_steps),
                    diffusion_samples=int(n_models),
                    output_format="pdb",
                    num_workers=0,
                    override=False,
                    seed=int(args.seed),
                    use_msa_server=False,
                    model="boltz1",
                    preprocessing_threads=1,
                )
            except Exception as exc:  # noqa: BLE001
                raise_error(stage, location, "boltz predict falhou", impact="1", examples=[target_id, f"{type(exc).__name__}:{exc}"])

            results_dir = out_dir / f"boltz_results_{fasta_path.stem}"
            if not results_dir.exists():
                raise_error(stage, location, "diretorio de resultados do boltz ausente", impact="1", examples=[str(results_dir)])
            record_dirs = [p for p in results_dir.iterdir() if p.is_dir()]
            if not record_dirs:
                raise_error(stage, location, "boltz nao gerou diretorio de record", impact="1", examples=[str(results_dir)])
            if len(record_dirs) != 1:
                raise_error(stage, location, "boltz gerou multiplos records (nao suportado)", impact=str(len(record_dirs)), examples=[p.name for p in record_dirs[:6]])
            record_dir = record_dirs[0]
            pdbs = sorted(record_dir.glob("*.pdb"))
            if len(pdbs) < n_models:
                raise_error(stage, location, "boltz nao gerou pdbs suficientes", impact="1", examples=[target_id, f"expected={n_models}", f"actual={len(pdbs)}"])

            # Prefer *_model_0.. sorted by model index if present.
            def _rank_key(p: Path) -> tuple[int, str]:
                name = p.name
                if "_model_" in name:
                    try:
                        idx = int(name.split("_model_")[-1].split(".")[0])
                        return (idx, name)
                    except Exception:
                        pass
                return (10_000, name)

            pdbs = sorted(pdbs, key=_rank_key)[:n_models]
            for model_id, pdb_path in enumerate(pdbs, start=1):
                coords, confidence = _extract_c1_from_pdb(
                    pdb_path=pdb_path,
                    chain_order=chain_order,
                    expected_seq_by_chain=seq_parts,
                    stage=stage,
                    location=location,
                    target_id=target_id,
                )
                for resid, (base, x, y, z) in enumerate(coords, start=1):
                    rows.append(
                        {
                            "target_id": target_id,
                            "model_id": int(model_id),
                            "resid": int(resid),
                            "resname": base,
                            "x": float(x),
                            "y": float(y),
                            "z": float(z),
                            "source": "boltz1",
                            "confidence": float(confidence),
                        }
                    )

    df = pl.DataFrame(rows)
    write_table(df, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
