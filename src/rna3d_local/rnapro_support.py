from __future__ import annotations

import gzip
import pickle
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .errors import raise_error
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class RnaProSupportFilesResult:
    model_dir: Path
    manifest_path: Path


def _download_to_file(
    url: str,
    *,
    out_path: Path,
    stage: str,
    location: str,
    timeout_seconds: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rna3d_local/0.1"}, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=float(timeout_seconds)) as resp:  # noqa: S310
            status = getattr(resp, "status", None)
            if status is not None and int(status) >= 400:
                raise RuntimeError(f"http_status={int(status)}")
            with tmp.open("wb") as f:
                shutil.copyfileobj(resp, f)
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "falha ao baixar arquivo", impact="1", examples=[url, f"{type(exc).__name__}:{exc}"])
    tmp.replace(out_path)


def _extract_components_blocks(
    *,
    components_cif_gz: Path,
    out_components_cif: Path,
    codes: list[str],
    stage: str,
    location: str,
) -> dict[str, int]:
    want = {str(c).strip().upper() for c in codes if str(c).strip()}
    if not want:
        raise_error(stage, location, "codes vazio para extracao CCD", impact="1", examples=["A,C,G,U"])
    found: set[str] = set()
    out_components_cif.parent.mkdir(parents=True, exist_ok=True)
    cur = None
    n_lines = 0
    with gzip.open(components_cif_gz, "rt", encoding="utf-8", errors="replace") as f_in, out_components_cif.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            n_lines += 1
            if line.startswith("data_"):
                code = line.strip()[5:].strip().upper()
                cur = code if code in want else None
                if cur is not None:
                    found.add(cur)
                    f_out.write(line)
                continue
            if cur is not None:
                f_out.write(line)
    missing = sorted(want - found)
    if missing:
        raise_error(stage, location, "codes ausentes no components.cif.gz", impact=str(len(missing)), examples=missing[:8])
    return {"n_lines_read": int(n_lines), "n_blocks": int(len(found))}


def _build_minimal_rdkit_mol_pkl(
    *,
    components_cif: Path,
    out_pkl: Path,
    codes: list[str],
    stage: str,
    location: str,
) -> dict[str, int]:
    try:
        import gemmi  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "dependencia gemmi ausente para CCD cache", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    try:
        from pdbeccdutils.core import ccd_reader  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "dependencia pdbeccdutils ausente para CCD cache", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "dependencia rdkit ausente para CCD cache", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "dependencia numpy ausente para CCD cache", impact="1", examples=[f"{type(exc).__name__}:{exc}"])

    want = [str(c).strip().upper() for c in codes if str(c).strip()]
    doc = gemmi.cif.read(str(components_cif))
    mols: dict[str, object] = {}
    for code in want:
        try:
            block = doc[code]
        except Exception as exc:  # noqa: BLE001
            raise_error(stage, location, "bloco CCD ausente no components.cif", impact="1", examples=[code, f"{type(exc).__name__}:{exc}"])
        try:
            parsed = ccd_reader._parse_pdb_mmcif(block, sanitize=True)  # noqa: SLF001
            mol = parsed.component.mol
        except Exception as exc:  # noqa: BLE001
            raise_error(stage, location, "falha ao converter bloco CCD para rdkit mol", impact="1", examples=[code, f"{type(exc).__name__}:{exc}"])

        # Atom name mapping.
        try:
            mol.atom_map = {atom.GetProp("name"): atom.GetIdx() for atom in mol.GetAtoms()}  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            raise_error(stage, location, "rdkit mol sem propriedades de atom name", impact="1", examples=[code, f"{type(exc).__name__}:{exc}"])

        mol.name = code  # type: ignore[attr-defined]
        mol.sanitized = bool(getattr(parsed, "sanitized", True))  # type: ignore[attr-defined]
        mol.ref_conf_id = 0  # type: ignore[attr-defined]
        mol.ref_conf_type = "idea"  # type: ignore[attr-defined]

        # Build ref_mask from ideal coords availability (match RNAPro preprocess script behavior).
        num_atom = int(mol.GetNumAtoms())
        ref_mask = np.zeros(num_atom, dtype=bool)
        try:
            atoms = block.find("_chem_comp_atom.", ["atom_id", "model_Cartn_x", "pdbx_model_Cartn_x_ideal"])
        except Exception as exc:  # noqa: BLE001
            raise_error(stage, location, "bloco CCD sem tabela _chem_comp_atom", impact="1", examples=[code, f"{type(exc).__name__}:{exc}"])
        if len(atoms) != num_atom:
            raise_error(stage, location, "mismatch n_atoms CCD vs rdkit mol", impact="1", examples=[code, f"ccd={len(atoms)}", f"rdkit={num_atom}"])
        for row in atoms:
            atom_id = gemmi.cif.as_string(row["_chem_comp_atom.atom_id"])
            atom_idx = mol.atom_map.get(atom_id)  # type: ignore[attr-defined]
            if atom_idx is None:
                raise_error(stage, location, "atom_id ausente no atom_map", impact="1", examples=[code, atom_id])
            x_ideal = row["_chem_comp_atom.pdbx_model_Cartn_x_ideal"]
            ref_mask[int(atom_idx)] = str(x_ideal) != "?"
        mol.ref_mask = ref_mask  # type: ignore[attr-defined]

        # Best-effort embed if sanitize succeeded (tiny set of residues; helps ensure coords exist).
        if bool(getattr(mol, "sanitized", True)) and num_atom > 0:
            try:
                options = AllChem.ETKDGv3()
                options.clearConfs = False
                conf_id = AllChem.EmbedMolecule(mol, options)
                mol.ref_conf_id = int(conf_id)  # type: ignore[attr-defined]
                mol.ref_conf_type = "rdkit"  # type: ignore[attr-defined]
                mol.ref_mask[:] = True  # type: ignore[attr-defined]
            except Exception:
                # Keep ideal coords; do not fail if embedding fails.
                pass

        mols[code] = mol

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump(mols, f)
    return {"n_mols": int(len(mols))}


def prepare_rnapro_support_files(
    *,
    repo_root: Path,
    model_dir: Path,
    codes: list[str] | None = None,
    components_cif_gz_url: str = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz",
    timeout_seconds: int = 10 * 60,
    overwrite: bool = False,
) -> RnaProSupportFilesResult:
    stage = "RNAPRO_SUPPORT"
    location = "src/rna3d_local/rnapro_support.py:prepare_rnapro_support_files"
    model_dir = model_dir.resolve()
    if not model_dir.exists():
        raise_error(stage, location, "model_dir ausente", impact="1", examples=[str(model_dir)])
    codes = ["A", "C", "G", "U"] if codes is None else [str(c) for c in codes]

    # 1) Empty templates placeholder (required because RNAPro InferenceDataset torch.load()s template_data unconditionally).
    template_pt = model_dir / "test_templates.pt"
    if template_pt.exists() and not overwrite:
        template_meta = {"status": "exists", "path": str(template_pt)}
    else:
        try:
            import torch  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise_error(stage, location, "torch ausente para criar test_templates.pt", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
        template_pt.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save({}, template_pt)
        except Exception as exc:  # noqa: BLE001
            raise_error(stage, location, "falha ao salvar test_templates.pt", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
        template_meta = {"status": "written", "path": str(template_pt)}

    # 2) Minimal CCD cache (components.cif + rdkit mol pkl).
    ccd_cache = model_dir / "ccd_cache"
    components_cif = ccd_cache / "components.cif"
    rdkit_pkl = ccd_cache / "components.cif.rdkit_mol.pkl"
    clusters = ccd_cache / "clusters-by-entity-40.txt"

    if components_cif.exists() and rdkit_pkl.exists() and clusters.exists() and not overwrite:
        ccd_meta = {"status": "exists", "ccd_cache_dir": str(ccd_cache)}
    else:
        ccd_cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="rnapro_ccd_") as tmpdir:
            tmp = Path(tmpdir)
            gz_path = tmp / "components.cif.gz"
            _download_to_file(
                components_cif_gz_url,
                out_path=gz_path,
                stage=stage,
                location=location,
                timeout_seconds=int(timeout_seconds),
            )
            stats_extract = _extract_components_blocks(
                components_cif_gz=gz_path,
                out_components_cif=components_cif,
                codes=codes,
                stage=stage,
                location=location,
            )
            stats_pkl = _build_minimal_rdkit_mol_pkl(
                components_cif=components_cif,
                out_pkl=rdkit_pkl,
                codes=codes,
                stage=stage,
                location=location,
            )
        clusters.parent.mkdir(parents=True, exist_ok=True)
        clusters.write_text("", encoding="utf-8")
        ccd_meta = {"status": "written", "ccd_cache_dir": str(ccd_cache), **stats_extract, **stats_pkl}

    manifest_path = model_dir / "rnapro_support_manifest.json"
    payload = {
        "created_utc": utc_now_iso(),
        "model_dir": rel_or_abs(model_dir, repo_root),
        "components_cif_gz_url": str(components_cif_gz_url),
        "codes": [str(c).strip().upper() for c in codes],
        "overwrite": bool(overwrite),
        "template": template_meta,
        "ccd_cache": ccd_meta,
        "sha256": {
            "test_templates.pt": sha256_file(template_pt),
            "ccd_cache/components.cif": sha256_file(components_cif),
            "ccd_cache/components.cif.rdkit_mol.pkl": sha256_file(rdkit_pkl),
            "ccd_cache/clusters-by-entity-40.txt": sha256_file(clusters),
        },
        "size_bytes": {
            "test_templates.pt": int(template_pt.stat().st_size),
            "ccd_cache/components.cif": int(components_cif.stat().st_size),
            "ccd_cache/components.cif.rdkit_mol.pkl": int(rdkit_pkl.stat().st_size),
            "ccd_cache/clusters-by-entity-40.txt": int(clusters.stat().st_size),
        },
    }
    write_json(manifest_path, payload)
    return RnaProSupportFilesResult(model_dir=model_dir, manifest_path=manifest_path)
