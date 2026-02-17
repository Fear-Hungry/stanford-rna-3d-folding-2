from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import polars as pl

from ..errors import raise_error
from ..io_tables import read_table, write_table


def _normalize_rnapro_ranking_score(raw: float, *, stage: str, location: str, target_id: str) -> float:
    if not float(raw) == float(raw):  # NaN
        raise_error(stage, location, "ranking_score do rnapro e NaN", impact="1", examples=[target_id])
    if float(raw) < 0.0:
        raise_error(stage, location, "ranking_score do rnapro negativo", impact="1", examples=[target_id, str(raw)])
    score = float(raw)
    if score > 1.0:
        if score <= 100.0:
            score = score / 100.0
        else:
            raise_error(stage, location, "ranking_score do rnapro fora do intervalo esperado", impact="1", examples=[target_id, str(raw)])
    if score < 0.0 or score > 1.0:
        raise_error(stage, location, "ranking_score do rnapro apos normalizacao fora de [0,1]", impact="1", examples=[target_id, str(score)])
    return float(score)


def _normalize_seq_parts(
    *,
    seq: str,
    chain_separator: str,
    stage: str,
    location: str,
    target_id: str,
) -> list[str]:
    sep = str(chain_separator)
    if len(sep) != 1:
        raise_error(stage, location, "chain_separator invalido", impact="1", examples=[repr(sep)])
    raw = str(seq or "").strip().upper().replace("T", "U")
    cleaned = "".join(ch for ch in raw if ch not in {" ", "\t", "\n", "\r"})
    if not cleaned:
        raise_error(stage, location, "sequencia vazia", impact="1", examples=[target_id])
    bad = sorted({ch for ch in cleaned if ch not in {"A", "C", "G", "U", sep}})
    if bad:
        raise_error(stage, location, "sequencia contem simbolos invalidos", impact=str(len(bad)), examples=[f"{target_id}:{''.join(bad[:8])}"])
    parts = [p for p in cleaned.split(sep)]
    if any(p == "" for p in parts):
        raise_error(stage, location, "sequencia multicadeia invalida (cadeia vazia)", impact="1", examples=[target_id])
    return parts


def _base_from_resname(resname: str) -> str:
    name = str(resname or "").strip().upper()
    mapping = {"A": "A", "C": "C", "G": "G", "U": "U", "DA": "A", "DC": "C", "DG": "G", "DT": "U", "DU": "U"}
    return mapping.get(name, name[:1] if name else "")


def _write_input_json(*, path: Path, target_id: str, seq_parts: list[str]) -> None:
    payload = [
        {
            "name": str(target_id),
            "sequences": [{"rnaSequence": {"sequence": str(part), "count": 1}} for part in seq_parts],
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extract_c1_from_cif(
    *,
    cif_path: Path,
    expected_seq_parts: list[str],
    stage: str,
    location: str,
    target_id: str,
) -> list[tuple[str, float, float, float]]:
    try:
        import gemmi  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "dependencia gemmi ausente para ler mmcif do rnapro", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    try:
        st = gemmi.read_structure(str(cif_path))
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "falha ao ler mmcif do rnapro", impact="1", examples=[str(cif_path), f"{type(exc).__name__}:{exc}"])
    if len(st) == 0:
        raise_error(stage, location, "mmcif sem models", impact="1", examples=[str(cif_path)])
    model = st[0]

    expected_chain_ids = [chr(ord("A") + idx) for idx in range(len(expected_seq_parts))]
    chain_map = {str(ch.name): ch for ch in model}
    missing = [c for c in expected_chain_ids if c not in chain_map]
    if missing:
        raise_error(stage, location, "cadeias esperadas ausentes no mmcif do rnapro", impact=str(len(missing)), examples=[target_id, *missing[:6]])

    out: list[tuple[str, float, float, float]] = []
    for chain_id, expected_seq in zip(expected_chain_ids, expected_seq_parts, strict=True):
        chain = chain_map[chain_id]
        residues = [res for res in chain if res is not None]
        if len(residues) != len(expected_seq):
            raise_error(
                stage,
                location,
                "numero de residuos do rnapro diverge do esperado para a cadeia",
                impact="1",
                examples=[f"{target_id}:{chain_id}", f"expected={len(expected_seq)}", f"actual={len(residues)}", str(cif_path)],
            )
        for idx, (res, expected_base) in enumerate(zip(residues, expected_seq, strict=True), start=1):
            base = _base_from_resname(res.name)
            if base != expected_base:
                raise_error(
                    stage,
                    location,
                    "resname do rnapro nao bate com sequencia esperada",
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
                raise_error(stage, location, "atomo C1' ausente no rnapro", impact="1", examples=[f"{target_id}:{chain_id}:{idx}", res.name])
            pos = atom.pos
            out.append((base, float(pos.x), float(pos.y), float(pos.z)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rnapro_runner")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-models", required=True, type=int)
    ap.add_argument("--chain-separator", default="|")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--diffusion-steps", type=int, default=200)
    ap.add_argument("--dtype", choices=["bf16", "fp32", "fp16"], default="bf16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    stage = "RNAPRO_RUNNER"
    location = "src/rna3d_local/runners/rnapro.py:main"

    model_dir = Path(args.model_dir).resolve()
    targets_path = Path(args.targets).resolve()
    out_path = Path(args.out).resolve()
    n_models = int(args.n_models)
    if n_models <= 0:
        raise_error(stage, location, "n_models invalido", impact="1", examples=[str(args.n_models)])
    if int(args.diffusion_steps) <= 0:
        raise_error(stage, location, "diffusion_steps invalido", impact="1", examples=[str(args.diffusion_steps)])

    ckpt = model_dir / "rnapro-public-best-500m.ckpt"
    ccd_cache_dir = model_dir / "ccd_cache"
    ribo_dir = model_dir / "ribonanzanet2_checkpoint"
    template_pt = model_dir / "test_templates.pt"
    missing = [str(p) for p in [ckpt, ccd_cache_dir / "components.cif", ccd_cache_dir / "components.cif.rdkit_mol.pkl", ribo_dir / "pairwise.yaml", ribo_dir / "pytorch_model_fsdp.bin", template_pt] if not p.exists()]
    if missing:
        raise_error(
            stage,
            location,
            "artefatos obrigatorios do RNAPro ausentes (use prepare-rnapro-support-files + fetch-pretrained-assets)",
            impact=str(len(missing)),
            examples=missing[:8],
        )

    # Ensure RNAPro uses our local CCD cache (required by rnapro.data.ccd).
    os.environ["RNAPRO_DATA_ROOT_DIR"] = str(ccd_cache_dir)

    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "torch indisponivel para runner rnapro", impact="1", examples=[f"{type(exc).__name__}:{exc}"])

    try:
        from configs.configs_base import configs as configs_base  # type: ignore
        from configs.configs_data import data_configs  # type: ignore
        from configs.configs_inference import inference_configs  # type: ignore
        from runner.inference import InferenceRunner  # type: ignore
        from rnapro.config import parse_configs  # type: ignore
        from rnapro.data.infer_data_pipeline import get_inference_dataloader  # type: ignore
        from rnapro.utils.inference import update_inference_configs  # type: ignore
        from rnapro.utils.seed import seed_everything  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(
            stage,
            location,
            "dependencias rnapro ausentes/invalidas (instale wheel rnapro + deps no wheelhouse)",
            impact="1",
            examples=[f"{type(exc).__name__}:{exc}"],
        )

    # Build configs (strict runner; no silent fallback).
    base = dict(configs_base)
    base["use_deepspeed_evo_attention"] = False
    configs = {**base, **{"data": data_configs}, **dict(inference_configs)}
    dump_root = Path(tempfile.mkdtemp(prefix="rnapro_dump_")).resolve()
    msa_dir = dump_root / "rna_msa"
    msa_dir.mkdir(parents=True, exist_ok=True)

    arg_items = [
        "--project",
        "rnapro_offline",
        "--run_name",
        "inference",
        "--base_dir",
        ".",
        "--eval_interval",
        "0",
        "--log_interval",
        "0",
        "--max_steps",
        "0",
        "--use_wandb",
        "false",
        "--logger",
        "print",
        "--dtype",
        str(args.dtype),
        "--dump_dir",
        str(dump_root),
        "--load_checkpoint_path",
        str(ckpt),
        "--seeds",
        str(int(args.seed)),
        "--num_workers",
        "0",
        "--use_msa",
        "false",
        "--rna_msa_dir",
        str(msa_dir),
        "--use_template",
        "None",
        "--template_data",
        str(template_pt),
        "--template_idx",
        "0",
        "--triangle_attention",
        "torch",
        "--triangle_multiplicative",
        "torch",
        "--sample_diffusion.N_sample",
        str(int(n_models)),
        "--sample_diffusion.N_step",
        str(int(args.diffusion_steps)),
        "--model.ribonanza_net_path",
        str(ribo_dir),
    ]
    cfg = parse_configs(configs=configs, arg_str=" ".join(arg_items), fill_required_with_null=True)

    # Build model once and reuse across targets.
    try:
        runner = InferenceRunner(cfg)
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "falha ao construir/carregar RNAPro", impact="1", examples=[f"{type(exc).__name__}:{exc}"])

    targets = read_table(targets_path, stage=stage, location=location)
    if "target_id" not in targets.columns or "sequence" not in targets.columns:
        raise_error(stage, location, "targets schema invalido (faltam colunas)", impact="1", examples=["target_id", "sequence"])

    rows: list[dict[str, object]] = []
    try:
        for target_id, sequence in targets.select("target_id", "sequence").iter_rows():
            tid = str(target_id)
            seq_parts = _normalize_seq_parts(seq=str(sequence), chain_separator=str(args.chain_separator), stage=stage, location=location, target_id=tid)
            concat_len = int(sum(len(p) for p in seq_parts))
            if concat_len <= 0:
                raise_error(stage, location, "sequencia com len=0 apos normalizacao", impact="1", examples=[tid])

            # Prepare input JSON in dump_root to keep all runtime I/O in a writable directory.
            target_dir = dump_root / tid
            if target_dir.exists():
                shutil.rmtree(target_dir)
            input_json = dump_root / "inputs" / f"{tid}_input.json"
            _write_input_json(path=input_json, target_id=tid, seq_parts=seq_parts)
            cfg.input_json_path = str(input_json)

            # Run inference for one batch.
            dataloader = get_inference_dataloader(configs=cfg)
            for seed in cfg.seeds:
                seed_everything(seed=seed, deterministic=cfg.deterministic)
                for batch in dataloader:
                    data, atom_array, data_error_message = batch[0]
                    sample_name = str(data.get("sample_name") or tid)
                    if str(sample_name) != tid:
                        raise_error(stage, location, "sample_name divergente do target_id", impact="1", examples=[tid, sample_name])
                    if str(data_error_message or "").strip():
                        raise_error(stage, location, "erro ao featurizar entrada do rnapro", impact="1", examples=[tid, str(data_error_message)[:200]])
                    new_cfg = update_inference_configs(cfg, int(data["N_token"].item()))
                    runner.update_model_configs(new_cfg)
                    prediction = runner.predict(data)
                    runner.dumper.dump(
                        dataset_name="",
                        pdb_id=sample_name,
                        seed=int(seed),
                        pred_dict=prediction,
                        atom_array=atom_array,
                        entity_poly_type=data["entity_poly_type"],
                    )
                    if hasattr(torch.cuda, "empty_cache") and bool(torch.cuda.is_available()):
                        torch.cuda.empty_cache()

            pred_dir = dump_root / tid / f"seed_{int(cfg.seeds[0])}" / "predictions"
            if not pred_dir.exists():
                raise_error(stage, location, "rnapro nao gerou diretorio de predicoes", impact="1", examples=[tid, str(pred_dir)])

            for model_id in range(1, int(n_models) + 1):
                rank = int(model_id - 1)
                cif_path = pred_dir / f"{tid}_sample_{rank}.cif"
                conf_path = pred_dir / f"{tid}_summary_confidence_sample_{rank}.json"
                if not cif_path.exists() or not conf_path.exists():
                    raise_error(
                        stage,
                        location,
                        "rnapro nao gerou mmcif/score esperados",
                        impact="1",
                        examples=[tid, f"model_id={model_id}", str(cif_path), str(conf_path)],
                    )
                payload = json.loads(conf_path.read_text(encoding="utf-8"))
                if "ranking_score" not in payload:
                    raise_error(stage, location, "summary_confidence sem ranking_score", impact="1", examples=[tid, str(conf_path)])
                confidence = _normalize_rnapro_ranking_score(float(payload["ranking_score"]), stage=stage, location=location, target_id=tid)
                coords = _extract_c1_from_cif(cif_path=cif_path, expected_seq_parts=seq_parts, stage=stage, location=location, target_id=tid)
                if len(coords) != concat_len:
                    raise_error(stage, location, "comprimento de coordenadas divergente", impact="1", examples=[tid, f"expected={concat_len}", f"actual={len(coords)}"])
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
                            "source": "rnapro",
                            "confidence": float(confidence),
                        }
                    )
    finally:
        # Best-effort cleanup.
        try:
            shutil.rmtree(dump_root)
        except Exception:
            pass

    df = pl.DataFrame(rows)
    write_table(df, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
