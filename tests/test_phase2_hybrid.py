from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.boltz1_offline import predict_boltz1_offline
from rna3d_local.chai1_offline import predict_chai1_offline
from rna3d_local.errors import PipelineError
from rna3d_local.hybrid_router import build_hybrid_candidates
from rna3d_local.hybrid_select import select_top5_hybrid
from rna3d_local.rnapro_offline import predict_rnapro_offline
from rna3d_local.submission import export_submission
from rna3d_local.submit_readiness import evaluate_submit_readiness


def _write_csv(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_csv(path)


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_stub_runner(model_dir: Path, *, source: str) -> None:
    script = (
        "from __future__ import annotations\n"
        "\n"
        "import argparse\n"
        "from pathlib import Path\n"
        "\n"
        "import polars as pl\n"
        "\n"
        "\n"
        "def main() -> int:\n"
        "    ap = argparse.ArgumentParser()\n"
        "    ap.add_argument('--targets', required=True)\n"
        "    ap.add_argument('--out', required=True)\n"
        "    ap.add_argument('--n-models', type=int, required=True)\n"
        "    ap.add_argument('--source', required=True)\n"
        "    args = ap.parse_args()\n"
        "    targets = pl.read_csv(args.targets)\n"
        "    if 'target_id' not in targets.columns or 'sequence' not in targets.columns:\n"
        "        raise SystemExit('targets schema invalido')\n"
        "    rows = []\n"
        "    for tid, seq in targets.select('target_id','sequence').iter_rows():\n"
        "        tid = str(tid)\n"
        "        seq = str(seq).replace('|','').upper().replace('T','U')\n"
        "        for model_id in range(1, int(args.n_models) + 1):\n"
        "            for resid, base in enumerate(seq, start=1):\n"
        "                rows.append({\n"
        "                    'target_id': tid,\n"
        "                    'model_id': int(model_id),\n"
        "                    'resid': int(resid),\n"
        "                    'resname': str(base),\n"
        "                    'x': float(model_id + resid),\n"
        "                    'y': float(model_id + resid + 1),\n"
        "                    'z': float(model_id + resid + 2),\n"
        "                    'source': str(args.source),\n"
        "                    'confidence': 0.75,\n"
        "                })\n"
        "    out = Path(args.out)\n"
        "    out.parent.mkdir(parents=True, exist_ok=True)\n"
        "    pl.DataFrame(rows).write_parquet(out)\n"
        "    return 0\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n"
    )
    _touch(model_dir / "run_infer.py", script)
    entry = [
        "python",
        "run_infer.py",
        "--targets",
        "{targets}",
        "--out",
        "{out}",
        "--n-models",
        "{n_models}",
        "--source",
        source,
    ]
    _touch(model_dir / "config.json", json.dumps({"entrypoint": entry}))


def _make_model_dirs(base: Path) -> tuple[Path, Path, Path]:
    rna = base / "rnapro"
    _touch(rna / "rnapro-public-best-500m.ckpt")
    _touch(rna / "test_templates.pt")
    _touch(rna / "ccd_cache" / "components.cif")
    _touch(rna / "ccd_cache" / "components.cif.rdkit_mol.pkl")
    _touch(rna / "ccd_cache" / "clusters-by-entity-40.txt")
    _touch(rna / "ribonanzanet2_checkpoint" / "pairwise.yaml")
    _touch(rna / "ribonanzanet2_checkpoint" / "pytorch_model_fsdp.bin")
    _write_stub_runner(rna, source="rnapro")
    chai = base / "chai1"
    _touch(chai / "conformers_v1.apkl")
    _touch(chai / "esm" / "traced_sdpa_esm2_t36_3B_UR50D_fp16.pt")
    _touch(chai / "models_v2" / "feature_embedding.pt")
    _touch(chai / "models_v2" / "bond_loss_input_proj.pt")
    _touch(chai / "models_v2" / "token_embedder.pt")
    _touch(chai / "models_v2" / "trunk.pt")
    _touch(chai / "models_v2" / "diffusion_module.pt")
    _touch(chai / "models_v2" / "confidence_head.pt")
    _write_stub_runner(chai, source="chai1")
    boltz = base / "boltz1"
    _touch(boltz / "boltz1_conf.ckpt")
    _touch(boltz / "ccd.pkl")
    _write_stub_runner(boltz, source="boltz1")
    return rna, chai, boltz


def _make_tbm(path: Path) -> None:
    rows = []
    for target_id, seq in [("T1", "AC"), ("T2", "GU"), ("T3", "AA")]:
        for model_id in range(1, 6):
            for resid, base in enumerate(seq, start=1):
                rows.append(
                    {
                        "target_id": target_id,
                        "model_id": model_id,
                        "resid": resid,
                        "resname": base,
                        "x": float(model_id + resid),
                        "y": float(model_id + resid + 1),
                        "z": float(model_id + resid + 2),
                        "source": "tbm",
                        "confidence": 0.9,
                    }
                )
    pl.DataFrame(rows).write_parquet(path)


def _make_se3(path: Path) -> None:
    rows = []
    for target_id, seq in [("T1", "AC"), ("T2", "GU"), ("T3", "AA")]:
        for model_id in range(1, 6):
            for resid, base in enumerate(seq, start=1):
                rows.append(
                    {
                        "target_id": target_id,
                        "model_id": model_id,
                        "resid": resid,
                        "resname": base,
                        "x": float(model_id + resid + 10),
                        "y": float(model_id + resid + 11),
                        "z": float(model_id + resid + 12),
                        "source": "generative_se3",
                        "confidence": 0.83,
                    }
                )
    pl.DataFrame(rows).write_parquet(path)


def _make_sample(path: Path) -> None:
    rows = []
    for target_id, seq in [("T1", "AC"), ("T2", "GU"), ("T3", "AA")]:
        for resid, base in enumerate(seq, start=1):
            row = {"ID": f"{target_id}_{resid}", "resname": base, "resid": resid}
            for model_id in range(1, 6):
                row[f"x_{model_id}"] = 0.0
                row[f"y_{model_id}"] = 0.0
                row[f"z_{model_id}"] = 0.0
            rows.append(row)
    pl.DataFrame(rows).write_csv(path)


def test_phase2_router_and_top5_selection(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    _write_csv(
        targets,
        [
            {"target_id": "T1", "sequence": "AC", "ligand_SMILES": ""},
            {"target_id": "T2", "sequence": "GU", "ligand_SMILES": ""},
            {"target_id": "T3", "sequence": "AA", "ligand_SMILES": "C1=CC=CC=C1"},
        ],
    )
    retrieval = tmp_path / "retrieval.parquet"
    pl.DataFrame(
        [
            {"target_id": "T1", "final_score": 0.95},
            {"target_id": "T2", "final_score": 0.10},
            {"target_id": "T3", "final_score": 0.10},
        ]
    ).write_parquet(retrieval)
    tbm = tmp_path / "tbm.parquet"
    _make_tbm(tbm)

    rna_dir, chai_dir, boltz_dir = _make_model_dirs(tmp_path / "models")
    rna = predict_rnapro_offline(repo_root=tmp_path, model_dir=rna_dir, targets_path=targets, out_path=tmp_path / "rnapro.parquet", n_models=5)
    chai = predict_chai1_offline(repo_root=tmp_path, model_dir=chai_dir, targets_path=targets, out_path=tmp_path / "chai.parquet", n_models=5)
    boltz = predict_boltz1_offline(repo_root=tmp_path, model_dir=boltz_dir, targets_path=targets, out_path=tmp_path / "boltz.parquet", n_models=5)

    routed = build_hybrid_candidates(
        repo_root=tmp_path,
        targets_path=targets,
        retrieval_path=retrieval,
        tbm_path=tbm,
        out_path=tmp_path / "hybrid_candidates.parquet",
        routing_path=tmp_path / "routing.parquet",
        template_score_threshold=0.65,
        rnapro_path=rna.predictions_path,
        chai1_path=chai.predictions_path,
        boltz1_path=boltz.predictions_path,
    )
    routing = pl.read_parquet(routed.routing_path).sort("target_id")
    rules = {row["target_id"]: row["route_rule"] for row in routing.iter_rows(named=True)}
    assert rules["T1"] == "len<=350->foundation_trio"
    assert rules["T2"] == "len<=350->foundation_trio"
    assert rules["T3"] == "len<=350->foundation_trio"

    candidates = pl.read_parquet(routed.candidates_path)
    for tid in ("T1", "T2", "T3"):
        tid_sources = set(candidates.filter(pl.col("target_id") == tid).select("source").unique().get_column("source").to_list())
        assert "chai1" in tid_sources
        assert "boltz1" in tid_sources
        assert "rnapro" in tid_sources

    top5 = select_top5_hybrid(
        repo_root=tmp_path,
        candidates_path=routed.candidates_path,
        out_path=tmp_path / "hybrid_top5.parquet",
        n_models=5,
    )
    out = pl.read_parquet(top5.predictions_path)
    assert out.get_column("target_id").n_unique() == 3
    per_target = out.group_by("target_id").agg(pl.col("model_id").n_unique().alias("n_models"))
    assert per_target.filter(pl.col("n_models") != 5).height == 0


def test_hybrid_router_injects_tbm_confidence_from_template_score(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    _write_csv(
        targets,
        [
            {"target_id": "TLONG", "sequence": ("A" * 601), "ligand_SMILES": ""},
        ],
    )
    retrieval = tmp_path / "retrieval.parquet"
    pl.DataFrame([{"target_id": "TLONG", "final_score": 0.88}]).write_parquet(retrieval)

    tbm = tmp_path / "tbm.parquet"
    pl.DataFrame(
        [
            {"target_id": "TLONG", "model_id": 1, "resid": 1, "resname": "A", "x": 0.0, "y": 0.0, "z": 0.0},
            {"target_id": "TLONG", "model_id": 1, "resid": 2, "resname": "A", "x": 1.0, "y": 0.0, "z": 0.0},
        ]
    ).write_parquet(tbm)

    out = build_hybrid_candidates(
        repo_root=tmp_path,
        targets_path=targets,
        retrieval_path=retrieval,
        tbm_path=tbm,
        out_path=tmp_path / "hybrid_candidates.parquet",
        routing_path=tmp_path / "routing.parquet",
        template_score_threshold=0.65,
        rnapro_path=None,
        chai1_path=None,
        boltz1_path=None,
        se3_path=None,
    )
    routing = pl.read_parquet(out.routing_path)
    assert routing.item(0, "route_rule") == "len>600->tbm+se3_mamba_fallback->tbm"
    assert routing.item(0, "primary_source") == "tbm"
    assert routing.item(0, "fallback_used") is True
    assert routing.item(0, "fallback_source") == "tbm"

    candidates = pl.read_parquet(out.candidates_path)
    tbm_rows = candidates.filter(pl.col("source") == "tbm")
    assert tbm_rows.height == 2
    conf = tbm_rows.select(pl.col("confidence").drop_nulls()).get_column("confidence").unique().to_list()
    assert conf == [pytest.approx(0.88)]


def test_phase2_router_falls_back_when_template_strong_without_tbm_coverage(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    _write_csv(
        targets,
        [
            {"target_id": "TMED", "sequence": ("A" * 400), "ligand_SMILES": ""},
        ],
    )
    retrieval = tmp_path / "retrieval.parquet"
    pl.DataFrame([{"target_id": "TMED", "final_score": 0.95}]).write_parquet(retrieval)
    tbm = tmp_path / "tbm.parquet"
    se3_flash = tmp_path / "se3_flash.parquet"
    tbm_rows: list[dict[str, object]] = []
    se3_rows: list[dict[str, object]] = []
    for model_id in range(1, 6):
        for resid in range(1, 401):
            tbm_rows.append(
                {
                    "target_id": "TMED",
                    "model_id": model_id,
                    "resid": resid,
                    "resname": "A",
                    "x": float(model_id + resid),
                    "y": float(model_id + resid + 1),
                    "z": float(model_id + resid + 2),
                    "source": "tbm",
                    "confidence": 0.9,
                }
            )
            se3_rows.append(
                {
                    "target_id": "TMED",
                    "model_id": model_id,
                    "resid": resid,
                    "resname": "A",
                    "x": float(model_id + resid + 10),
                    "y": float(model_id + resid + 11),
                    "z": float(model_id + resid + 12),
                    "source": "generative_se3",
                    "confidence": 0.82,
                }
            )
    pl.DataFrame(tbm_rows).write_parquet(tbm)
    pl.DataFrame(se3_rows).write_parquet(se3_flash)

    routed = build_hybrid_candidates(
        repo_root=tmp_path,
        targets_path=targets,
        retrieval_path=retrieval,
        tbm_path=tbm,
        out_path=tmp_path / "hybrid_candidates.parquet",
        routing_path=tmp_path / "routing.parquet",
        template_score_threshold=0.65,
        rnapro_path=None,
        chai1_path=None,
        boltz1_path=None,
        se3_flash_path=se3_flash,
    )
    routing = pl.read_parquet(routed.routing_path).sort("target_id")
    rules = {row["target_id"]: row["route_rule"] for row in routing.iter_rows(named=True)}
    assert rules["TMED"] == "350<len<=600->se3_flash"
    assert routing.item(0, "primary_source") == "se3_flash"


def test_phase2_router_prefers_foundation_over_se3_when_available(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    _write_csv(
        targets,
        [
            {"target_id": "T1", "sequence": "AC", "ligand_SMILES": ""},
            {"target_id": "T2", "sequence": "GU", "ligand_SMILES": ""},
        ],
    )
    retrieval = tmp_path / "retrieval.parquet"
    pl.DataFrame(
        [
            {"target_id": "T1", "final_score": 0.80},
            {"target_id": "T2", "final_score": 0.10},
        ]
    ).write_parquet(retrieval)
    tbm = tmp_path / "tbm.parquet"
    _make_tbm(tbm)
    tbm_df = pl.read_parquet(tbm).filter(pl.col("target_id") != "T2")
    tbm_df.write_parquet(tbm)
    se3 = tmp_path / "se3.parquet"
    _make_se3(se3)

    rna_dir, chai_dir, boltz_dir = _make_model_dirs(tmp_path / "models")
    rna = predict_rnapro_offline(repo_root=tmp_path, model_dir=rna_dir, targets_path=targets, out_path=tmp_path / "rnapro.parquet", n_models=5)
    chai = predict_chai1_offline(repo_root=tmp_path, model_dir=chai_dir, targets_path=targets, out_path=tmp_path / "chai.parquet", n_models=5)
    boltz = predict_boltz1_offline(repo_root=tmp_path, model_dir=boltz_dir, targets_path=targets, out_path=tmp_path / "boltz.parquet", n_models=5)

    routed = build_hybrid_candidates(
        repo_root=tmp_path,
        targets_path=targets,
        retrieval_path=retrieval,
        tbm_path=tbm,
        out_path=tmp_path / "hybrid_candidates.parquet",
        routing_path=tmp_path / "routing.parquet",
        template_score_threshold=0.65,
        rnapro_path=rna.predictions_path,
        chai1_path=chai.predictions_path,
        boltz1_path=boltz.predictions_path,
        se3_path=se3,
    )
    routing = pl.read_parquet(routed.routing_path).sort("target_id")
    rules = {row["target_id"]: row["route_rule"] for row in routing.iter_rows(named=True)}
    assert rules["T1"] == "len<=350->foundation_trio"
    assert rules["T2"] == "len<=350->foundation_trio"
    assert routing.filter(pl.col("primary_source") != "foundation_trio").height == 0


def test_phase2_router_falls_back_to_se3_when_chai_boltz_missing_for_target(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    _write_csv(
        targets,
        [
            {"target_id": "T1", "sequence": "AC", "ligand_SMILES": ""},
            {"target_id": "T2", "sequence": "GU", "ligand_SMILES": ""},
        ],
    )
    retrieval = tmp_path / "retrieval.parquet"
    pl.DataFrame(
        [
            {"target_id": "T1", "final_score": 0.10},
            {"target_id": "T2", "final_score": 0.10},
        ]
    ).write_parquet(retrieval)
    tbm = tmp_path / "tbm.parquet"
    _make_tbm(tbm)

    # Chai/Boltz only run for a subset (T1), so short bucket must fail-fast (sem fallback silencioso).
    foundation_targets = tmp_path / "foundation_targets.csv"
    _write_csv(
        foundation_targets,
        [
            {"target_id": "T1", "sequence": "AC", "ligand_SMILES": ""},
        ],
    )

    _rna_dir, chai_dir, boltz_dir = _make_model_dirs(tmp_path / "models")
    chai = predict_chai1_offline(repo_root=tmp_path, model_dir=chai_dir, targets_path=foundation_targets, out_path=tmp_path / "chai.parquet", n_models=5)
    boltz = predict_boltz1_offline(repo_root=tmp_path, model_dir=boltz_dir, targets_path=foundation_targets, out_path=tmp_path / "boltz.parquet", n_models=5)

    se3 = tmp_path / "se3.parquet"
    rows = []
    for target_id, seq in [("T1", "AC"), ("T2", "GU")]:
        for model_id in range(1, 6):
            for resid, base in enumerate(seq, start=1):
                rows.append(
                    {
                        "target_id": target_id,
                        "model_id": model_id,
                        "resid": resid,
                        "resname": base,
                        "x": float(model_id + resid + 10),
                        "y": float(model_id + resid + 11),
                        "z": float(model_id + resid + 12),
                        "source": "generative_se3",
                        "confidence": 0.83,
                    }
                )
    pl.DataFrame(rows).write_parquet(se3)

    with pytest.raises(PipelineError, match="bucket short exige foundation trio completo"):
        build_hybrid_candidates(
            repo_root=tmp_path,
            targets_path=targets,
            retrieval_path=retrieval,
            tbm_path=tbm,
            out_path=tmp_path / "hybrid_candidates.parquet",
            routing_path=tmp_path / "routing.parquet",
            template_score_threshold=0.65,
            rnapro_path=None,
            chai1_path=chai.predictions_path,
            boltz1_path=boltz.predictions_path,
            se3_path=se3,
        )


def test_readiness_gate_blocks_non_improvement(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    _make_sample(sample)
    long = []
    for target_id, seq in [("T1", "AC"), ("T2", "GU"), ("T3", "AA")]:
        for model_id in range(1, 6):
            for resid, base in enumerate(seq, start=1):
                long.append(
                    {
                        "target_id": target_id,
                        "model_id": model_id,
                        "resid": resid,
                        "resname": base,
                        "x": 1.0,
                        "y": 2.0,
                        "z": 3.0,
                    }
                )
    long_path = tmp_path / "pred.parquet"
    pl.DataFrame(long).write_parquet(long_path)
    submission = tmp_path / "submission.csv"
    export_submission(sample_path=sample, predictions_long_path=long_path, out_path=submission)
    score_json = tmp_path / "score.json"
    score_json.write_text(json.dumps({"score": 0.10}), encoding="utf-8")
    with pytest.raises(PipelineError, match="gate de melhoria estrita"):
        evaluate_submit_readiness(
            repo_root=tmp_path,
            sample_path=sample,
            submission_path=submission,
            score_json_path=score_json,
            baseline_score=0.20,
            report_path=tmp_path / "readiness.json",
            fail_on_disallow=True,
        )


def test_phase2_router_ultralong_forces_se3_fallback(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    _write_csv(
        targets,
        [
            {"target_id": "TLONG", "sequence": ("A" * 1601), "ligand_SMILES": ""},
        ],
    )
    retrieval = tmp_path / "retrieval.parquet"
    pl.DataFrame([{"target_id": "TLONG", "final_score": 0.99}]).write_parquet(retrieval)
    tbm = tmp_path / "tbm.parquet"
    se3 = tmp_path / "se3.parquet"
    rows = []
    for model_id in range(1, 6):
        for resid, base in enumerate("AA", start=1):
            rows.append(
                {
                    "target_id": "TLONG",
                    "model_id": model_id,
                    "resid": resid,
                    "resname": base,
                    "x": float(model_id + resid),
                    "y": float(model_id + resid + 1),
                    "z": float(model_id + resid + 2),
                    "source": "tbm",
                    "confidence": 0.91,
                }
            )
    pl.DataFrame(rows).write_parquet(tbm)
    se3_rows = []
    for model_id in range(1, 6):
        for resid, base in enumerate("AA", start=1):
            se3_rows.append(
                {
                    "target_id": "TLONG",
                    "model_id": model_id,
                    "resid": resid,
                    "resname": base,
                    "x": float(model_id + resid + 10),
                    "y": float(model_id + resid + 11),
                    "z": float(model_id + resid + 12),
                    "source": "generative_se3",
                    "confidence": 0.85,
                }
            )
    pl.DataFrame(se3_rows).write_parquet(se3)

    out = build_hybrid_candidates(
        repo_root=tmp_path,
        targets_path=targets,
        retrieval_path=retrieval,
        tbm_path=tbm,
        out_path=tmp_path / "hybrid_candidates.parquet",
        routing_path=tmp_path / "routing.parquet",
        template_score_threshold=0.65,
        ultra_long_seq_threshold=1500,
        rnapro_path=None,
        chai1_path=None,
        boltz1_path=None,
        se3_path=se3,
    )
    routing = pl.read_parquet(out.routing_path)
    assert routing.item(0, "route_rule") == "len>1500->tbm+se3_mamba"
    assert routing.item(0, "primary_source") == "tbm+se3_mamba"
    assert routing.item(0, "fallback_used") is False
    candidates = pl.read_parquet(out.candidates_path)
    assert candidates.filter(pl.col("source") == "generative_se3").height > 0
    assert candidates.filter(pl.col("source") == "tbm").height > 0


def test_phase2_router_ultralong_falls_back_to_tbm_without_se3(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    _write_csv(
        targets,
        [
            {"target_id": "TLONG", "sequence": ("A" * 1601), "ligand_SMILES": ""},
        ],
    )
    retrieval = tmp_path / "retrieval.parquet"
    pl.DataFrame([{"target_id": "TLONG", "final_score": 0.99}]).write_parquet(retrieval)
    tbm = tmp_path / "tbm.parquet"
    pl.DataFrame(
        [
            {"target_id": "TLONG", "model_id": 1, "resid": 1, "resname": "A", "x": 1.0, "y": 2.0, "z": 3.0, "source": "tbm", "confidence": 0.9},
        ]
    ).write_parquet(tbm)
    out = build_hybrid_candidates(
        repo_root=tmp_path,
        targets_path=targets,
        retrieval_path=retrieval,
        tbm_path=tbm,
        out_path=tmp_path / "hybrid_candidates.parquet",
        routing_path=tmp_path / "routing.parquet",
        template_score_threshold=0.65,
        ultra_long_seq_threshold=1500,
        rnapro_path=None,
        chai1_path=None,
        boltz1_path=None,
        se3_path=None,
    )
    routing = pl.read_parquet(out.routing_path)
    assert routing.item(0, "route_rule") == "len>1500->tbm+se3_mamba_fallback->tbm"
    assert routing.item(0, "primary_source") == "tbm"
    assert routing.item(0, "fallback_used") is True


def test_phase2_router_long_fails_when_mamba_and_tbm_missing(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    _write_csv(
        targets,
        [
            {"target_id": "TLONG", "sequence": ("A" * 601), "ligand_SMILES": ""},
        ],
    )
    retrieval = tmp_path / "retrieval.parquet"
    pl.DataFrame([{"target_id": "TLONG", "final_score": 0.99}]).write_parquet(retrieval)
    tbm = tmp_path / "tbm.parquet"
    pl.DataFrame([], schema={"target_id": pl.Utf8, "model_id": pl.Int32, "resid": pl.Int32, "resname": pl.Utf8, "x": pl.Float64, "y": pl.Float64, "z": pl.Float64}).write_parquet(tbm)
    with pytest.raises(PipelineError, match="bucket long sem cobertura em se3_mamba e tbm"):
        build_hybrid_candidates(
            repo_root=tmp_path,
            targets_path=targets,
            retrieval_path=retrieval,
            tbm_path=tbm,
            out_path=tmp_path / "hybrid_candidates.parquet",
            routing_path=tmp_path / "routing.parquet",
            template_score_threshold=0.65,
            rnapro_path=None,
            chai1_path=None,
            boltz1_path=None,
            se3_mamba_path=None,
        )


def test_phase2_router_fails_when_medium_threshold_not_greater_than_short(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    _write_csv(targets, [{"target_id": "T1", "sequence": "AC", "ligand_SMILES": ""}])
    retrieval = tmp_path / "retrieval.parquet"
    pl.DataFrame([{"target_id": "T1", "final_score": 0.1}]).write_parquet(retrieval)
    tbm = tmp_path / "tbm.parquet"
    _make_tbm(tbm)
    with pytest.raises(PipelineError, match="medium_max_len deve ser maior que short_max_len"):
        build_hybrid_candidates(
            repo_root=tmp_path,
            targets_path=targets,
            retrieval_path=retrieval,
            tbm_path=tbm,
            out_path=tmp_path / "hybrid_candidates.parquet",
            routing_path=tmp_path / "routing.parquet",
            template_score_threshold=0.65,
            short_max_len=600,
            medium_max_len=600,
            rnapro_path=None,
            chai1_path=None,
            boltz1_path=None,
            se3_path=None,
        )
