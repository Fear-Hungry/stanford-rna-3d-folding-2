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
    _touch(rna / "model.pt")
    _write_stub_runner(rna, source="rnapro")
    chai = base / "chai1"
    _touch(chai / "model.bin")
    _write_stub_runner(chai, source="chai1")
    boltz = base / "boltz1"
    _touch(boltz / "model.safetensors")
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
    assert rules["T1"] == "template->tbm"
    assert rules["T2"] == "orphan->chai1+boltz1"
    assert rules["T3"] == "ligand->boltz1"

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


def test_phase2_router_falls_back_when_template_strong_without_tbm_coverage(tmp_path: Path) -> None:
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
            {"target_id": "T1", "final_score": 0.95},
            {"target_id": "T2", "final_score": 0.10},
        ]
    ).write_parquet(retrieval)
    tbm = tmp_path / "tbm.parquet"
    _make_tbm(tbm)
    tbm_df = pl.read_parquet(tbm).filter(pl.col("target_id") != "T1")
    tbm_df.write_parquet(tbm)

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
    assert rules["T1"] == "template_missing->chai1+boltz1"


def test_phase2_router_prefers_se3_when_available(tmp_path: Path) -> None:
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
    assert rules["T2"] == "orphan_or_weak_template->generative_se3"


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
    assert routing.item(0, "route_rule") == "ultralong->generative_se3"
    assert routing.item(0, "primary_source") == "generative_se3"
    candidates = pl.read_parquet(out.candidates_path)
    assert candidates.filter(pl.col("source") != "generative_se3").height == 0


def test_phase2_router_ultralong_fails_without_se3(tmp_path: Path) -> None:
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
    with pytest.raises(PipelineError, match="ultralongo exige se3_path"):
        build_hybrid_candidates(
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
