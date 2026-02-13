from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

from rna3d_local.research import run_experiment


def test_run_experiment_generates_results_and_manifest(tmp_path: Path) -> None:
    config = {
        "experiment_name": "smoke_run",
        "solver": {
            "type": "command_json",
            "timeout_s": 30,
            "command": [
                sys.executable,
                "-c",
                "import json; seed=int('{seed}'); print(json.dumps(dict(instance_id='inst', solver_status='success', objective=float(seed)/1000.0, feasible=True, gap=0.0)))",
            ],
        },
        "seeds": [7, 9],
        "required_outputs": [],
        "repro_command": [sys.executable, "-c", "print('ok')"],
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")

    res = run_experiment(
        repo_root=tmp_path,
        config_path=cfg_path,
        run_id="run01",
        out_base_dir=tmp_path / "runs" / "research" / "experiments",
        allow_existing_run_dir=False,
    )

    assert res.results_path.exists()
    assert res.manifest_path.exists()
    df = pl.read_parquet(res.results_path)
    assert df.height == 2
    assert sorted(df.get_column("seed").to_list()) == [7, 9]
