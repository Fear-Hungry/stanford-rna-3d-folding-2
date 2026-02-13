from __future__ import annotations

import json
import sys
from pathlib import Path

from rna3d_local.research import generate_report, run_experiment, verify_run


def test_generate_report_contains_required_sections(tmp_path: Path) -> None:
    config = {
        "experiment_name": "smoke_report",
        "solver": {
            "type": "command_json",
            "timeout_s": 30,
            "command": [
                sys.executable,
                "-c",
                "import json; print(json.dumps(dict(instance_id='inst', solver_status='success', objective=0.55, feasible=True, gap=0.0)))",
            ],
        },
        "seeds": [42],
        "required_outputs": [],
        "repro_command": [sys.executable, "-c", "print('ok')"],
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")

    run = run_experiment(
        repo_root=tmp_path,
        config_path=cfg_path,
        run_id="run_report",
        out_base_dir=tmp_path / "runs" / "research" / "experiments",
        allow_existing_run_dir=False,
    )
    verify_run(repo_root=tmp_path, run_dir=run.run_dir, allowed_statuses=("success",))

    report_path = tmp_path / "report.md"
    out = generate_report(run_dir=run.run_dir, out_path=report_path)

    text = out.read_text(encoding="utf-8")
    assert "## Verification" in text
    assert "## Results" in text
    assert "## Literature" in text
