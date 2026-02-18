from __future__ import annotations

import json

import pytest
import torch

from rna3d_local.errors import PipelineError
from rna3d_local.training.config_se3 import load_se3_train_config
from rna3d_local.training.losses_se3 import compute_structural_loss_terms


def test_structural_loss_terms_near_zero_when_prediction_matches_truth() -> None:
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    chain_index = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    residue_index = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    terms = compute_structural_loss_terms(
        x_pred=coords,
        x_true=coords.clone(),
        chain_index=chain_index,
        residue_index=residue_index,
        fape_clamp_distance=10.0,
        fape_length_scale=10.0,
        vdw_min_distance=2.1,
        vdw_repulsion_power=4,
        loss_chunk_size=2,
        stage="TEST",
        location="tests/test_se3_losses.py:test_structural_loss_terms_near_zero_when_prediction_matches_truth",
    )
    assert float(terms.mse.item()) == pytest.approx(0.0, abs=1e-8)
    assert float(terms.fape.item()) == pytest.approx(0.0, abs=1e-6)
    assert float(terms.tm.item()) == pytest.approx(0.0, abs=1e-6)
    assert float(terms.clash.item()) == pytest.approx(0.0, abs=1e-8)


def test_structural_loss_terms_clash_penalizes_overlap() -> None:
    x_true = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    x_pred = x_true.clone()
    x_pred[0] = x_pred[3]
    chain_index = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    residue_index = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    terms = compute_structural_loss_terms(
        x_pred=x_pred,
        x_true=x_true,
        chain_index=chain_index,
        residue_index=residue_index,
        fape_clamp_distance=10.0,
        fape_length_scale=10.0,
        vdw_min_distance=2.1,
        vdw_repulsion_power=4,
        loss_chunk_size=2,
        stage="TEST",
        location="tests/test_se3_losses.py:test_structural_loss_terms_clash_penalizes_overlap",
    )
    assert float(terms.clash.item()) > 0.0
    assert float(terms.fape.item()) > 0.0


def test_structural_loss_terms_clash_exponential_is_stronger_below_2a() -> None:
    x_true = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    chain_index = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    residue_index = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    x_pred_mild = x_true.clone()
    x_pred_mild[0] = torch.tensor([7.10, 0.0, 0.0], dtype=torch.float32)
    x_pred_severe = x_true.clone()
    x_pred_severe[0] = torch.tensor([8.30, 0.0, 0.0], dtype=torch.float32)
    mild = compute_structural_loss_terms(
        x_pred=x_pred_mild,
        x_true=x_true,
        chain_index=chain_index,
        residue_index=residue_index,
        fape_clamp_distance=10.0,
        fape_length_scale=10.0,
        vdw_min_distance=2.1,
        vdw_repulsion_power=4,
        loss_chunk_size=2,
        stage="TEST",
        location="tests/test_se3_losses.py:test_structural_loss_terms_clash_exponential_is_stronger_below_2a:mild",
    )
    severe = compute_structural_loss_terms(
        x_pred=x_pred_severe,
        x_true=x_true,
        chain_index=chain_index,
        residue_index=residue_index,
        fape_clamp_distance=10.0,
        fape_length_scale=10.0,
        vdw_min_distance=2.1,
        vdw_repulsion_power=4,
        loss_chunk_size=2,
        stage="TEST",
        location="tests/test_se3_losses.py:test_structural_loss_terms_clash_exponential_is_stronger_below_2a:severe",
    )
    assert float(severe.clash.item()) > float(mild.clash.item())


def test_structural_loss_terms_fail_for_invalid_chunk_size() -> None:
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    chain_index = torch.tensor([0, 0, 0], dtype=torch.long)
    residue_index = torch.tensor([0, 1, 2], dtype=torch.long)
    with pytest.raises(PipelineError, match="loss_chunk_size"):
        compute_structural_loss_terms(
            x_pred=coords,
            x_true=coords.clone(),
            chain_index=chain_index,
            residue_index=residue_index,
            fape_clamp_distance=10.0,
            fape_length_scale=10.0,
            vdw_min_distance=2.1,
            vdw_repulsion_power=4,
            loss_chunk_size=0,
            stage="TEST",
            location="tests/test_se3_losses.py:test_structural_loss_terms_fail_for_invalid_chunk_size",
        )


def test_config_fails_when_all_structural_weights_are_zero(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "hidden_dim": 16,
                "num_layers": 1,
                "ipa_heads": 4,
                "diffusion_steps": 4,
                "flow_steps": 4,
                "epochs": 1,
                "learning_rate": 1e-3,
                "method": "diffusion",
                "thermo_backend": "rnafold",
                "msa_backend": "mmseqs2",
                "mmseqs_db": "/tmp/fake_db",
                "loss_weight_mse": 0.0,
                "loss_weight_fape": 0.0,
                "loss_weight_tm": 0.0,
                "loss_weight_clash": 0.0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PipelineError, match="soma dos loss_weight"):
        load_se3_train_config(
            config,
            stage="TEST",
            location="tests/test_se3_losses.py:test_config_fails_when_all_structural_weights_are_zero",
        )


def test_config_local_16gb_fails_for_invalid_accumulation(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "hidden_dim": 16,
                "num_layers": 1,
                "ipa_heads": 4,
                "diffusion_steps": 4,
                "flow_steps": 4,
                "epochs": 1,
                "learning_rate": 1e-3,
                "method": "diffusion",
                "thermo_backend": "rnafold",
                "msa_backend": "mmseqs2",
                "mmseqs_db": "/tmp/fake_db",
                "training_protocol": "local_16gb",
                "dynamic_cropping": True,
                "crop_min_length": 256,
                "crop_max_length": 384,
                "graph_backend": "torch_geometric",
                "use_gradient_checkpointing": True,
                "ipa_edge_chunk_size": 128,
                "autocast_bfloat16": True,
                "gradient_accumulation_steps": 8,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PipelineError, match="gradient_accumulation_steps"):
        load_se3_train_config(
            config,
            stage="TEST",
            location="tests/test_se3_losses.py:test_config_local_16gb_fails_for_invalid_accumulation",
        )


def test_config_local_16gb_accepts_expected_window(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "hidden_dim": 16,
                "num_layers": 1,
                "ipa_heads": 4,
                "diffusion_steps": 4,
                "flow_steps": 4,
                "epochs": 1,
                "learning_rate": 1e-3,
                "method": "diffusion",
                "thermo_backend": "rnafold",
                "msa_backend": "mmseqs2",
                "mmseqs_db": "/tmp/fake_db",
                "training_protocol": "local_16gb",
                "dynamic_cropping": True,
                "crop_min_length": 256,
                "crop_max_length": 384,
                "graph_backend": "torch_geometric",
                "use_gradient_checkpointing": True,
                "ipa_edge_chunk_size": 128,
                "autocast_bfloat16": True,
                "gradient_accumulation_steps": 16,
            }
        ),
        encoding="utf-8",
    )
    cfg = load_se3_train_config(
        config,
        stage="TEST",
        location="tests/test_se3_losses.py:test_config_local_16gb_accepts_expected_window",
    )
    assert cfg.training_protocol == "local_16gb"


def test_config_local_16gb_requires_torch_geometric_backend(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "hidden_dim": 16,
                "num_layers": 1,
                "ipa_heads": 4,
                "diffusion_steps": 4,
                "flow_steps": 4,
                "epochs": 1,
                "learning_rate": 1e-3,
                "method": "diffusion",
                "thermo_backend": "rnafold",
                "msa_backend": "mmseqs2",
                "mmseqs_db": "/tmp/fake_db",
                "training_protocol": "local_16gb",
                "dynamic_cropping": True,
                "crop_min_length": 256,
                "crop_max_length": 384,
                "graph_backend": "torch_sparse",
                "use_gradient_checkpointing": True,
                "ipa_edge_chunk_size": 128,
                "autocast_bfloat16": True,
                "gradient_accumulation_steps": 16,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PipelineError, match="graph_backend=torch_geometric"):
        load_se3_train_config(
            config,
            stage="TEST",
            location="tests/test_se3_losses.py:test_config_local_16gb_requires_torch_geometric_backend",
        )


def test_config_local_16gb_requires_ipa_chunk_upto_256(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "hidden_dim": 16,
                "num_layers": 1,
                "ipa_heads": 4,
                "diffusion_steps": 4,
                "flow_steps": 4,
                "epochs": 1,
                "learning_rate": 1e-3,
                "method": "diffusion",
                "thermo_backend": "rnafold",
                "msa_backend": "mmseqs2",
                "mmseqs_db": "/tmp/fake_db",
                "training_protocol": "local_16gb",
                "dynamic_cropping": True,
                "crop_min_length": 256,
                "crop_max_length": 384,
                "graph_backend": "torch_geometric",
                "use_gradient_checkpointing": True,
                "ipa_edge_chunk_size": 512,
                "autocast_bfloat16": True,
                "gradient_accumulation_steps": 16,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PipelineError, match="ipa_edge_chunk_size <= 256"):
        load_se3_train_config(
            config,
            stage="TEST",
            location="tests/test_se3_losses.py:test_config_local_16gb_requires_ipa_chunk_upto_256",
        )


def test_config_fails_for_negative_thermo_soft_constraint_strength(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "hidden_dim": 16,
                "num_layers": 1,
                "ipa_heads": 4,
                "diffusion_steps": 4,
                "flow_steps": 4,
                "epochs": 1,
                "learning_rate": 1e-3,
                "method": "diffusion",
                "thermo_backend": "rnafold",
                "msa_backend": "mmseqs2",
                "mmseqs_db": "/tmp/fake_db",
                "thermo_soft_constraint_strength": -0.1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PipelineError, match="thermo_soft_constraint_strength invalido"):
        load_se3_train_config(
            config,
            stage="TEST",
            location="tests/test_se3_losses.py:test_config_fails_for_negative_thermo_soft_constraint_strength",
        )
