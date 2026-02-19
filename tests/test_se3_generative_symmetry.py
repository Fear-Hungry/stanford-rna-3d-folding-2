from __future__ import annotations

import math

import torch

from rna3d_local.generative.diffusion_se3 import Se3Diffusion
from rna3d_local.generative.flow_matching_se3 import Se3FlowMatching


def _rotation_z(angle_rad: float) -> torch.Tensor:
    c = float(math.cos(float(angle_rad)))
    s = float(math.sin(float(angle_rad)))
    return torch.tensor(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _make_non_degenerate_chain(n: int) -> torch.Tensor:
    t = torch.linspace(0.0, 2.0 * math.pi, n)
    x = torch.linspace(0.0, 6.0, n)
    y = 0.8 * torch.sin(t)
    z = 0.8 * torch.cos(t)
    return torch.stack([x, y, z], dim=-1).to(dtype=torch.float32)


def test_diffusion_predict_noise_is_se3_equivariant() -> None:
    torch.manual_seed(0)
    n = 7
    hidden = 8
    model = Se3Diffusion(hidden_dim=hidden, num_steps=8).eval()

    x_cond = _make_non_degenerate_chain(n)
    x_noisy = x_cond + 0.1 * torch.randn(n, 3)
    h = torch.randn(n, hidden)
    t = torch.tensor([0.4], dtype=torch.float32)

    delta_noisy = x_noisy - x_cond
    noise = model._predict_noise(h, delta_noisy, x_cond, t)

    r = _rotation_z(0.7)
    x_cond_r = x_cond @ r.transpose(0, 1)
    delta_noisy_r = delta_noisy @ r.transpose(0, 1)
    noise_r = model._predict_noise(h, delta_noisy_r, x_cond_r, t)
    assert torch.allclose(noise_r, noise @ r.transpose(0, 1), atol=2e-4, rtol=2e-4)

    shift = torch.tensor([[3.0, -2.0, 1.0]], dtype=torch.float32)
    noise_t = model._predict_noise(h, delta_noisy, x_cond + shift, t)
    assert torch.allclose(noise_t, noise, atol=2e-4, rtol=2e-4)


def test_flow_predict_velocity_is_se3_equivariant() -> None:
    torch.manual_seed(0)
    n = 7
    hidden = 8
    model = Se3FlowMatching(hidden_dim=hidden, num_steps=8).eval()

    x_cond = _make_non_degenerate_chain(n)
    x_t = x_cond + 0.2 * torch.randn(n, 3)
    h = torch.randn(n, hidden)
    tau = torch.tensor([0.25], dtype=torch.float32)

    vel = model._predict_velocity(h, x_t, x_cond, tau)

    r = _rotation_z(0.7)
    x_cond_r = x_cond @ r.transpose(0, 1)
    x_t_r = x_t @ r.transpose(0, 1)
    vel_r = model._predict_velocity(h, x_t_r, x_cond_r, tau)
    assert torch.allclose(vel_r, vel @ r.transpose(0, 1), atol=2e-4, rtol=2e-4)

    shift = torch.tensor([[3.0, -2.0, 1.0]], dtype=torch.float32)
    vel_t = model._predict_velocity(h, x_t + shift, x_cond + shift, tau)
    assert torch.allclose(vel_t, vel, atol=2e-4, rtol=2e-4)


def test_diffusion_predict_noise_uses_neighbor_context() -> None:
    torch.manual_seed(0)
    n = 8
    hidden = 8
    model = Se3Diffusion(hidden_dim=hidden, num_steps=8).eval()

    x_cond = _make_non_degenerate_chain(n)
    h = torch.randn(n, hidden)
    t = torch.tensor([0.5], dtype=torch.float32)
    delta_base = 0.1 * torch.randn(n, 3)
    noise_base = model._predict_noise(h, delta_base, x_cond, t)

    delta_changed = delta_base.clone()
    delta_changed[6] = delta_changed[6] + torch.tensor([0.4, -0.2, 0.3], dtype=torch.float32)
    noise_changed = model._predict_noise(h, delta_changed, x_cond, t)

    # Residuo 2 manteve as mesmas features locais; variacao vem da mensagem dos vizinhos.
    delta_norm = float(torch.linalg.norm(noise_changed[2] - noise_base[2]).item())
    assert delta_norm > 1e-5


def test_flow_predict_velocity_uses_neighbor_context() -> None:
    torch.manual_seed(0)
    n = 8
    hidden = 8
    model = Se3FlowMatching(hidden_dim=hidden, num_steps=8).eval()

    x_cond = _make_non_degenerate_chain(n)
    h = torch.randn(n, hidden)
    tau = torch.tensor([0.3], dtype=torch.float32)
    x_t_base = x_cond + (0.2 * torch.randn(n, 3))
    vel_base = model._predict_velocity(h, x_t_base, x_cond, tau)

    x_t_changed = x_t_base.clone()
    x_t_changed[6] = x_t_changed[6] + torch.tensor([0.3, -0.2, 0.35], dtype=torch.float32)
    vel_changed = model._predict_velocity(h, x_t_changed, x_cond, tau)

    # Residuo 2 manteve as mesmas features locais; variacao vem da mensagem dos vizinhos.
    delta_norm = float(torch.linalg.norm(vel_changed[2] - vel_base[2]).item())
    assert delta_norm > 1e-5


def test_diffusion_forward_loss_is_rigid_invariant_in_x_true() -> None:
    torch.manual_seed(0)
    n = 7
    hidden = 8
    model = Se3Diffusion(hidden_dim=hidden, num_steps=8).eval()

    x_cond = _make_non_degenerate_chain(n)
    x_true = x_cond + (0.35 * torch.randn(n, 3))
    h = torch.randn(n, hidden)
    r = _rotation_z(0.7)
    shift = torch.tensor([[3.0, -2.0, 1.0]], dtype=torch.float32)
    x_true_rt = (x_true @ r.transpose(0, 1)) + shift

    torch.manual_seed(1234)
    loss_a = model.forward_loss(h, x_cond, x_true)
    torch.manual_seed(1234)
    loss_b = model.forward_loss(h, x_cond, x_true_rt)
    assert torch.allclose(loss_a, loss_b, atol=1e-5, rtol=1e-5)


def test_flow_forward_loss_is_rigid_invariant_in_x_true() -> None:
    torch.manual_seed(0)
    n = 7
    hidden = 8
    model = Se3FlowMatching(hidden_dim=hidden, num_steps=8).eval()

    x_cond = _make_non_degenerate_chain(n)
    x_true = x_cond + (0.35 * torch.randn(n, 3))
    h = torch.randn(n, hidden)
    r = _rotation_z(0.7)
    shift = torch.tensor([[3.0, -2.0, 1.0]], dtype=torch.float32)
    x_true_rt = (x_true @ r.transpose(0, 1)) + shift

    torch.manual_seed(1234)
    loss_a = model.forward_loss(h, x_cond, x_true)
    torch.manual_seed(1234)
    loss_b = model.forward_loss(h, x_cond, x_true_rt)
    assert torch.allclose(loss_a, loss_b, atol=1e-5, rtol=1e-5)


def test_diffusion_sample_is_se3_equivariant_for_fixed_seed() -> None:
    torch.manual_seed(0)
    n = 7
    hidden = 8
    model = Se3Diffusion(hidden_dim=hidden, num_steps=6).eval()

    x_cond = _make_non_degenerate_chain(n)
    h = torch.randn(n, hidden)
    seed = 123
    x = model.sample(h, x_cond, seed=seed)

    r = _rotation_z(0.7)
    x_cond_r = x_cond @ r.transpose(0, 1)
    x_r = model.sample(h, x_cond_r, seed=seed)
    assert torch.allclose(x_r, x @ r.transpose(0, 1), atol=4e-4, rtol=4e-4)

    shift = torch.tensor([[3.0, -2.0, 1.0]], dtype=torch.float32)
    x_t = model.sample(h, x_cond + shift, seed=seed)
    assert torch.allclose(x_t, x + shift, atol=4e-4, rtol=4e-4)


def test_flow_sample_is_se3_equivariant_for_fixed_seed() -> None:
    torch.manual_seed(0)
    n = 7
    hidden = 8
    model = Se3FlowMatching(hidden_dim=hidden, num_steps=6).eval()

    x_cond = _make_non_degenerate_chain(n)
    h = torch.randn(n, hidden)
    seed = 123
    x = model.sample(h, x_cond, seed=seed)

    r = _rotation_z(0.7)
    x_cond_r = x_cond @ r.transpose(0, 1)
    x_r = model.sample(h, x_cond_r, seed=seed)
    assert torch.allclose(x_r, x @ r.transpose(0, 1), atol=4e-4, rtol=4e-4)

    shift = torch.tensor([[3.0, -2.0, 1.0]], dtype=torch.float32)
    x_t = model.sample(h, x_cond + shift, seed=seed)
    assert torch.allclose(x_t, x + shift, atol=4e-4, rtol=4e-4)
