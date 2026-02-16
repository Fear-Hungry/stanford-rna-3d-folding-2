from __future__ import annotations

import math

import torch

from ..errors import raise_error
from .diffusion_se3 import Se3Diffusion
from .flow_matching_se3 import Se3FlowMatching


def _resolve_fast_steps(*, total_steps: int, n_samples: int) -> int:
    if total_steps <= 1:
        return 2
    if n_samples >= 24:
        return max(4, min(total_steps, 8))
    if n_samples >= 16:
        return max(4, min(total_steps, 10))
    return max(4, min(total_steps, 12))


@torch.no_grad()
def _sample_diffusion_dpm_like(
    *,
    model: Se3Diffusion,
    h: torch.Tensor,
    x_cond: torch.Tensor,
    seed: int,
    fast_steps: int,
) -> torch.Tensor:
    generator = torch.Generator(device=x_cond.device)
    generator.manual_seed(int(seed))
    x = torch.randn(x_cond.shape, generator=generator, device=x_cond.device, dtype=x_cond.dtype)
    total_steps = int(model.num_steps)
    step_ids = torch.linspace(float(total_steps - 1), 0.0, steps=int(fast_steps), device=x_cond.device).round().to(dtype=torch.long)
    step_ids = torch.unique_consecutive(step_ids)
    if int(step_ids[-1].item()) != 0:
        step_ids = torch.cat([step_ids, torch.zeros((1,), dtype=torch.long, device=x_cond.device)], dim=0)
    for cursor in range(int(step_ids.shape[0]) - 1):
        idx_cur = int(step_ids[cursor].item())
        idx_next = int(step_ids[cursor + 1].item())
        tau_cur = torch.tensor([idx_cur / float(total_steps - 1)], device=x_cond.device, dtype=x_cond.dtype)
        eps_cur = model._predict_noise(h, x, x_cond, tau_cur)
        alpha_cur = torch.clamp(model.alpha_hat[idx_cur].to(dtype=x_cond.dtype), min=1e-8, max=1.0)
        alpha_next = torch.clamp(model.alpha_hat[idx_next].to(dtype=x_cond.dtype), min=1e-8, max=1.0)
        x0 = (x - torch.sqrt(torch.clamp(1.0 - alpha_cur, min=1e-8)) * eps_cur) / torch.sqrt(alpha_cur)
        x = (torch.sqrt(alpha_next) * x0) + (torch.sqrt(torch.clamp(1.0 - alpha_next, min=1e-8)) * eps_cur)
        if idx_next > 0:
            noise = torch.randn(x.shape, generator=generator, device=x_cond.device, dtype=x_cond.dtype)
            x = x + (0.01 * noise)
    return x


@torch.no_grad()
def _sample_flow_heun(
    *,
    model: Se3FlowMatching,
    h: torch.Tensor,
    x_cond: torch.Tensor,
    seed: int,
    fast_steps: int,
) -> torch.Tensor:
    generator = torch.Generator(device=x_cond.device)
    generator.manual_seed(int(seed))
    x = x_cond + (0.03 * torch.randn(x_cond.shape, generator=generator, device=x_cond.device, dtype=x_cond.dtype))
    tau_grid = torch.linspace(0.0, 1.0, steps=int(fast_steps), device=x_cond.device, dtype=x_cond.dtype)
    for cursor in range(int(tau_grid.shape[0]) - 1):
        tau_cur = tau_grid[cursor].view(1)
        tau_next = tau_grid[cursor + 1].view(1)
        dt = tau_next - tau_cur
        vel_cur = model._predict_velocity(h, x, x_cond, tau_cur)
        x_euler = x + (dt * vel_cur)
        vel_next = model._predict_velocity(h, x_euler, x_cond, tau_next)
        x = x + (0.5 * dt * (vel_cur + vel_next))
    return x


def _validate_sample_tensor(
    *,
    sample: torch.Tensor,
    x_cond: torch.Tensor,
    stage: str,
    location: str,
    target_id: str,
    method_name: str,
    sample_rank: int,
) -> None:
    if sample.shape != x_cond.shape:
        raise_error(
            stage,
            location,
            "sample com shape divergente de x_cond",
            impact="1",
            examples=[f"{target_id}:{method_name}:{sample_rank}:sample={tuple(sample.shape)}:x_cond={tuple(x_cond.shape)}"],
        )
    if not bool(torch.isfinite(sample).all()):
        raise_error(
            stage,
            location,
            "sample contem coordenadas nao-finitas",
            impact="1",
            examples=[f"{target_id}:{method_name}:{sample_rank}"],
        )


def sample_methods_for_target(
    *,
    target_id: str,
    h: torch.Tensor,
    x_cond: torch.Tensor,
    method: str,
    n_samples: int,
    base_seed: int,
    diffusion: Se3Diffusion | None,
    flow: Se3FlowMatching | None,
    stage: str,
    location: str,
) -> list[tuple[str, int, torch.Tensor]]:
    if n_samples <= 0:
        raise_error(stage, location, "n_samples deve ser > 0", impact="1", examples=[str(n_samples)])
    mode = str(method).strip().lower()
    if mode not in {"diffusion", "flow", "both"}:
        raise_error(stage, location, "method invalido", impact="1", examples=[method])
    if mode in {"diffusion", "both"} and diffusion is None:
        raise_error(stage, location, "modelo diffusion ausente para amostragem", impact="1", examples=[target_id])
    if mode in {"flow", "both"} and flow is None:
        raise_error(stage, location, "modelo flow ausente para amostragem", impact="1", examples=[target_id])

    count_diffusion = int(n_samples) if mode == "diffusion" else (math.ceil(n_samples / 2) if mode == "both" else 0)
    count_flow = int(n_samples) if mode == "flow" else (n_samples - count_diffusion if mode == "both" else 0)
    outputs: list[tuple[str, int, torch.Tensor]] = []

    if diffusion is not None:
        diffusion_fast_steps = _resolve_fast_steps(total_steps=int(diffusion.num_steps), n_samples=int(n_samples))
        for idx in range(count_diffusion):
            sample = _sample_diffusion_dpm_like(
                model=diffusion,
                h=h,
                x_cond=x_cond,
                seed=int(base_seed + idx),
                fast_steps=int(diffusion_fast_steps),
            )
            _validate_sample_tensor(
                sample=sample,
                x_cond=x_cond,
                stage=stage,
                location=location,
                target_id=str(target_id),
                method_name="diffusion",
                sample_rank=idx + 1,
            )
            outputs.append(("diffusion", idx + 1, sample))
    if flow is not None:
        flow_fast_steps = _resolve_fast_steps(total_steps=int(flow.num_steps), n_samples=int(n_samples))
        for idx in range(count_flow):
            sample = _sample_flow_heun(
                model=flow,
                h=h,
                x_cond=x_cond,
                seed=int(base_seed + 100_000 + idx),
                fast_steps=int(flow_fast_steps),
            )
            _validate_sample_tensor(
                sample=sample,
                x_cond=x_cond,
                stage=stage,
                location=location,
                target_id=str(target_id),
                method_name="flow",
                sample_rank=idx + 1,
            )
            outputs.append(("flow", idx + 1, sample))
    if not outputs:
        raise_error(stage, location, "nenhuma amostra gerada", impact="0", examples=[target_id])
    return outputs
