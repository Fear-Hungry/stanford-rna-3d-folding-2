from __future__ import annotations

import math

import torch

from ..errors import raise_error
from .diffusion_se3 import Se3Diffusion
from .flow_matching_se3 import Se3FlowMatching


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
        for idx in range(count_diffusion):
            sample = diffusion.sample(h=h, x_cond=x_cond, seed=int(base_seed + idx))
            outputs.append(("diffusion", idx + 1, sample))
    if flow is not None:
        for idx in range(count_flow):
            sample = flow.sample(h=h, x_cond=x_cond, seed=int(base_seed + 100_000 + idx))
            outputs.append(("flow", idx + 1, sample))
    if not outputs:
        raise_error(stage, location, "nenhuma amostra gerada", impact="0", examples=[target_id])
    return outputs
