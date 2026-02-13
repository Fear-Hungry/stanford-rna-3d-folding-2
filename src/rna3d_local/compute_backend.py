from __future__ import annotations

from dataclasses import dataclass

from .errors import raise_error

_ALLOWED_BACKENDS = {"auto", "cpu", "cuda"}
_ALLOWED_PRECISIONS = {"fp32", "fp16"}


@dataclass(frozen=True)
class ComputeBackend:
    requested: str
    backend: str
    device: str
    precision: str
    torch_available: bool
    cuda_available: bool
    gpu_name: str | None
    gpu_memory_budget_mb: int

    def to_manifest_dict(self) -> dict:
        return {
            "requested": str(self.requested),
            "backend": str(self.backend),
            "device": str(self.device),
            "precision": str(self.precision),
            "torch_available": bool(self.torch_available),
            "cuda_available": bool(self.cuda_available),
            "gpu_name": None if self.gpu_name is None else str(self.gpu_name),
            "gpu_memory_budget_mb": int(self.gpu_memory_budget_mb),
        }


def _load_torch_state() -> tuple[bool, bool, str | None]:
    try:
        import torch  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False, False, None
    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_ok = False
    if not cuda_ok:
        return True, False, None
    try:
        name = str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001
        name = None
    return True, True, name


def resolve_compute_backend(
    *,
    requested: str,
    precision: str,
    gpu_memory_budget_mb: int,
    stage: str,
    location: str,
) -> ComputeBackend:
    req = str(requested).strip().lower()
    if req not in _ALLOWED_BACKENDS:
        raise_error(
            stage,
            location,
            "compute_backend invalido",
            impact="1",
            examples=[str(requested), ",".join(sorted(_ALLOWED_BACKENDS))],
        )
    prec = str(precision).strip().lower()
    if prec not in _ALLOWED_PRECISIONS:
        raise_error(
            stage,
            location,
            "gpu_precision invalida",
            impact="1",
            examples=[str(precision), ",".join(sorted(_ALLOWED_PRECISIONS))],
        )
    try:
        gpu_budget_i = int(gpu_memory_budget_mb)
    except (TypeError, ValueError):
        raise_error(stage, location, "gpu_memory_budget_mb invalido", impact="1", examples=[str(gpu_memory_budget_mb)])
    if gpu_budget_i <= 0:
        raise_error(stage, location, "gpu_memory_budget_mb deve ser > 0", impact="1", examples=[str(gpu_budget_i)])

    torch_available, cuda_available, gpu_name = _load_torch_state()
    if req == "cpu":
        return ComputeBackend(
            requested=req,
            backend="cpu",
            device="cpu",
            precision=prec,
            torch_available=bool(torch_available),
            cuda_available=bool(cuda_available),
            gpu_name=None,
            gpu_memory_budget_mb=int(gpu_budget_i),
        )
    if req == "cuda":
        if not torch_available:
            raise_error(
                stage,
                location,
                "compute_backend=cuda requer torch instalado",
                impact="1",
                examples=["pip install '.[gpu]'"],
            )
        if not cuda_available:
            raise_error(
                stage,
                location,
                "compute_backend=cuda solicitado sem CUDA disponivel",
                impact="1",
                examples=["torch.cuda.is_available=False"],
            )
        return ComputeBackend(
            requested=req,
            backend="cuda",
            device="cuda",
            precision=prec,
            torch_available=True,
            cuda_available=True,
            gpu_name=gpu_name,
            gpu_memory_budget_mb=int(gpu_budget_i),
        )

    # auto
    if torch_available and cuda_available:
        return ComputeBackend(
            requested=req,
            backend="cuda",
            device="cuda",
            precision=prec,
            torch_available=True,
            cuda_available=True,
            gpu_name=gpu_name,
            gpu_memory_budget_mb=int(gpu_budget_i),
        )
    return ComputeBackend(
        requested=req,
        backend="cpu",
        device="cpu",
        precision=prec,
        torch_available=bool(torch_available),
        cuda_available=bool(cuda_available),
        gpu_name=None,
        gpu_memory_budget_mb=int(gpu_budget_i),
    )

