from __future__ import annotations

from contextlib import nullcontext

import pytest

from rna3d_local.errors import PipelineError
from rna3d_local.experiments import runner
from rna3d_local.experiments.runner import safe_predict


class _FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self, *, available: bool = True) -> None:
        self._available = bool(available)
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return bool(self._available)

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeTorch:
    def __init__(self, *, cuda_available: bool = True) -> None:
        self.cuda = _FakeCuda(available=cuda_available)
        self.bfloat16 = object()

    def inference_mode(self):
        return nullcontext()

    def autocast(self, *, device_type: str, dtype: object):
        return nullcontext()


def test_safe_predict_success_uses_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = _FakeTorch(cuda_available=True)

    class _DummyRunner:
        moved_to_cpu: list[str] = []

        def predict(self, sequence: str, *args: object, **kwargs: object):
            return {"sequence_len": len(sequence)}

        def to(self, device: str) -> None:
            self.__class__.moved_to_cpu.append(str(device))

    monkeypatch.setattr(runner, "_load_torch_module", lambda: fake_torch)
    out = safe_predict(_DummyRunner, "ACGU")
    assert out.ok is True
    assert out.error_kind is None
    assert out.model_name == "_DummyRunner"
    assert out.prediction == {"sequence_len": 4}
    assert fake_torch.cuda.empty_cache_calls == 1
    assert _DummyRunner.moved_to_cpu == ["cpu"]


def test_safe_predict_returns_explicit_oom_result(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = _FakeTorch(cuda_available=True)

    class _OomRunner:
        def predict(self, sequence: str, *args: object, **kwargs: object):
            raise fake_torch.cuda.OutOfMemoryError("oom")

    monkeypatch.setattr(runner, "_load_torch_module", lambda: fake_torch)
    out = safe_predict(_OomRunner, "A" * 1200)
    assert out.ok is False
    assert out.prediction is None
    assert out.error_kind == "oom"
    assert "OOM" in str(out.error_message)
    assert fake_torch.cuda.empty_cache_calls == 1


def test_safe_predict_fails_fast_on_non_oom_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = _FakeTorch(cuda_available=True)

    class _BrokenRunner:
        def predict(self, sequence: str, *args: object, **kwargs: object):
            raise RuntimeError("boom")

    monkeypatch.setattr(runner, "_load_torch_module", lambda: fake_torch)
    with pytest.raises(PipelineError, match="falha na inferencia do modelo"):
        safe_predict(_BrokenRunner, "ACGU")
