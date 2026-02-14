from __future__ import annotations

import types

import pytest

from rna3d_local.kaggle_submissions import list_kaggle_submissions
from rna3d_local.errors import PipelineError


def _install_fake_kaggle_api(monkeypatch: pytest.MonkeyPatch, raw_submissions: list[dict]) -> None:
    class _FakeSub:
        def __init__(self, d: dict):
            self._d = dict(d)

        def to_dict(self) -> dict:
            return dict(self._d)

    class _FakeApi:
        def authenticate(self) -> None:
            return

        def competition_submissions(self, *, competition: str, page_token: str = "", page_size: int = 20):  # noqa: ARG002
            return [_FakeSub(x) for x in raw_submissions[: int(page_size)]]

    fake_ext = types.ModuleType("kaggle.api.kaggle_api_extended")
    fake_api_pkg = types.ModuleType("kaggle.api")
    fake_kaggle_pkg = types.ModuleType("kaggle")

    fake_ext.KaggleApi = _FakeApi
    fake_api_pkg.kaggle_api_extended = fake_ext
    fake_kaggle_pkg.api = fake_api_pkg

    monkeypatch.setitem(__import__("sys").modules, "kaggle", fake_kaggle_pkg)
    monkeypatch.setitem(__import__("sys").modules, "kaggle.api", fake_api_pkg)
    monkeypatch.setitem(__import__("sys").modules, "kaggle.api.kaggle_api_extended", fake_ext)

def _install_fake_kaggle_api_unknown_item(monkeypatch: pytest.MonkeyPatch) -> None:
    class _WeirdSub:
        pass

    class _FakeApi:
        def authenticate(self) -> None:
            return

        def competition_submissions(self, *, competition: str, page_token: str = "", page_size: int = 20):  # noqa: ARG002
            return [_WeirdSub()]

    fake_ext = types.ModuleType("kaggle.api.kaggle_api_extended")
    fake_api_pkg = types.ModuleType("kaggle.api")
    fake_kaggle_pkg = types.ModuleType("kaggle")

    fake_ext.KaggleApi = _FakeApi
    fake_api_pkg.kaggle_api_extended = fake_ext
    fake_kaggle_pkg.api = fake_api_pkg

    monkeypatch.setitem(__import__("sys").modules, "kaggle", fake_kaggle_pkg)
    monkeypatch.setitem(__import__("sys").modules, "kaggle.api", fake_api_pkg)
    monkeypatch.setitem(__import__("sys").modules, "kaggle.api.kaggle_api_extended", fake_ext)


def test_list_kaggle_submissions_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_kaggle_api(
        monkeypatch,
        [
            {"ref": "1", "status": "complete", "publicScore": "0.26", "description": "a"},
            {"ref": "2", "status": "complete", "publicScore": None, "description": "b"},
        ],
    )
    out = list_kaggle_submissions(competition="x", page_size=10)
    assert out["competition"] == "x"
    assert out["n"] == 2
    assert out["submissions"][0]["ref"] == "1"


def test_list_kaggle_submissions_rejects_bad_page_size(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_kaggle_api(monkeypatch, [])
    with pytest.raises(PipelineError):
        list_kaggle_submissions(competition="x", page_size=0)


def test_list_kaggle_submissions_rejects_unknown_item_type(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_kaggle_api_unknown_item(monkeypatch)
    with pytest.raises(PipelineError):
        list_kaggle_submissions(competition="x", page_size=10)
