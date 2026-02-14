from __future__ import annotations

from datetime import datetime, timezone

from .errors import raise_error


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def list_kaggle_submissions(
    *,
    competition: str,
    page_size: int = 50,
    page_token: str = "",
) -> dict:
    """
    List recent competition submissions using KaggleApi (Python), without relying on
    the `kaggle competitions submissions` CLI subcommand (which can break due to
    upstream CLI/API signature mismatches).
    """
    location = "src/rna3d_local/kaggle_submissions.py:list_kaggle_submissions"
    if not str(competition or "").strip():
        raise_error("KAGGLE", location, "competition vazio", impact="1", examples=[str(competition)])
    if int(page_size) <= 0:
        raise_error("KAGGLE", location, "page_size invalido", impact="1", examples=[str(page_size)])
    token = str(page_token or "")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:  # noqa: BLE001
        raise_error("KAGGLE", location, "falha ao importar KaggleApi", impact="1", examples=[f"{type(e).__name__}:{e}"])

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:  # noqa: BLE001
        raise_error("KAGGLE", location, "falha ao autenticar Kaggle API", impact="1", examples=[f"{type(e).__name__}:{e}"])

    try:
        subs = api.competition_submissions(competition=str(competition), page_token=token, page_size=int(page_size))
    except Exception as e:  # noqa: BLE001
        raise_error(
            "KAGGLE",
            location,
            "falha ao listar submissões da competicao via KaggleApi",
            impact="1",
            examples=[f"{type(e).__name__}:{e}", str(competition)],
        )
    if subs is None:
        raise_error(
            "KAGGLE",
            location,
            "KaggleApi retornou None em competition_submissions",
            impact="1",
            examples=[str(competition)],
        )

    items: list[dict] = []
    for sub in subs:
        if sub is None:
            continue
        if hasattr(sub, "to_dict"):
            d = sub.to_dict()
        elif isinstance(sub, dict):
            d = dict(sub)
        else:
            raise_error(
                "KAGGLE",
                location,
                "tipo inesperado de submissao retornada pela KaggleApi (esperado dict ou objeto com to_dict)",
                impact="1",
                examples=[f"type={type(sub).__name__}", f"repr={repr(sub)[:200]}"],
            )

        items.append(
            {
                "ref": d.get("ref"),
                "date": d.get("date"),
                "status": d.get("status"),
                "publicScore": d.get("publicScore"),
                "privateScore": d.get("privateScore"),
                "description": d.get("description"),
            }
        )

    return {
        "created_utc": _utc_now(),
        "competition": str(competition),
        "page_size": int(page_size),
        "page_token": token,
        "n": int(len(items)),
        "submissions": items,
    }


__all__ = ["list_kaggle_submissions"]
