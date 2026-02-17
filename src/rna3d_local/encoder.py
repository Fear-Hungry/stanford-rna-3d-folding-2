from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

from .errors import raise_error


def _tokenize_sequence(seq: str, *, stage: str, location: str) -> list[int]:
    mapping = {"A": 0, "U": 1, "G": 2, "C": 3, "N": 5}
    cleaned = str(seq or "").strip().upper().replace("T", "U")
    tokens: list[int] = []
    bad: set[str] = set()
    for ch in cleaned:
        if ch in {"|", " ", "\t", "\n", "\r"}:
            continue
        if ch in mapping:
            tokens.append(mapping[ch])
        else:
            bad.add(ch)
    if not tokens:
        raise_error(stage, location, "sequencia vazia para encoder", impact="1", examples=["empty_sequence"])
    if bad:
        raise_error(stage, location, "sequencia contem simbolos invalidos para encoder", impact=str(len(bad)), examples=sorted(bad)[:8])
    return tokens


def _maybe_parse_max_len_from_diffusion_config(model_path: Path) -> int | None:
    cfg_path = model_path.parent / "diffusion_config.yaml"
    if not cfg_path.exists():
        return None
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except Exception:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.lower().startswith("max_len:"):
            raw = stripped.split(":", 1)[1].strip()
            try:
                return int(raw)
            except Exception:
                return None
    return None


def _aggressive_offload_cuda(*, model: object | None) -> None:
    if model is not None:
        to_fn = getattr(model, "to", None)
        if callable(to_fn):
            try:
                to_fn("cpu")
            except Exception:
                pass
        del model
    gc.collect()
    try:
        import torch  # type: ignore
    except Exception:
        return
    if bool(torch.cuda.is_available()):
        try:
            torch.cuda.empty_cache()
        except Exception:
            return


def _encode_with_torchscript(
    model: object,
    sequences: list[str],
    *,
    embedding_dim: int,
    stage: str,
    location: str,
) -> np.ndarray:
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "torch indisponivel para encoder ribonanzanet2", impact="1", examples=[f"{type(exc).__name__}:{exc}"])

    rows: list[np.ndarray] = []
    with torch.no_grad():
        for seq in sequences:
            tokens = _tokenize_sequence(str(seq), stage=stage, location=location)
            x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            try:
                out = model(x)  # type: ignore[misc]
            except Exception as exc:  # noqa: BLE001
                raise_error(stage, location, "falha ao executar modelo ribonanzanet2 (torchscript)", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
            if isinstance(out, (tuple, list)):
                if not out:
                    raise_error(stage, location, "modelo ribonanzanet2 retornou output vazio", impact="1", examples=["empty_output"])
                out = out[0]
            if not hasattr(out, "shape"):
                raise_error(stage, location, "modelo ribonanzanet2 retornou tipo invalido", impact="1", examples=[str(type(out).__name__)])
            emb = out
            if emb.ndim == 3:
                emb = emb.mean(dim=1)
            elif emb.ndim == 2:
                if emb.shape[0] != 1:
                    emb = emb.mean(dim=0, keepdim=True)
            elif emb.ndim == 1:
                emb = emb.unsqueeze(0)
            else:
                raise_error(stage, location, "dimensao de embedding invalida do modelo", impact="1", examples=[str(tuple(int(x) for x in emb.shape))])
            if int(emb.shape[1]) != int(embedding_dim):
                raise_error(
                    stage,
                    location,
                    "embedding_dim divergente do modelo",
                    impact="1",
                    examples=[f"expected={embedding_dim}", f"actual={int(emb.shape[1])}"],
                )
            vec = emb[0].detach().cpu().to(dtype=torch.float32).numpy()
            norm = float(np.linalg.norm(vec))
            if not np.isfinite(norm) or norm <= 0:
                raise_error(stage, location, "embedding invalido (norm <= 0)", impact="1", examples=[f"norm={norm}"])
            vec = vec.astype(np.float32) / float(norm)
            rows.append(vec)
    return np.asarray(rows, dtype=np.float32)


def _load_ribonanzanet2_from_state_dict(
    *,
    model_path: Path,
    embedding_dim: int,
    stage: str,
    location: str,
) -> tuple[object, int]:
    try:
        import importlib.util
        import types

        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "dependencias indisponiveis para carregar ribonanzanet2 state_dict", impact="1", examples=[f"{type(exc).__name__}:{exc}"])

    try:
        state = torch.load(str(model_path), map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "falha ao carregar state_dict ribonanzanet2", impact="1", examples=[str(model_path), f"{type(exc).__name__}:{exc}"])
    if not isinstance(state, dict) or not state:
        raise_error(stage, location, "arquivo ribonanzanet2 nao parece ser um state_dict valido", impact="1", examples=[str(type(state).__name__)])
    if not all(isinstance(k, str) for k in state.keys()):
        raise_error(stage, location, "state_dict ribonanzanet2 possui chaves nao-string", impact="1", examples=[str(model_path)])

    encoder_key = None
    for key in state.keys():
        if key.endswith("encoder.weight"):
            encoder_key = key
            break
    if encoder_key is not None:
        weight = state[encoder_key]
        if hasattr(weight, "shape") and len(weight.shape) == 2:
            actual_dim = int(weight.shape[1])
            if actual_dim != int(embedding_dim):
                raise_error(
                    stage,
                    location,
                    "embedding_dim divergente do state_dict ribonanzanet2",
                    impact="1",
                    examples=[f"expected={embedding_dim}", f"actual={actual_dim}", encoder_key],
                )

    assets_root: Path | None = None
    for parent in model_path.resolve().parents:
        cand = parent / "models" / "rnapro" / "ribonanzanet2_checkpoint" / "Network.py"
        if cand.exists():
            assets_root = parent
            break
    if assets_root is None:
        raise_error(
            stage,
            location,
            "nao foi possivel localizar Network.py para instanciar ribonanzanet2",
            impact="1",
            examples=[str(model_path), "esperado: <assets>/models/rnapro/ribonanzanet2_checkpoint/Network.py"],
        )
    ribo_dir = assets_root / "models" / "rnapro" / "ribonanzanet2_checkpoint"
    network_py = ribo_dir / "Network.py"
    if not network_py.exists():
        raise_error(stage, location, "Network.py ausente para ribonanzanet2", impact="1", examples=[str(network_py)])

    try:
        import matplotlib.pyplot  # type: ignore  # noqa: F401
    except Exception:
        matplotlib_mod = types.ModuleType("matplotlib")
        pyplot_mod = types.ModuleType("matplotlib.pyplot")

        def _noop(*_args: object, **_kwargs: object) -> None:
            return None

        for name in ("imshow", "show", "savefig", "figure", "plot"):
            setattr(pyplot_mod, name, _noop)
        setattr(matplotlib_mod, "pyplot", pyplot_mod)
        sys.modules.setdefault("matplotlib", matplotlib_mod)
        sys.modules.setdefault("matplotlib.pyplot", pyplot_mod)

    sys_path_added = False
    try:
        sys.path.insert(0, str(ribo_dir))
        sys_path_added = True
        spec = importlib.util.spec_from_file_location("rna3d_local_ribonanzanet2", network_py)
        if spec is None or spec.loader is None:
            raise_error(stage, location, "falha ao criar spec para Network.py", impact="1", examples=[str(network_py)])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "falha ao importar Network.py do ribonanzanet2", impact="1", examples=[f"{type(exc).__name__}:{exc}", str(network_py)])
    finally:
        if sys_path_added:
            try:
                if sys.path and sys.path[0] == str(ribo_dir):
                    sys.path.pop(0)
            except Exception:
                pass

    if not hasattr(module, "RibonanzaNet"):
        raise_error(stage, location, "Network.py nao exporta RibonanzaNet", impact="1", examples=[str(network_py)])

    class _Cfg:
        ntoken = 6
        ninp = int(embedding_dim)
        nhead = 12
        nlayers = 48
        nclass = 10
        dropout = 0.1
        k = 5
        use_triangular_attention = False
        pairwise_dimension = 128
        dim_msa = 32

    try:
        model = module.RibonanzaNet(_Cfg())  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "falha ao instanciar RibonanzaNet (state_dict)", impact="1", examples=[f"{type(exc).__name__}:{exc}"])

    cleaned = {}
    for key, value in state.items():
        k = key[len("module.") :] if key.startswith("module.") else key
        cleaned[k] = value
    try:
        model_keys = set(getattr(model, "state_dict")().keys())
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "falha ao inspecionar state_dict do modelo ribonanzanet2", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    filtered = {k: v for k, v in cleaned.items() if k in model_keys}
    if not filtered:
        raise_error(stage, location, "state_dict ribonanzanet2 nao contem chaves do modelo esperado", impact="1", examples=[str(model_path)])
    try:
        incompatible = model.load_state_dict(filtered, strict=False)
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "falha ao aplicar state_dict no RibonanzaNet", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing:
        examples: list[str] = []
        if missing:
            examples.append(f"missing={missing[0]}")
        if unexpected:
            examples.append(f"unexpected={unexpected[0]}")
        raise_error(
            stage,
            location,
            "state_dict ribonanzanet2 nao carregou completamente (missing keys)",
            impact=str(len(missing)),
            examples=examples[:8],
        )
    if unexpected:
        raise_error(
            stage,
            location,
            "state_dict ribonanzanet2 carregou com unexpected keys apos filtro (bug)",
            impact=str(len(unexpected)),
            examples=[f"unexpected={unexpected[0]}"],
        )
    model.eval()
    max_len = _maybe_parse_max_len_from_diffusion_config(model_path) or 384
    return model, int(max_len)


def _encode_with_state_dict_model(
    model: object,
    sequences: list[str],
    *,
    embedding_dim: int,
    max_len: int,
    stage: str,
    location: str,
) -> np.ndarray:
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "torch indisponivel para encoder ribonanzanet2", impact="1", examples=[f"{type(exc).__name__}:{exc}"])

    if int(max_len) <= 0:
        raise_error(stage, location, "max_len invalido para encoder ribonanzanet2", impact="1", examples=[str(max_len)])

    device = torch.device("cuda" if bool(torch.cuda.is_available()) else "cpu")
    try:
        to_fn = getattr(model, "to", None)
        if callable(to_fn):
            model = to_fn(device)
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "falha ao mover ribonanzanet2 para device", impact="1", examples=[str(device), f"{type(exc).__name__}:{exc}"])

    rows: list[np.ndarray] = []
    with torch.no_grad():
        for seq in sequences:
            tokens = _tokenize_sequence(str(seq), stage=stage, location=location)
            if len(tokens) <= int(max_len):
                windows = [tokens]
            else:
                step = int(max_len)
                windows = [tokens[i : i + int(max_len)] for i in range(0, len(tokens), step)]
            vec_sum = torch.zeros((int(embedding_dim),), dtype=torch.float32, device=device)
            weight_sum = 0.0
            for window in windows:
                x = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
                src_mask = torch.ones((1, int(x.shape[1])), dtype=torch.long, device=device)
                try:
                    seq_rep = getattr(model, "encoder")(x).reshape(1, x.shape[1], -1)
                    pair_rep = getattr(model, "outer_product_mean")(seq_rep)
                    pair_rep = pair_rep + getattr(model, "pos_encoder")(seq_rep)
                    for layer in getattr(model, "transformer_encoder"):
                        seq_rep, pair_rep = layer([seq_rep, pair_rep, src_mask, False])
                except Exception as exc:  # noqa: BLE001
                    raise_error(stage, location, "falha ao executar ribonanzanet2 (state_dict)", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
                pooled = seq_rep.mean(dim=1).squeeze(0).to(dtype=torch.float32)
                if pooled.ndim != 1 or int(pooled.shape[0]) != int(embedding_dim):
                    raise_error(
                        stage,
                        location,
                        "embedding retornado pelo ribonanzanet2 (state_dict) tem shape invalido",
                        impact="1",
                        examples=[str(tuple(int(x) for x in pooled.shape)), f"expected_dim={embedding_dim}"],
                    )
                weight = float(len(window))
                vec_sum = vec_sum + (pooled * float(weight))
                weight_sum += float(weight)
                del x, src_mask, seq_rep, pair_rep, pooled
            if not np.isfinite(weight_sum) or weight_sum <= 0:
                raise_error(stage, location, "peso total invalido ao pool de chunks", impact="1", examples=[f"weight_sum={weight_sum}"])
            vec = (vec_sum / float(weight_sum)).detach().cpu().numpy()
            norm = float(np.linalg.norm(vec))
            if not np.isfinite(norm) or norm <= 0:
                raise_error(stage, location, "embedding invalido (norm <= 0)", impact="1", examples=[f"norm={norm}"])
            vec = vec.astype(np.float32) / float(norm)
            rows.append(vec)
    _aggressive_offload_cuda(model=model)
    return np.asarray(rows, dtype=np.float32)


def encode_sequences(
    sequences: list[str],
    *,
    encoder: str,
    embedding_dim: int,
    model_path: Path | None,
    stage: str,
    location: str,
) -> np.ndarray:
    if embedding_dim <= 0:
        raise_error(stage, location, "embedding_dim deve ser > 0", impact="1", examples=[str(embedding_dim)])
    mode = str(encoder).strip().lower()
    if mode not in {"ribonanzanet2"}:
        raise_error(stage, location, "encoder invalido", impact="1", examples=[encoder])
    if model_path is None:
        raise_error(stage, location, "model_path obrigatorio para ribonanzanet2", impact="1", examples=["model_path=None"])
    if not model_path.exists():
        raise_error(stage, location, "model_path nao encontrado para ribonanzanet2", impact="1", examples=[str(model_path)])

    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "torch indisponivel para encoder ribonanzanet2", impact="1", examples=[f"{type(exc).__name__}:{exc}"])

    try:
        model_ts = torch.jit.load(str(model_path), map_location="cpu")
        model_ts.eval()
        return _encode_with_torchscript(model_ts, sequences, embedding_dim=embedding_dim, stage=stage, location=location)
    except Exception as exc_torchscript:  # noqa: BLE001
        try:
            model_sd, max_len = _load_ribonanzanet2_from_state_dict(
                model_path=model_path,
                embedding_dim=embedding_dim,
                stage=stage,
                location=location,
            )
        except Exception as exc_state_dict:  # noqa: BLE001
            raise_error(
                stage,
                location,
                "falha ao carregar ribonanzanet2 (torchscript e state_dict)",
                impact="1",
                examples=[f"torchscript={type(exc_torchscript).__name__}:{exc_torchscript}", f"state_dict={type(exc_state_dict).__name__}:{exc_state_dict}"],
            )
        return _encode_with_state_dict_model(
            model_sd,
            sequences,
            embedding_dim=embedding_dim,
            max_len=max_len,
            stage=stage,
            location=location,
        )
