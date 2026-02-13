from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from .errors import raise_error
from .utils import sha256_file

QA_GNN_DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    "coverage",
    "similarity",
    "path_length",
    "step_mean",
    "step_std",
    "radius_gyr",
    "gap_open_score",
    "gap_extend_score",
)


@dataclass(frozen=True)
class QaGnnMetrics:
    mae: float
    rmse: float
    r2: float
    spearman: float
    pearson: float
    n_samples: int


@dataclass(frozen=True)
class QaGnnModel:
    version: int
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    label_col: str
    group_col: str
    hidden_dim: int
    num_layers: int
    dropout: float
    knn_k: int
    epochs: int
    best_epoch: int
    lr: float
    weight_decay: float
    seed: int
    weights_file: str
    train_metrics: QaGnnMetrics
    val_metrics: QaGnnMetrics

    def to_json_dict(self) -> dict:
        return {
            "version": int(self.version),
            "model_type": "qa_gnn",
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "stds": list(self.stds),
            "label_col": str(self.label_col),
            "group_col": str(self.group_col),
            "hidden_dim": int(self.hidden_dim),
            "num_layers": int(self.num_layers),
            "dropout": float(self.dropout),
            "knn_k": int(self.knn_k),
            "epochs": int(self.epochs),
            "best_epoch": int(self.best_epoch),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "seed": int(self.seed),
            "weights_file": str(self.weights_file),
            "train_metrics": {
                "mae": float(self.train_metrics.mae),
                "rmse": float(self.train_metrics.rmse),
                "r2": float(self.train_metrics.r2),
                "spearman": float(self.train_metrics.spearman),
                "pearson": float(self.train_metrics.pearson),
                "n_samples": int(self.train_metrics.n_samples),
            },
            "val_metrics": {
                "mae": float(self.val_metrics.mae),
                "rmse": float(self.val_metrics.rmse),
                "r2": float(self.val_metrics.r2),
                "spearman": float(self.val_metrics.spearman),
                "pearson": float(self.val_metrics.pearson),
                "n_samples": int(self.val_metrics.n_samples),
            },
        }

    @classmethod
    def from_json_dict(cls, payload: dict, *, location: str) -> "QaGnnModel":
        required = (
            "feature_names",
            "means",
            "stds",
            "label_col",
            "group_col",
            "hidden_dim",
            "num_layers",
            "dropout",
            "knn_k",
            "epochs",
            "best_epoch",
            "lr",
            "weight_decay",
            "seed",
            "weights_file",
        )
        missing = [k for k in required if k not in payload]
        if missing:
            raise_error("QA_GNN", location, "qa_gnn_model.json sem chave obrigatoria", impact=str(len(missing)), examples=missing[:8])
        feat = tuple(str(v) for v in payload["feature_names"])
        means = tuple(float(v) for v in payload["means"])
        stds = tuple(float(v) for v in payload["stds"])
        if len(feat) == 0:
            raise_error("QA_GNN", location, "qa_gnn_model sem features", impact="0", examples=[])
        if not (len(feat) == len(means) == len(stds)):
            raise_error(
                "QA_GNN",
                location,
                "qa_gnn_model com dimensoes inconsistentes",
                impact=f"features={len(feat)} means={len(means)} stds={len(stds)}",
                examples=[],
            )
        tm = payload.get("train_metrics", {})
        vm = payload.get("val_metrics", {})
        return cls(
            version=int(payload.get("version", 1)),
            feature_names=feat,
            means=means,
            stds=stds,
            label_col=str(payload["label_col"]),
            group_col=str(payload["group_col"]),
            hidden_dim=int(payload["hidden_dim"]),
            num_layers=int(payload["num_layers"]),
            dropout=float(payload["dropout"]),
            knn_k=int(payload["knn_k"]),
            epochs=int(payload["epochs"]),
            best_epoch=int(payload["best_epoch"]),
            lr=float(payload["lr"]),
            weight_decay=float(payload["weight_decay"]),
            seed=int(payload["seed"]),
            weights_file=str(payload["weights_file"]),
            train_metrics=QaGnnMetrics(
                mae=float(tm.get("mae", 0.0)),
                rmse=float(tm.get("rmse", 0.0)),
                r2=float(tm.get("r2", 0.0)),
                spearman=float(tm.get("spearman", 0.0)),
                pearson=float(tm.get("pearson", 0.0)),
                n_samples=int(tm.get("n_samples", 0)),
            ),
            val_metrics=QaGnnMetrics(
                mae=float(vm.get("mae", 0.0)),
                rmse=float(vm.get("rmse", 0.0)),
                r2=float(vm.get("r2", 0.0)),
                spearman=float(vm.get("spearman", 0.0)),
                pearson=float(vm.get("pearson", 0.0)),
                n_samples=int(vm.get("n_samples", 0)),
            ),
        )


@dataclass(frozen=True)
class QaGnnRuntime:
    info: QaGnnModel
    model: object
    means: np.ndarray
    stds: np.ndarray
    device: object
    weights_path: Path


def _safe_float(value: object, *, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return float(x)


def _load_table(*, path: Path, location: str) -> pl.DataFrame:
    if not path.exists():
        raise_error("QA_GNN", location, "arquivo de candidatos ausente", impact="1", examples=[str(path)])
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=10_000)
    raise_error("QA_GNN", location, "formato de tabela nao suportado", impact="1", examples=[str(path)])
    return pl.DataFrame()


def _require_torch(*, location: str):
    try:
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        raise_error("QA_GNN", location, "PyTorch nao disponivel para GNN", impact="1", examples=[f"{type(e).__name__}:{e}"])
    return torch, F


def _split_groups(*, groups: np.ndarray, val_fraction: float, seed: int, location: str) -> tuple[np.ndarray, np.ndarray]:
    if val_fraction <= 0.0 or val_fraction >= 0.9:
        raise_error("QA_GNN", location, "val_fraction invalido (0,0.9)", impact="1", examples=[str(val_fraction)])
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise_error("QA_GNN", location, "grupos insuficientes para validacao", impact=str(len(uniq)), examples=[str(v) for v in uniq[:8]])
    rng = np.random.default_rng(int(seed))
    shuffled = np.array(uniq, dtype=object)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * float(val_fraction))))
    val_set = set(shuffled[:n_val].tolist())
    val_mask = np.array([g in val_set for g in groups], dtype=bool)
    train_mask = ~val_mask
    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
        raise_error(
            "QA_GNN",
            location,
            "split train/val invalido",
            impact=f"train={int(train_mask.sum())} val={int(val_mask.sum())}",
            examples=[],
        )
    return train_mask, val_mask


def _compute_metrics(*, y_true: np.ndarray, y_pred: np.ndarray) -> QaGnnMetrics:
    n = int(len(y_true))
    if n == 0:
        return QaGnnMetrics(mae=0.0, rmse=0.0, r2=0.0, spearman=0.0, pearson=0.0, n_samples=0)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = float(0.0 if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot))
    if n >= 2:
        pear = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 0 and np.std(y_pred) > 0 else 0.0
        rank_t = np.argsort(np.argsort(y_true))
        rank_p = np.argsort(np.argsort(y_pred))
        sp = float(np.corrcoef(rank_t, rank_p)[0, 1]) if np.std(rank_t) > 0 and np.std(rank_p) > 0 else 0.0
    else:
        pear = 0.0
        sp = 0.0
    if not np.isfinite(pear):
        pear = 0.0
    if not np.isfinite(sp):
        sp = 0.0
    return QaGnnMetrics(mae=mae, rmse=rmse, r2=r2, spearman=sp, pearson=pear, n_samples=n)


def _resolve_device(*, requested: str, location: str):
    torch, _ = _require_torch(location=location)
    req = str(requested).strip().lower()
    if req in ("", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise_error("QA_GNN", location, "device=cuda solicitado sem CUDA disponivel", impact="1", examples=["torch.cuda.is_available=False"])
        return torch.device("cuda")
    raise_error("QA_GNN", location, "device invalido", impact="1", examples=[str(requested)])
    return torch.device("cpu")


class _GraphRegressor:
    def __init__(self, *, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, location: str) -> None:
        torch, _ = _require_torch(location=location)
        if in_dim <= 0:
            raise_error("QA_GNN", location, "in_dim invalido", impact="1", examples=[str(in_dim)])
        if hidden_dim <= 0:
            raise_error("QA_GNN", location, "hidden_dim invalido", impact="1", examples=[str(hidden_dim)])
        if num_layers <= 0:
            raise_error("QA_GNN", location, "num_layers invalido", impact="1", examples=[str(num_layers)])
        if dropout < 0.0 or dropout >= 1.0:
            raise_error("QA_GNN", location, "dropout invalido [0,1)", impact="1", examples=[str(dropout)])

        self._torch = torch
        self.layers = torch.nn.ModuleList()
        last_dim = int(in_dim)
        for _ in range(int(num_layers)):
            block = torch.nn.ModuleDict(
                {
                    "self_lin": torch.nn.Linear(last_dim, int(hidden_dim)),
                    "neigh_lin": torch.nn.Linear(last_dim, int(hidden_dim)),
                }
            )
            self.layers.append(block)
            last_dim = int(hidden_dim)
        self.out = torch.nn.Linear(last_dim, 1)
        self.dropout = float(dropout)
        self.net = torch.nn.Module()
        self.net.layers = self.layers
        self.net.out = self.out

    def to(self, device):
        self.net.to(device)
        return self

    def parameters(self):
        return self.net.parameters()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.net.load_state_dict(state)

    def train(self) -> None:
        self.net.train()

    def eval(self) -> None:
        self.net.eval()

    def __call__(self, x, a):
        _, F = _require_torch(location="src/rna3d_local/qa_gnn_ranker.py:_GraphRegressor.__call__")
        h = x
        for block in self.layers:
            m = a @ h
            h = block["self_lin"](h) + block["neigh_lin"](m)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.net.training)
        out = self.out(h).squeeze(-1)
        return out


def _build_knn_adjacency(*, x, knn_k: int, location: str):
    torch, _ = _require_torch(location=location)
    n = int(x.shape[0])
    if n <= 0:
        raise_error("QA_GNN", location, "grafo vazio", impact="0", examples=[])
    if n == 1:
        return torch.eye(1, device=x.device, dtype=x.dtype)
    if knn_k <= 0:
        raise_error("QA_GNN", location, "knn_k deve ser > 0", impact="1", examples=[str(knn_k)])
    k_eff = min(int(knn_k), n - 1)
    dist = torch.cdist(x, x, p=2.0)
    idx = torch.argsort(dist, dim=1)[:, 1 : (k_eff + 1)]
    adj = torch.zeros((n, n), device=x.device, dtype=x.dtype)
    src = torch.arange(n, device=x.device).unsqueeze(1).expand(n, k_eff)
    adj[src.reshape(-1), idx.reshape(-1)] = 1.0
    adj = torch.maximum(adj, adj.T)
    adj.fill_diagonal_(1.0)
    row_sum = adj.sum(dim=1, keepdim=True)
    row_sum = torch.where(row_sum <= 1e-12, torch.ones_like(row_sum), row_sum)
    return adj / row_sum


def _prepare_training_arrays(
    *,
    candidates_path: Path,
    feature_names: tuple[str, ...],
    label_col: str,
    group_col: str,
    location: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = _load_table(path=candidates_path, location=location)
    required = set(feature_names) | {str(label_col), str(group_col)}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise_error("QA_GNN", location, "tabela sem coluna obrigatoria", impact=str(len(missing)), examples=missing[:8])
    if df.height < 10:
        raise_error("QA_GNN", location, "amostras insuficientes para treino QA_GNN", impact=str(int(df.height)), examples=[])

    sel = df.select(
        *[pl.col(c).cast(pl.Float64).alias(c) for c in feature_names],
        pl.col(label_col).cast(pl.Float64).alias(label_col),
        pl.col(group_col).cast(pl.Utf8).alias(group_col),
    )
    null_count = int(sel.null_count().select(pl.sum_horizontal(pl.all()).alias("n")).item())
    if null_count > 0:
        raise_error("QA_GNN", location, "dados de treino QA_GNN contem nulos", impact=str(null_count), examples=[])

    x = sel.select([pl.col(c) for c in feature_names]).to_numpy().astype(np.float64, copy=False)
    y = sel.get_column(label_col).to_numpy().astype(np.float64, copy=False)
    groups = sel.get_column(group_col).to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise_error("QA_GNN", location, "dados de treino QA_GNN contem nao-finito", impact="1", examples=["nan_or_inf"])
    return x, y, groups


def _group_indices(*, groups: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, list[int]] = {}
    for idx, g in enumerate(groups.tolist()):
        key = str(g)
        out.setdefault(key, []).append(int(idx))
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def _predict_groupwise(
    *,
    model: _GraphRegressor,
    xz: np.ndarray,
    group_to_idx: dict[str, np.ndarray],
    group_keys: list[str],
    knn_k: int,
    device,
    location: str,
) -> np.ndarray:
    torch, _ = _require_torch(location=location)
    pred = np.zeros((xz.shape[0],), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for g in group_keys:
            idx = group_to_idx[g]
            xg = torch.tensor(xz[idx, :], dtype=torch.float32, device=device)
            adj = _build_knn_adjacency(x=xg, knn_k=int(knn_k), location=location)
            yg = model(xg, adj).detach().cpu().numpy().astype(np.float64, copy=False)
            pred[idx] = yg
    return pred


def train_qa_gnn_ranker(
    *,
    candidates_path: Path,
    out_model_path: Path,
    out_weights_path: Path,
    label_col: str = "label",
    group_col: str = "target_id",
    feature_names: tuple[str, ...] = QA_GNN_DEFAULT_FEATURE_NAMES,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    knn_k: int = 8,
    epochs: int = 120,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    seed: int = 123,
    device: str = "cuda",
) -> dict:
    location = "src/rna3d_local/qa_gnn_ranker.py:train_qa_gnn_ranker"
    if len(feature_names) == 0:
        raise_error("QA_GNN", location, "feature_names vazio", impact="0", examples=[])
    if epochs <= 0:
        raise_error("QA_GNN", location, "epochs deve ser > 0", impact="1", examples=[str(epochs)])
    if lr <= 0.0:
        raise_error("QA_GNN", location, "lr deve ser > 0", impact="1", examples=[str(lr)])
    if weight_decay < 0.0:
        raise_error("QA_GNN", location, "weight_decay deve ser >= 0", impact="1", examples=[str(weight_decay)])

    torch, F = _require_torch(location=location)
    x, y, groups = _prepare_training_arrays(
        candidates_path=candidates_path,
        feature_names=feature_names,
        label_col=label_col,
        group_col=group_col,
        location=location,
    )
    train_mask, val_mask = _split_groups(groups=groups, val_fraction=float(val_fraction), seed=int(seed), location=location)
    if int(train_mask.sum()) < 8 or int(val_mask.sum()) < 4:
        raise_error(
            "QA_GNN",
            location,
            "split train/val com poucas amostras",
            impact=f"train={int(train_mask.sum())} val={int(val_mask.sum())}",
            examples=[],
        )

    means = np.mean(x[train_mask, :], axis=0)
    stds = np.std(x[train_mask, :], axis=0)
    stds = np.where(stds <= 1e-9, 1.0, stds)
    xz = (x - means) / stds

    group_to_idx = _group_indices(groups=groups)
    train_groups = sorted({str(v) for v in groups[train_mask].tolist()})
    val_groups = sorted({str(v) for v in groups[val_mask].tolist()})
    if len(train_groups) == 0 or len(val_groups) == 0:
        raise_error("QA_GNN", location, "grupos train/val vazios", impact="1", examples=[])

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    device_obj = _resolve_device(requested=device, location=location)
    model = _GraphRegressor(
        in_dim=int(xz.shape[1]),
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        dropout=float(dropout),
        location=location,
    ).to(device_obj)
    optim = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    rng = np.random.default_rng(int(seed))
    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_order = list(train_groups)
        rng.shuffle(train_order)
        for g in train_order:
            idx = group_to_idx[g]
            xg = torch.tensor(xz[idx, :], dtype=torch.float32, device=device_obj)
            yg = torch.tensor(y[idx], dtype=torch.float32, device=device_obj)
            adj = _build_knn_adjacency(x=xg, knn_k=int(knn_k), location=location)
            pred = model(xg, adj)
            loss = F.mse_loss(pred, yg)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for g in val_groups:
                idx = group_to_idx[g]
                xg = torch.tensor(xz[idx, :], dtype=torch.float32, device=device_obj)
                yg = torch.tensor(y[idx], dtype=torch.float32, device=device_obj)
                adj = _build_knn_adjacency(x=xg, knn_k=int(knn_k), location=location)
                pred = model(xg, adj)
                cur = F.mse_loss(pred, yg).item()
                val_loss_sum += float(cur) * float(len(idx))
                val_n += int(len(idx))
        if val_n <= 0:
            raise_error("QA_GNN", location, "validacao sem amostras", impact="0", examples=[])
        val_loss = float(val_loss_sum / float(val_n))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise_error("QA_GNN", location, "falha ao obter melhor estado do modelo", impact="1", examples=[])
    model.load_state_dict(best_state)

    pred_all = _predict_groupwise(
        model=model,
        xz=xz,
        group_to_idx=group_to_idx,
        group_keys=sorted(group_to_idx.keys()),
        knn_k=int(knn_k),
        device=device_obj,
        location=location,
    )
    train_metrics = _compute_metrics(y_true=y[train_mask], y_pred=pred_all[train_mask])
    val_metrics = _compute_metrics(y_true=y[val_mask], y_pred=pred_all[val_mask])

    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    out_weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_weights_path)
    model_info = QaGnnModel(
        version=1,
        feature_names=tuple(str(v) for v in feature_names),
        means=tuple(float(v) for v in means.tolist()),
        stds=tuple(float(v) for v in stds.tolist()),
        label_col=str(label_col),
        group_col=str(group_col),
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        dropout=float(dropout),
        knn_k=int(knn_k),
        epochs=int(epochs),
        best_epoch=int(best_epoch),
        lr=float(lr),
        weight_decay=float(weight_decay),
        seed=int(seed),
        weights_file=str(out_weights_path.name),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )
    out_model_path.write_text(json.dumps(model_info.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "model_path": str(out_model_path),
        "weights_path": str(out_weights_path),
        "model_sha256": sha256_file(out_model_path),
        "weights_sha256": sha256_file(out_weights_path),
        "device_used": str(device_obj),
        "train_metrics": model_info.to_json_dict()["train_metrics"],
        "val_metrics": model_info.to_json_dict()["val_metrics"],
    }


def _load_qa_gnn_model(
    *,
    model_path: Path,
    weights_path: Path | None,
    device: str,
    location: str,
) -> tuple[QaGnnModel, _GraphRegressor, np.ndarray, np.ndarray, object]:
    torch, _ = _require_torch(location=location)
    if not model_path.exists():
        raise_error("QA_GNN", location, "qa_gnn_model.json ausente", impact="1", examples=[str(model_path)])
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise_error("QA_GNN", location, "falha ao ler qa_gnn_model.json", impact="1", examples=[f"{type(e).__name__}:{e}"])
    info = QaGnnModel.from_json_dict(payload, location=location)

    resolved_weights = weights_path
    if resolved_weights is None:
        resolved_weights = model_path.parent / info.weights_file
    if not resolved_weights.exists():
        raise_error("QA_GNN", location, "arquivo de pesos QA_GNN ausente", impact="1", examples=[str(resolved_weights)])

    dev = _resolve_device(requested=device, location=location)
    model = _GraphRegressor(
        in_dim=len(info.feature_names),
        hidden_dim=int(info.hidden_dim),
        num_layers=int(info.num_layers),
        dropout=float(info.dropout),
        location=location,
    ).to(dev)
    state = torch.load(resolved_weights, map_location=dev)
    model.load_state_dict(state)
    means = np.asarray(info.means, dtype=np.float64)
    stds = np.asarray(info.stds, dtype=np.float64)
    stds = np.where(stds <= 1e-9, 1.0, stds)
    return info, model, means, stds, dev


def is_qa_gnn_model_file(*, model_path: Path, location: str) -> bool:
    if not model_path.exists():
        raise_error("QA_GNN", location, "arquivo de modelo QA ausente", impact="1", examples=[str(model_path)])
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise_error("QA_GNN", location, "falha ao ler modelo QA", impact="1", examples=[f"{type(e).__name__}:{e}"])
    model_type = str(payload.get("model_type", "")).strip().lower()
    return model_type == "qa_gnn"


def load_qa_gnn_runtime(
    *,
    model_path: Path,
    weights_path: Path | None = None,
    device: str = "cuda",
    location: str,
) -> QaGnnRuntime:
    info, model, means, stds, dev = _load_qa_gnn_model(
        model_path=model_path,
        weights_path=weights_path,
        device=device,
        location=location,
    )
    resolved_weights = weights_path if weights_path is not None else (model_path.parent / info.weights_file)
    return QaGnnRuntime(
        info=info,
        model=model,
        means=means,
        stds=stds,
        device=dev,
        weights_path=resolved_weights,
    )


def score_candidate_feature_dicts_with_qa_gnn_runtime(
    *,
    feature_dicts: list[dict[str, float]],
    runtime: QaGnnRuntime,
    location: str,
) -> list[float]:
    if len(feature_dicts) == 0:
        raise_error("QA_GNN", location, "nenhuma feature para score QA_GNN", impact="0", examples=[])
    rows: list[list[float]] = []
    for idx, feat in enumerate(feature_dicts):
        vals: list[float] = []
        for name in runtime.info.feature_names:
            if name not in feat:
                raise_error(
                    "QA_GNN",
                    location,
                    "feature ausente para score QA_GNN",
                    impact="1",
                    examples=[f"idx={idx}:{name}"],
                )
            vals.append(float(feat[name]))
        rows.append(vals)
    x = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(x).all():
        raise_error("QA_GNN", location, "feature QA_GNN nao-finita", impact="1", examples=["nan_or_inf"])
    xz = (x - runtime.means) / runtime.stds
    idx = np.arange(xz.shape[0], dtype=np.int64)
    pred = _predict_groupwise(
        model=runtime.model,  # type: ignore[arg-type]
        xz=xz,
        group_to_idx={"__target__": idx},
        group_keys=["__target__"],
        knn_k=int(runtime.info.knn_k),
        device=runtime.device,
        location=location,
    )
    if not np.isfinite(pred).all():
        raise_error("QA_GNN", location, "predicao QA_GNN nao-finita", impact="1", examples=["nan_or_inf"])
    return [float(v) for v in pred.tolist()]


def score_candidates_with_qa_gnn(
    *,
    candidates_path: Path,
    model_path: Path,
    out_scores_path: Path,
    weights_path: Path | None = None,
    device: str = "cuda",
) -> dict:
    location = "src/rna3d_local/qa_gnn_ranker.py:score_candidates_with_qa_gnn"
    info, model, means, stds, dev = _load_qa_gnn_model(
        model_path=model_path,
        weights_path=weights_path,
        device=device,
        location=location,
    )
    df = _load_table(path=candidates_path, location=location)
    required = set(info.feature_names) | {str(info.group_col)}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise_error("QA_GNN", location, "tabela sem coluna obrigatoria para score", impact=str(len(missing)), examples=missing[:8])
    if df.height == 0:
        raise_error("QA_GNN", location, "tabela de candidatos vazia", impact="0", examples=[str(candidates_path)])

    feat_df = df.select(*[pl.col(c).cast(pl.Float64).alias(c) for c in info.feature_names], pl.col(info.group_col).cast(pl.Utf8))
    null_count = int(feat_df.null_count().select(pl.sum_horizontal(pl.all()).alias("n")).item())
    if null_count > 0:
        raise_error("QA_GNN", location, "tabela de score QA_GNN contem nulos", impact=str(null_count), examples=[])

    x = feat_df.select([pl.col(c) for c in info.feature_names]).to_numpy().astype(np.float64, copy=False)
    groups = feat_df.get_column(info.group_col).to_numpy()
    if not np.isfinite(x).all():
        raise_error("QA_GNN", location, "tabela de score QA_GNN contem nao-finito", impact="1", examples=["nan_or_inf"])
    xz = (x - means) / stds
    group_to_idx = _group_indices(groups=groups)
    pred = _predict_groupwise(
        model=model,
        xz=xz,
        group_to_idx=group_to_idx,
        group_keys=sorted(group_to_idx.keys()),
        knn_k=int(info.knn_k),
        device=dev,
        location=location,
    )
    if not np.isfinite(pred).all():
        raise_error("QA_GNN", location, "predicao QA_GNN nao-finita", impact="1", examples=["nan_or_inf"])

    out_df = df.with_columns(pl.Series(name="gnn_score", values=pred.tolist(), dtype=pl.Float64))
    out_scores_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_scores_path.suffix.lower()
    if suffix == ".parquet":
        out_df.write_parquet(out_scores_path)
    elif suffix == ".csv":
        out_df.write_csv(out_scores_path)
    else:
        raise_error("QA_GNN", location, "formato de saida nao suportado para score", impact="1", examples=[str(out_scores_path)])

    payload = {
        "out_path": str(out_scores_path),
        "rows": int(out_df.height),
        "gnn_score_min": _safe_float(np.min(pred), default=0.0),
        "gnn_score_max": _safe_float(np.max(pred), default=0.0),
    }
    if str(info.label_col) in out_df.columns:
        y_true = out_df.get_column(info.label_col).cast(pl.Float64).to_numpy().astype(np.float64, copy=False)
        metrics = _compute_metrics(y_true=y_true, y_pred=pred)
        payload["metrics"] = {
            "mae": float(metrics.mae),
            "rmse": float(metrics.rmse),
            "r2": float(metrics.r2),
            "spearman": float(metrics.spearman),
            "pearson": float(metrics.pearson),
            "n_samples": int(metrics.n_samples),
        }
    return payload


__all__ = [
    "QA_GNN_DEFAULT_FEATURE_NAMES",
    "QaGnnModel",
    "QaGnnMetrics",
    "QaGnnRuntime",
    "is_qa_gnn_model_file",
    "load_qa_gnn_runtime",
    "score_candidate_feature_dicts_with_qa_gnn_runtime",
    "score_candidates_with_qa_gnn",
    "train_qa_gnn_ranker",
]
