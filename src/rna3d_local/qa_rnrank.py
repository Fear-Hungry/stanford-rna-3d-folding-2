from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .bigdata import DEFAULT_MAX_ROWS_IN_MEMORY, DEFAULT_MEMORY_BUDGET_MB, assert_memory_budget, assert_row_budget
from .candidate_pool import CANDIDATE_POOL_DEFAULT_FEATURE_NAMES
from .errors import raise_error
from .qa_ranker import select_candidates_with_diversity
from .utils import sha256_file

QA_RNRANK_DEFAULT_FEATURE_NAMES: tuple[str, ...] = CANDIDATE_POOL_DEFAULT_FEATURE_NAMES


@dataclass(frozen=True)
class QaRnRankMetrics:
    mae: float
    rmse: float
    r2: float
    spearman: float
    pearson: float
    n_samples: int


@dataclass(frozen=True)
class QaRnRankModel:
    version: int
    model_type: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    label_col: str
    group_col: str
    hidden_dim: int
    dropout: float
    epochs: int
    best_epoch: int
    lr: float
    weight_decay: float
    rank_weight: float
    regression_weight: float
    combined_reg_weight: float
    combined_rank_weight: float
    seed: int
    weights_file: str
    train_metrics: QaRnRankMetrics
    val_metrics: QaRnRankMetrics

    def to_json_dict(self) -> dict:
        return {
            "version": int(self.version),
            "model_type": str(self.model_type),
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "stds": list(self.stds),
            "label_col": str(self.label_col),
            "group_col": str(self.group_col),
            "hidden_dim": int(self.hidden_dim),
            "dropout": float(self.dropout),
            "epochs": int(self.epochs),
            "best_epoch": int(self.best_epoch),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "rank_weight": float(self.rank_weight),
            "regression_weight": float(self.regression_weight),
            "combined_reg_weight": float(self.combined_reg_weight),
            "combined_rank_weight": float(self.combined_rank_weight),
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
    def from_json_dict(cls, payload: dict, *, location: str) -> "QaRnRankModel":
        required = (
            "feature_names",
            "means",
            "stds",
            "label_col",
            "group_col",
            "hidden_dim",
            "dropout",
            "epochs",
            "best_epoch",
            "lr",
            "weight_decay",
            "rank_weight",
            "regression_weight",
            "combined_reg_weight",
            "combined_rank_weight",
            "seed",
            "weights_file",
            "train_metrics",
            "val_metrics",
        )
        missing = [k for k in required if k not in payload]
        if missing:
            raise_error("QA_RNRANK", location, "qa_rnrank_model.json sem chave obrigatoria", impact=str(len(missing)), examples=missing[:8])
        feature_names = tuple(str(v) for v in payload["feature_names"])
        means = tuple(float(v) for v in payload["means"])
        stds = tuple(float(v) for v in payload["stds"])
        if len(feature_names) == 0:
            raise_error("QA_RNRANK", location, "feature_names vazio no qa_rnrank_model", impact="0", examples=[])
        if not (len(feature_names) == len(means) == len(stds)):
            raise_error(
                "QA_RNRANK",
                location,
                "dimensoes inconsistentes no qa_rnrank_model",
                impact=f"features={len(feature_names)} means={len(means)} stds={len(stds)}",
                examples=[],
            )
        train_metrics = payload.get("train_metrics", {})
        val_metrics = payload.get("val_metrics", {})
        return cls(
            version=int(payload.get("version", 1)),
            model_type=str(payload.get("model_type", "qa_rnrank")),
            feature_names=feature_names,
            means=means,
            stds=stds,
            label_col=str(payload["label_col"]),
            group_col=str(payload["group_col"]),
            hidden_dim=int(payload["hidden_dim"]),
            dropout=float(payload["dropout"]),
            epochs=int(payload["epochs"]),
            best_epoch=int(payload["best_epoch"]),
            lr=float(payload["lr"]),
            weight_decay=float(payload["weight_decay"]),
            rank_weight=float(payload["rank_weight"]),
            regression_weight=float(payload["regression_weight"]),
            combined_reg_weight=float(payload["combined_reg_weight"]),
            combined_rank_weight=float(payload["combined_rank_weight"]),
            seed=int(payload["seed"]),
            weights_file=str(payload["weights_file"]),
            train_metrics=QaRnRankMetrics(
                mae=float(train_metrics.get("mae", 0.0)),
                rmse=float(train_metrics.get("rmse", 0.0)),
                r2=float(train_metrics.get("r2", 0.0)),
                spearman=float(train_metrics.get("spearman", 0.0)),
                pearson=float(train_metrics.get("pearson", 0.0)),
                n_samples=int(train_metrics.get("n_samples", 0)),
            ),
            val_metrics=QaRnRankMetrics(
                mae=float(val_metrics.get("mae", 0.0)),
                rmse=float(val_metrics.get("rmse", 0.0)),
                r2=float(val_metrics.get("r2", 0.0)),
                spearman=float(val_metrics.get("spearman", 0.0)),
                pearson=float(val_metrics.get("pearson", 0.0)),
                n_samples=int(val_metrics.get("n_samples", 0)),
            ),
        )


@dataclass(frozen=True)
class QaRnRankRuntime:
    info: QaRnRankModel
    model: object
    means: np.ndarray
    stds: np.ndarray
    device: object
    weights_path: Path


def _require_torch(*, location: str):
    try:
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        raise_error("QA_RNRANK", location, "PyTorch nao disponivel para QA RNArank", impact="1", examples=[f"{type(e).__name__}:{e}"])
    return torch, F


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
            raise_error("QA_RNRANK", location, "device=cuda solicitado sem CUDA disponivel", impact="1", examples=["torch.cuda.is_available=False"])
        return torch.device("cuda")
    raise_error("QA_RNRANK", location, "device invalido", impact="1", examples=[str(requested)])
    return torch.device("cpu")


def _load_table(*, path: Path, location: str) -> pl.DataFrame:
    if not path.exists():
        raise_error("QA_RNRANK", location, "arquivo de candidatos ausente", impact="1", examples=[str(path)])
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=10_000)
    raise_error("QA_RNRANK", location, "formato de tabela nao suportado", impact="1", examples=[str(path)])
    return pl.DataFrame()


def _split_groups(*, groups: np.ndarray, val_fraction: float, seed: int, location: str) -> tuple[np.ndarray, np.ndarray]:
    if val_fraction <= 0.0 or val_fraction >= 0.9:
        raise_error("QA_RNRANK", location, "val_fraction invalido (0,0.9)", impact="1", examples=[str(val_fraction)])
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise_error("QA_RNRANK", location, "grupos insuficientes para validacao", impact=str(len(uniq)), examples=[str(v) for v in uniq[:8]])
    rng = np.random.default_rng(int(seed))
    shuffled = np.array(uniq, dtype=object)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * float(val_fraction))))
    val_set = set(shuffled[:n_val].tolist())
    val_mask = np.array([g in val_set for g in groups], dtype=bool)
    train_mask = ~val_mask
    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
        raise_error(
            "QA_RNRANK",
            location,
            "split train/val invalido",
            impact=f"train={int(train_mask.sum())} val={int(val_mask.sum())}",
            examples=[],
        )
    return train_mask, val_mask


def _compute_metrics(*, y_true: np.ndarray, y_pred: np.ndarray) -> QaRnRankMetrics:
    n = int(len(y_true))
    if n == 0:
        return QaRnRankMetrics(mae=0.0, rmse=0.0, r2=0.0, spearman=0.0, pearson=0.0, n_samples=0)
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
    return QaRnRankMetrics(mae=mae, rmse=rmse, r2=r2, spearman=sp, pearson=pear, n_samples=n)


class _RnRankNet:
    def __init__(self, *, in_dim: int, hidden_dim: int, dropout: float, location: str) -> None:
        torch, _ = _require_torch(location=location)
        if in_dim <= 0:
            raise_error("QA_RNRANK", location, "in_dim invalido", impact="1", examples=[str(in_dim)])
        if hidden_dim <= 0:
            raise_error("QA_RNRANK", location, "hidden_dim invalido", impact="1", examples=[str(hidden_dim)])
        if dropout < 0.0 or dropout >= 1.0:
            raise_error("QA_RNRANK", location, "dropout invalido [0,1)", impact="1", examples=[str(dropout)])
        self.net = torch.nn.Module()
        self.net.trunk = torch.nn.Sequential(
            torch.nn.Linear(int(in_dim), int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Dropout(float(dropout)),
        )
        self.net.reg_head = torch.nn.Linear(int(hidden_dim), 1)
        self.net.rank_head = torch.nn.Linear(int(hidden_dim), 1)

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

    def __call__(self, x):
        h = self.net.trunk(x)
        reg = self.net.reg_head(h).squeeze(-1)
        rank = self.net.rank_head(h).squeeze(-1)
        return reg, rank


def _pairwise_rank_loss(*, y_true, rank_pred, groups, location: str):
    torch, F = _require_torch(location=location)
    uniq = torch.unique(groups)
    loss_sum = torch.tensor(0.0, dtype=torch.float32, device=rank_pred.device)
    pair_count = 0
    for g in uniq:
        idx = torch.where(groups == g)[0]
        if int(idx.numel()) < 2:
            continue
        y = y_true[idx]
        p = rank_pred[idx]
        dy = y.unsqueeze(1) - y.unsqueeze(0)
        dp = p.unsqueeze(1) - p.unsqueeze(0)
        tri = torch.triu(torch.ones_like(dy, dtype=torch.bool), diagonal=1)
        mask = tri & (torch.abs(dy) > 1e-8)
        if not bool(torch.any(mask)):
            continue
        sign = torch.sign(dy[mask])
        logits = dp[mask]
        cur = F.softplus(-sign * logits)
        loss_sum = loss_sum + torch.sum(cur)
        pair_count += int(cur.numel())
    if pair_count <= 0:
        return torch.tensor(0.0, dtype=torch.float32, device=rank_pred.device)
    return loss_sum / float(pair_count)


def _combined_score(*, reg_pred: np.ndarray, rank_pred: np.ndarray, reg_weight: float, rank_weight: float) -> np.ndarray:
    return (float(reg_weight) * reg_pred) + (float(rank_weight) * rank_pred)


def train_qa_rnrank(
    *,
    candidates_path: Path,
    out_model_path: Path,
    out_weights_path: Path,
    label_col: str = "label",
    group_col: str = "target_id",
    feature_names: tuple[str, ...] = QA_RNRANK_DEFAULT_FEATURE_NAMES,
    hidden_dim: int = 128,
    dropout: float = 0.10,
    epochs: int = 160,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    rank_weight: float = 0.4,
    regression_weight: float = 0.6,
    combined_reg_weight: float = 0.6,
    combined_rank_weight: float = 0.4,
    seed: int = 123,
    device: str = "cuda",
) -> dict:
    location = "src/rna3d_local/qa_rnrank.py:train_qa_rnrank"
    if len(feature_names) == 0:
        raise_error("QA_RNRANK", location, "feature_names vazio", impact="0", examples=[])
    if hidden_dim <= 0:
        raise_error("QA_RNRANK", location, "hidden_dim deve ser > 0", impact="1", examples=[str(hidden_dim)])
    if epochs <= 0:
        raise_error("QA_RNRANK", location, "epochs deve ser > 0", impact="1", examples=[str(epochs)])
    if lr <= 0.0:
        raise_error("QA_RNRANK", location, "lr deve ser > 0", impact="1", examples=[str(lr)])
    if weight_decay < 0.0:
        raise_error("QA_RNRANK", location, "weight_decay deve ser >= 0", impact="1", examples=[str(weight_decay)])
    if rank_weight < 0.0 or regression_weight < 0.0:
        raise_error(
            "QA_RNRANK",
            location,
            "rank_weight e regression_weight devem ser >= 0",
            impact="1",
            examples=[f"rank_weight={rank_weight}", f"regression_weight={regression_weight}"],
        )
    if float(rank_weight + regression_weight) <= 0.0:
        raise_error("QA_RNRANK", location, "soma rank_weight+regression_weight deve ser > 0", impact="1", examples=[])
    if combined_reg_weight < 0.0 or combined_rank_weight < 0.0:
        raise_error(
            "QA_RNRANK",
            location,
            "combined_reg_weight e combined_rank_weight devem ser >= 0",
            impact="1",
            examples=[f"combined_reg_weight={combined_reg_weight}", f"combined_rank_weight={combined_rank_weight}"],
        )
    if float(combined_reg_weight + combined_rank_weight) <= 0.0:
        raise_error("QA_RNRANK", location, "soma combined_reg_weight+combined_rank_weight deve ser > 0", impact="1", examples=[])

    df = _load_table(path=candidates_path, location=location)
    required = set(feature_names) | {str(label_col), str(group_col)}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise_error("QA_RNRANK", location, "tabela sem coluna obrigatoria para treino", impact=str(len(missing)), examples=missing[:8])
    if df.height < 16:
        raise_error("QA_RNRANK", location, "amostras insuficientes para treino", impact=str(int(df.height)), examples=[])
    sel = df.select(
        *[pl.col(c).cast(pl.Float64).alias(c) for c in feature_names],
        pl.col(label_col).cast(pl.Float64).alias(label_col),
        pl.col(group_col).cast(pl.Utf8).alias(group_col),
    )
    null_count = int(sel.null_count().select(pl.sum_horizontal(pl.all()).alias("n")).item())
    if null_count > 0:
        raise_error("QA_RNRANK", location, "dados de treino contem nulos", impact=str(null_count), examples=[])

    x = sel.select([pl.col(c) for c in feature_names]).to_numpy().astype(np.float64, copy=False)
    y = sel.get_column(label_col).to_numpy().astype(np.float64, copy=False)
    groups = sel.get_column(group_col).to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise_error("QA_RNRANK", location, "dados de treino contem nao-finito", impact="1", examples=["nan_or_inf"])

    train_mask, val_mask = _split_groups(groups=groups, val_fraction=float(val_fraction), seed=int(seed), location=location)
    if int(train_mask.sum()) < 12 or int(val_mask.sum()) < 4:
        raise_error(
            "QA_RNRANK",
            location,
            "split train/val com poucas amostras",
            impact=f"train={int(train_mask.sum())} val={int(val_mask.sum())}",
            examples=[],
        )
    means = np.mean(x[train_mask, :], axis=0)
    stds = np.std(x[train_mask, :], axis=0)
    stds = np.where(stds <= 1e-9, 1.0, stds)
    xz = (x - means) / stds

    _uniq_groups, group_ids = np.unique(groups.astype(str), return_inverse=True)
    train_group_ids = group_ids[train_mask]
    val_group_ids = group_ids[val_mask]

    torch, F = _require_torch(location=location)
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    device_obj = _resolve_device(requested=device, location=location)
    model = _RnRankNet(in_dim=int(xz.shape[1]), hidden_dim=int(hidden_dim), dropout=float(dropout), location=location).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    x_train_t = torch.tensor(xz[train_mask, :], dtype=torch.float32, device=device_obj)
    y_train_t = torch.tensor(y[train_mask], dtype=torch.float32, device=device_obj)
    g_train_t = torch.tensor(train_group_ids, dtype=torch.long, device=device_obj)
    x_val_t = torch.tensor(xz[val_mask, :], dtype=torch.float32, device=device_obj)
    y_val_t = torch.tensor(y[val_mask], dtype=torch.float32, device=device_obj)
    g_val_t = torch.tensor(val_group_ids, dtype=torch.long, device=device_obj)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    for epoch in range(1, int(epochs) + 1):
        model.train()
        reg_train, rank_train = model(x_train_t)
        loss_reg = F.mse_loss(reg_train, y_train_t)
        loss_rank = _pairwise_rank_loss(y_true=y_train_t, rank_pred=rank_train, groups=g_train_t, location=location)
        loss = (float(regression_weight) * loss_reg) + (float(rank_weight) * loss_rank)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            reg_val, rank_val = model(x_val_t)
            loss_reg_val = F.mse_loss(reg_val, y_val_t)
            loss_rank_val = _pairwise_rank_loss(y_true=y_val_t, rank_pred=rank_val, groups=g_val_t, location=location)
            val_loss = float((float(regression_weight) * loss_reg_val) + (float(rank_weight) * loss_rank_val))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise_error("QA_RNRANK", location, "falha ao obter melhor estado do modelo", impact="1", examples=[])
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        x_all_t = torch.tensor(xz, dtype=torch.float32, device=device_obj)
        reg_all_t, rank_all_t = model(x_all_t)
    reg_all = reg_all_t.detach().cpu().numpy().astype(np.float64, copy=False)
    rank_all = rank_all_t.detach().cpu().numpy().astype(np.float64, copy=False)
    y_pred_all = _combined_score(
        reg_pred=reg_all,
        rank_pred=rank_all,
        reg_weight=float(combined_reg_weight),
        rank_weight=float(combined_rank_weight),
    )

    train_metrics = _compute_metrics(y_true=y[train_mask], y_pred=y_pred_all[train_mask])
    val_metrics = _compute_metrics(y_true=y[val_mask], y_pred=y_pred_all[val_mask])

    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    out_weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_weights_path)
    info = QaRnRankModel(
        version=1,
        model_type="qa_rnrank",
        feature_names=tuple(str(v) for v in feature_names),
        means=tuple(float(v) for v in means.tolist()),
        stds=tuple(float(v) for v in stds.tolist()),
        label_col=str(label_col),
        group_col=str(group_col),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        epochs=int(epochs),
        best_epoch=int(best_epoch),
        lr=float(lr),
        weight_decay=float(weight_decay),
        rank_weight=float(rank_weight),
        regression_weight=float(regression_weight),
        combined_reg_weight=float(combined_reg_weight),
        combined_rank_weight=float(combined_rank_weight),
        seed=int(seed),
        weights_file=str(out_weights_path.name),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )
    out_model_path.write_text(json.dumps(info.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "model_path": str(out_model_path),
        "weights_path": str(out_weights_path),
        "model_sha256": sha256_file(out_model_path),
        "weights_sha256": sha256_file(out_weights_path),
        "device_used": str(device_obj),
        "train_metrics": info.to_json_dict()["train_metrics"],
        "val_metrics": info.to_json_dict()["val_metrics"],
    }


def _load_qa_rnrank_model(
    *,
    model_path: Path,
    weights_path: Path | None,
    device: str,
    location: str,
) -> tuple[QaRnRankModel, _RnRankNet, np.ndarray, np.ndarray, object]:
    torch, _ = _require_torch(location=location)
    if not model_path.exists():
        raise_error("QA_RNRANK", location, "qa_rnrank_model.json ausente", impact="1", examples=[str(model_path)])
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise_error("QA_RNRANK", location, "falha ao ler qa_rnrank_model.json", impact="1", examples=[f"{type(e).__name__}:{e}"])
    info = QaRnRankModel.from_json_dict(payload, location=location)
    resolved_weights = weights_path if weights_path is not None else (model_path.parent / info.weights_file)
    if not resolved_weights.exists():
        raise_error("QA_RNRANK", location, "arquivo de pesos QA_RNRANK ausente", impact="1", examples=[str(resolved_weights)])

    dev = _resolve_device(requested=device, location=location)
    model = _RnRankNet(
        in_dim=len(info.feature_names),
        hidden_dim=int(info.hidden_dim),
        dropout=float(info.dropout),
        location=location,
    ).to(dev)
    state = torch.load(resolved_weights, map_location=dev)
    model.load_state_dict(state)
    means = np.asarray(info.means, dtype=np.float64)
    stds = np.asarray(info.stds, dtype=np.float64)
    stds = np.where(stds <= 1e-9, 1.0, stds)
    return info, model, means, stds, dev


def is_qa_rnrank_model_file(*, model_path: Path, location: str) -> bool:
    if not model_path.exists():
        raise_error("QA_RNRANK", location, "arquivo de modelo QA ausente", impact="1", examples=[str(model_path)])
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise_error("QA_RNRANK", location, "falha ao ler modelo QA", impact="1", examples=[f"{type(e).__name__}:{e}"])
    model_type = str(payload.get("model_type", "")).strip().lower()
    return model_type == "qa_rnrank"


def load_qa_rnrank_runtime(
    *,
    model_path: Path,
    weights_path: Path | None = None,
    device: str = "cuda",
    location: str,
) -> QaRnRankRuntime:
    info, model, means, stds, dev = _load_qa_rnrank_model(
        model_path=model_path,
        weights_path=weights_path,
        device=device,
        location=location,
    )
    resolved_weights = weights_path if weights_path is not None else (model_path.parent / info.weights_file)
    return QaRnRankRuntime(
        info=info,
        model=model,
        means=means,
        stds=stds,
        device=dev,
        weights_path=resolved_weights,
    )


def score_candidate_feature_dicts_with_qa_rnrank_runtime(
    *,
    feature_dicts: list[dict[str, float]],
    runtime: QaRnRankRuntime,
    location: str,
) -> list[float]:
    if len(feature_dicts) == 0:
        raise_error("QA_RNRANK", location, "nenhuma feature para score QA_RNRANK", impact="0", examples=[])
    rows: list[list[float]] = []
    for idx, feat in enumerate(feature_dicts):
        vals: list[float] = []
        for name in runtime.info.feature_names:
            if name not in feat:
                raise_error("QA_RNRANK", location, "feature ausente para score QA_RNRANK", impact="1", examples=[f"idx={idx}:{name}"])
            vals.append(float(feat[name]))
        rows.append(vals)
    x = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(x).all():
        raise_error("QA_RNRANK", location, "feature QA_RNRANK nao-finita", impact="1", examples=["nan_or_inf"])
    xz = (x - runtime.means) / runtime.stds
    torch, _ = _require_torch(location=location)
    runtime.model.eval()
    with torch.no_grad():
        xt = torch.tensor(xz, dtype=torch.float32, device=runtime.device)
        reg_t, rank_t = runtime.model(xt)
    reg = reg_t.detach().cpu().numpy().astype(np.float64, copy=False)
    rank = rank_t.detach().cpu().numpy().astype(np.float64, copy=False)
    pred = _combined_score(
        reg_pred=reg,
        rank_pred=rank,
        reg_weight=float(runtime.info.combined_reg_weight),
        rank_weight=float(runtime.info.combined_rank_weight),
    )
    if not np.isfinite(pred).all():
        raise_error("QA_RNRANK", location, "predicao QA_RNRANK nao-finita", impact="1", examples=["nan_or_inf"])
    return [float(v) for v in pred.tolist()]


def score_candidates_with_qa_rnrank(
    *,
    candidates_path: Path,
    model_path: Path,
    out_scores_path: Path,
    weights_path: Path | None = None,
    device: str = "cuda",
) -> dict:
    location = "src/rna3d_local/qa_rnrank.py:score_candidates_with_qa_rnrank"
    info, model, means, stds, dev = _load_qa_rnrank_model(
        model_path=model_path,
        weights_path=weights_path,
        device=device,
        location=location,
    )
    df = _load_table(path=candidates_path, location=location)
    required = set(info.feature_names) | {str(info.group_col)}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise_error("QA_RNRANK", location, "tabela sem coluna obrigatoria para score", impact=str(len(missing)), examples=missing[:8])
    if df.height == 0:
        raise_error("QA_RNRANK", location, "tabela de candidatos vazia", impact="0", examples=[str(candidates_path)])

    feat_df = df.select(*[pl.col(c).cast(pl.Float64).alias(c) for c in info.feature_names], pl.col(info.group_col).cast(pl.Utf8))
    null_count = int(feat_df.null_count().select(pl.sum_horizontal(pl.all()).alias("n")).item())
    if null_count > 0:
        raise_error("QA_RNRANK", location, "tabela de score QA_RNRANK contem nulos", impact=str(null_count), examples=[])

    x = feat_df.select([pl.col(c) for c in info.feature_names]).to_numpy().astype(np.float64, copy=False)
    if not np.isfinite(x).all():
        raise_error("QA_RNRANK", location, "tabela de score QA_RNRANK contem nao-finito", impact="1", examples=["nan_or_inf"])
    xz = (x - means) / stds
    torch, _ = _require_torch(location=location)
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(xz, dtype=torch.float32, device=dev)
        reg_t, rank_t = model(xt)
    reg = reg_t.detach().cpu().numpy().astype(np.float64, copy=False)
    rank = rank_t.detach().cpu().numpy().astype(np.float64, copy=False)
    pred = _combined_score(reg_pred=reg, rank_pred=rank, reg_weight=float(info.combined_reg_weight), rank_weight=float(info.combined_rank_weight))
    if not np.isfinite(pred).all():
        raise_error("QA_RNRANK", location, "predicao QA_RNRANK nao-finita", impact="1", examples=["nan_or_inf"])

    out_df = df.with_columns(pl.Series(name="qa_rnrank_score", values=pred.tolist(), dtype=pl.Float64))
    out_scores_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_scores_path.suffix.lower()
    if suffix == ".parquet":
        out_df.write_parquet(out_scores_path)
    elif suffix == ".csv":
        out_df.write_csv(out_scores_path)
    else:
        raise_error("QA_RNRANK", location, "formato de saida nao suportado", impact="1", examples=[str(out_scores_path)])
    payload = {
        "out_path": str(out_scores_path),
        "rows": int(out_df.height),
        "qa_rnrank_score_min": float(np.min(pred)),
        "qa_rnrank_score_max": float(np.max(pred)),
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


def select_top5_global_with_qa_rnrank(
    *,
    candidates_path: Path,
    model_path: Path,
    out_predictions_path: Path,
    n_models: int = 5,
    qa_top_pool: int = 80,
    diversity_lambda: float = 0.15,
    weights_path: Path | None = None,
    device: str = "cuda",
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    max_rows_in_memory: int = DEFAULT_MAX_ROWS_IN_MEMORY,
) -> tuple[Path, Path]:
    location = "src/rna3d_local/qa_rnrank.py:select_top5_global_with_qa_rnrank"
    assert_memory_budget(stage="QA_RNRANK_SELECT", location=location, budget_mb=memory_budget_mb)
    if n_models <= 0:
        raise_error("QA_RNRANK_SELECT", location, "n_models invalido (deve ser > 0)", impact="1", examples=[str(n_models)])
    if qa_top_pool <= 0:
        raise_error("QA_RNRANK_SELECT", location, "qa_top_pool invalido (deve ser > 0)", impact="1", examples=[str(qa_top_pool)])
    if diversity_lambda < 0.0:
        raise_error("QA_RNRANK_SELECT", location, "diversity_lambda invalido (>=0)", impact="1", examples=[str(diversity_lambda)])

    scored = score_candidates_with_qa_rnrank(
        candidates_path=candidates_path,
        model_path=model_path,
        out_scores_path=out_predictions_path.parent / f"{out_predictions_path.stem}.scored_candidates.parquet",
        weights_path=weights_path,
        device=device,
    )
    scored_path = Path(str(scored["out_path"]))
    df = _load_table(path=scored_path, location=location)
    required = {"target_id", "candidate_id", "source", "model_id", "template_uid", "coverage", "similarity", "resids", "resnames", "coords", "qa_rnrank_score"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise_error("QA_RNRANK_SELECT", location, "candidate pool sem colunas obrigatorias para selecao", impact=str(len(missing)), examples=missing[:8])
    assert_row_budget(
        stage="QA_RNRANK_SELECT",
        location=location,
        rows=int(df.height),
        max_rows_in_memory=max_rows_in_memory,
        label="scored_candidate_pool",
    )
    grouped: dict[str, pl.DataFrame] = {}
    for key, part in df.partition_by("target_id", as_dict=True).items():
        if isinstance(key, tuple):
            if len(key) != 1:
                raise_error("QA_RNRANK_SELECT", location, "chave invalida em partition_by(target_id)", impact="1", examples=[str(key)])
            tid = str(key[0])
        else:
            tid = str(key)
        grouped[tid] = part
    if len(grouped) == 0:
        raise_error("QA_RNRANK_SELECT", location, "candidate pool vazio", impact="0", examples=[])

    out_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_predictions_path.with_suffix(out_predictions_path.suffix + ".tmp")
    if tmp_out.exists():
        tmp_out.unlink()
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    selected_candidates_total = 0
    finalized = False

    try:
        for tid, gdf in grouped.items():
            if gdf.height < int(n_models):
                raise_error(
                    "QA_RNRANK_SELECT",
                    location,
                    "target com candidatos insuficientes para selecao final",
                    impact=f"target={tid} need={n_models} got={int(gdf.height)}",
                    examples=[tid],
                )
            rows = sorted(
                gdf.to_dicts(),
                key=lambda r: (
                    -float(r["qa_rnrank_score"]),
                    -float(r.get("coverage", 0.0)),
                    -float(r.get("similarity", 0.0)),
                    str(r.get("candidate_id", "")),
                ),
            )[: max(int(qa_top_pool), int(n_models))]
            candidates = []
            for r in rows:
                coords_raw = r["coords"]
                resids_raw = r["resids"]
                resnames_raw = r["resnames"]
                coords = [(float(v[0]), float(v[1]), float(v[2])) for v in coords_raw]
                resids = [int(v) for v in resids_raw]
                resnames = [str(v) for v in resnames_raw]
                if not (len(coords) == len(resids) == len(resnames)):
                    raise_error(
                        "QA_RNRANK_SELECT",
                        location,
                        "candidato com shape invalido (coords/resids/resnames)",
                        impact="1",
                        examples=[f"{tid}:{r.get('candidate_id')}"],
                    )
                candidates.append(
                    {
                        "candidate_id": str(r["candidate_id"]),
                        "source": str(r["source"]),
                        "template_uid": str(r["template_uid"]),
                        "model_id_original": int(r["model_id"]),
                        "coverage": float(r["coverage"]),
                        "similarity": float(r["similarity"]),
                        "qa_score": float(r["qa_rnrank_score"]),
                        "coords": coords,
                        "resids": resids,
                        "resnames": resnames,
                    }
                )
            selected = select_candidates_with_diversity(
                candidates=candidates,
                n_models=int(n_models),
                diversity_lambda=float(diversity_lambda),
                location=location,
            )
            selected_candidates_total += int(len(selected))
            out_rows: list[dict] = []
            out_model_id = 0
            for cand in selected:
                out_model_id += 1
                for resid, resname, xyz in zip(cand["resids"], cand["resnames"], cand["coords"], strict=True):
                    out_rows.append(
                        {
                            "branch": str(cand["source"]),
                            "target_id": str(tid),
                            "ID": f"{tid}_{int(resid)}",
                            "resid": int(resid),
                            "resname": str(resname),
                            "model_id": int(out_model_id),
                            "x": float(xyz[0]),
                            "y": float(xyz[1]),
                            "z": float(xyz[2]),
                            "template_uid": str(cand["template_uid"]),
                            "similarity": float(cand["similarity"]),
                            "coverage": float(cand["coverage"]),
                            "qa_score": float(cand["qa_score"]),
                            "candidate_id": str(cand["candidate_id"]),
                        }
                    )
            table = pa.Table.from_pylist(out_rows)
            if writer is None:
                writer = pq.ParquetWriter(str(tmp_out), table.schema, compression="zstd")
            writer.write_table(table)
            rows_written += len(out_rows)
            assert_memory_budget(stage="QA_RNRANK_SELECT", location=location, budget_mb=memory_budget_mb)
        if writer is not None:
            writer.close()
            writer = None
        if rows_written <= 0:
            raise_error("QA_RNRANK_SELECT", location, "nenhuma linha gerada na selecao global", impact="0", examples=[])
        if out_predictions_path.exists():
            out_predictions_path.unlink()
        tmp_out.replace(out_predictions_path)
        finalized = True
    finally:
        if writer is not None:
            writer.close()
        if (not finalized) and tmp_out.exists():
            tmp_out.unlink()

    manifest = {
        "paths": {
            "candidates": str(candidates_path),
            "model": str(model_path),
            "selected_predictions": str(out_predictions_path),
            "scored_candidates": str(scored_path),
        },
        "params": {
            "n_models": int(n_models),
            "qa_top_pool": int(qa_top_pool),
            "diversity_lambda": float(diversity_lambda),
            "device": str(device),
        },
        "stats": {
            "rows_written": int(rows_written),
            "selected_candidates": int(selected_candidates_total),
            "n_targets": int(df.select("target_id").n_unique()),
        },
        "sha256": {
            "selected_predictions.parquet": sha256_file(out_predictions_path),
            "scored_candidates.parquet": sha256_file(scored_path),
        },
    }
    manifest_path = out_predictions_path.parent / "qa_rnrank_select_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_predictions_path, manifest_path


__all__ = [
    "QA_RNRANK_DEFAULT_FEATURE_NAMES",
    "QaRnRankMetrics",
    "QaRnRankModel",
    "QaRnRankRuntime",
    "is_qa_rnrank_model_file",
    "load_qa_rnrank_runtime",
    "score_candidate_feature_dicts_with_qa_rnrank_runtime",
    "score_candidates_with_qa_rnrank",
    "select_top5_global_with_qa_rnrank",
    "train_qa_rnrank",
]
