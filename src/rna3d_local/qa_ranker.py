from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from .errors import raise_error
from .utils import sha256_file

QA_FEATURE_NAMES: tuple[str, ...] = (
    "coverage",
    "similarity",
    "match_ratio",
    "mismatch_ratio",
    "chem_compatible_ratio",
    "path_length",
    "step_mean",
    "step_std",
    "radius_gyr",
    "gap_open_score",
    "gap_extend_score",
)

QA_DIVERSITY_RMSD_SIGMA_ANGSTROM = 10.0


@dataclass(frozen=True)
class QaMetrics:
    mae: float
    rmse: float
    r2: float
    spearman: float
    pearson: float
    n_samples: int


@dataclass(frozen=True)
class QaModel:
    version: int
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    l2_lambda: float
    label_col: str
    group_col: str
    train_metrics: QaMetrics
    val_metrics: QaMetrics

    def to_json_dict(self) -> dict:
        return {
            "version": int(self.version),
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "stds": list(self.stds),
            "weights": list(self.weights),
            "bias": float(self.bias),
            "l2_lambda": float(self.l2_lambda),
            "label_col": str(self.label_col),
            "group_col": str(self.group_col),
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
    def from_json_dict(cls, payload: dict, *, location: str) -> "QaModel":
        required = ("feature_names", "means", "stds", "weights", "bias", "l2_lambda", "label_col", "group_col")
        missing = [k for k in required if k not in payload]
        if missing:
            raise_error("QA", location, "qa_model.json sem chave obrigatoria", impact=str(len(missing)), examples=missing[:8])
        feat = tuple(str(v) for v in payload["feature_names"])
        means = tuple(float(v) for v in payload["means"])
        stds = tuple(float(v) for v in payload["stds"])
        weights = tuple(float(v) for v in payload["weights"])
        if len(feat) == 0:
            raise_error("QA", location, "qa_model sem features", impact="0", examples=[])
        if not (len(feat) == len(means) == len(stds) == len(weights)):
            raise_error(
                "QA",
                location,
                "qa_model com dimensoes inconsistentes",
                impact=f"features={len(feat)} means={len(means)} stds={len(stds)} weights={len(weights)}",
                examples=[],
            )
        tm = payload.get("train_metrics", {})
        vm = payload.get("val_metrics", {})
        return cls(
            version=int(payload.get("version", 1)),
            feature_names=feat,
            means=means,
            stds=stds,
            weights=weights,
            bias=float(payload["bias"]),
            l2_lambda=float(payload["l2_lambda"]),
            label_col=str(payload["label_col"]),
            group_col=str(payload["group_col"]),
            train_metrics=QaMetrics(
                mae=float(tm.get("mae", 0.0)),
                rmse=float(tm.get("rmse", 0.0)),
                r2=float(tm.get("r2", 0.0)),
                spearman=float(tm.get("spearman", 0.0)),
                pearson=float(tm.get("pearson", 0.0)),
                n_samples=int(tm.get("n_samples", 0)),
            ),
            val_metrics=QaMetrics(
                mae=float(vm.get("mae", 0.0)),
                rmse=float(vm.get("rmse", 0.0)),
                r2=float(vm.get("r2", 0.0)),
                spearman=float(vm.get("spearman", 0.0)),
                pearson=float(vm.get("pearson", 0.0)),
                n_samples=int(vm.get("n_samples", 0)),
            ),
        )


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


def _coords_to_np(coords: list[tuple[float, float, float]], *, location: str) -> np.ndarray:
    if len(coords) == 0:
        raise_error("QA", location, "coords candidato vazio", impact="0", examples=[])
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise_error("QA", location, "coords candidato com shape invalido", impact="1", examples=[str(arr.shape)])
    if not np.isfinite(arr).all():
        raise_error("QA", location, "coords candidato contem nao-finito", impact="1", examples=["nan_or_inf"])
    return arr


def candidate_geometry_features(*, coords: list[tuple[float, float, float]], location: str) -> dict[str, float]:
    arr = _coords_to_np(coords, location=location)
    if arr.shape[0] == 1:
        return {
            "path_length": 0.0,
            "step_mean": 0.0,
            "step_std": 0.0,
            "radius_gyr": 0.0,
        }
    diffs = arr[1:, :] - arr[:-1, :]
    steps = np.linalg.norm(diffs, axis=1)
    center = arr.mean(axis=0, keepdims=True)
    rg = float(np.sqrt(np.mean(np.sum((arr - center) ** 2, axis=1))))
    return {
        "path_length": float(np.sum(steps)),
        "step_mean": float(np.mean(steps)),
        "step_std": float(np.std(steps)),
        "radius_gyr": float(rg),
    }


def build_candidate_feature_dict(
    *,
    candidate: dict,
    target_length: int,
    location: str,
) -> dict[str, float]:
    if target_length <= 0:
        raise_error("QA", location, "target_length invalido", impact="1", examples=[str(target_length)])
    coords = candidate.get("coords")
    if not isinstance(coords, list):
        raise_error("QA", location, "candidate sem coords para QA", impact="1", examples=[str(candidate.get("uid", "?"))])

    geom = candidate_geometry_features(coords=coords, location=location)
    mapped_count = int(candidate.get("mapped_count", 0))
    match_count = int(candidate.get("match_count", 0))
    mismatch_count = int(candidate.get("mismatch_count", 0))
    chem_compatible_count = int(candidate.get("chem_compatible_count", 0))
    cov = _safe_float(candidate.get("coverage"), default=0.0)
    sim = _safe_float(candidate.get("similarity"), default=0.0)
    feat = {
        "coverage": float(cov),
        "similarity": float(sim),
        "match_ratio": float(match_count) / float(target_length),
        "mismatch_ratio": float(mismatch_count) / float(target_length),
        "chem_compatible_ratio": float(chem_compatible_count) / float(target_length),
        "path_length": float(geom["path_length"]),
        "step_mean": float(geom["step_mean"]),
        "step_std": float(geom["step_std"]),
        "radius_gyr": float(geom["radius_gyr"]),
        "gap_open_score": _safe_float(candidate.get("gap_open_score"), default=0.0),
        "gap_extend_score": _safe_float(candidate.get("gap_extend_score"), default=0.0),
    }
    # If mapping counts were not supplied, derive from coverage.
    if mapped_count <= 0 and cov > 0:
        feat["match_ratio"] = max(feat["match_ratio"], float(cov))
    return feat


def _vectorize_features(*, feature_dict: dict[str, float], feature_names: Iterable[str], location: str) -> np.ndarray:
    values: list[float] = []
    for name in feature_names:
        if name not in feature_dict:
            raise_error("QA", location, "feature ausente para QA", impact="1", examples=[str(name)])
        values.append(_safe_float(feature_dict[name], default=0.0))
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise_error("QA", location, "feature QA nao-finita", impact="1", examples=["nan_or_inf"])
    return arr


def load_qa_model(*, model_path: Path, location: str) -> QaModel:
    if not model_path.exists():
        raise_error("QA", location, "qa_model.json ausente", impact="1", examples=[str(model_path)])
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise_error("QA", location, "falha ao ler qa_model.json", impact="1", examples=[f"{type(e).__name__}:{e}"])
    return QaModel.from_json_dict(payload, location=location)


def score_candidate_with_model(
    *,
    candidate_features: dict[str, float],
    model: QaModel,
    location: str,
) -> float:
    x = _vectorize_features(feature_dict=candidate_features, feature_names=model.feature_names, location=location)
    means = np.asarray(model.means, dtype=np.float64)
    stds = np.asarray(model.stds, dtype=np.float64)
    w = np.asarray(model.weights, dtype=np.float64)
    xz = (x - means) / stds
    score = float(np.dot(xz, w) + float(model.bias))
    if not np.isfinite(score):
        raise_error("QA", location, "qa_score nao-finito", impact="1", examples=["nan_or_inf"])
    return score


def _kabsch_rmsd(*, a: np.ndarray, b: np.ndarray, location: str) -> float:
    if a.shape != b.shape:
        raise_error("QA", location, "coords com shape divergente para diversidade", impact="1", examples=[f"a={a.shape}", f"b={b.shape}"])
    a0 = a - np.mean(a, axis=0, keepdims=True)
    b0 = b - np.mean(b, axis=0, keepdims=True)
    h = b0.T @ a0
    if not np.isfinite(h).all():
        raise_error("QA", location, "matriz de covariancia nao-finita para diversidade", impact="1", examples=["nan_or_inf"])
    try:
        u, _s, vt = np.linalg.svd(h)
    except Exception as e:  # noqa: BLE001
        raise_error("QA", location, "falha no SVD para diversidade", impact="1", examples=[f"{type(e).__name__}:{e}"])
    r = u @ vt
    det = float(np.linalg.det(r))
    if det < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    b_aligned = b0 @ r
    diff = a0 - b_aligned
    rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    if not np.isfinite(rmsd):
        raise_error("QA", location, "rmsd nao-finito para diversidade", impact="1", examples=["nan_or_inf"])
    if rmsd < 0.0:
        raise_error("QA", location, "rmsd negativo para diversidade", impact="1", examples=[str(rmsd)])
    return rmsd


def _pair_similarity(*, coords_a: list[tuple[float, float, float]], coords_b: list[tuple[float, float, float]], location: str) -> float:
    a = _coords_to_np(coords_a, location=location)
    b = _coords_to_np(coords_b, location=location)
    rmsd = _kabsch_rmsd(a=a, b=b, location=location)
    sigma = float(QA_DIVERSITY_RMSD_SIGMA_ANGSTROM)
    if sigma <= 0.0:
        raise_error("QA", location, "sigma de diversidade invalido", impact="1", examples=[str(sigma)])
    sim = float(np.exp(-(rmsd / sigma)))
    if sim < 0.0:
        sim = 0.0
    if sim > 1.0:
        sim = 1.0
    return sim


def select_candidates_with_diversity(
    *,
    candidates: list[dict],
    n_models: int,
    diversity_lambda: float,
    location: str,
) -> list[dict]:
    if n_models <= 0:
        raise_error("QA", location, "n_models invalido", impact="1", examples=[str(n_models)])
    if diversity_lambda < 0.0:
        raise_error("QA", location, "diversity_lambda invalido (>=0)", impact="1", examples=[str(diversity_lambda)])
    if len(candidates) < n_models:
        raise_error(
            "QA",
            location,
            "candidatos insuficientes para selecao final",
            impact=f"need={n_models} got={len(candidates)}",
            examples=[],
        )
    selected: list[dict] = []
    remaining = list(candidates)
    while len(selected) < n_models:
        best_idx = -1
        best_score = -1e30
        for idx, cand in enumerate(remaining):
            base = _safe_float(cand.get("qa_score"), default=0.0)
            if len(selected) == 0:
                effective = base
            else:
                max_sim = max(_pair_similarity(coords_a=cand["coords"], coords_b=s["coords"], location=location) for s in selected)
                effective = float(base - float(diversity_lambda) * float(max_sim))
            if effective > best_score:
                best_score = effective
                best_idx = idx
        if best_idx < 0:
            raise_error("QA", location, "falha interna na selecao por diversidade", impact="1", examples=[])
        picked = remaining.pop(best_idx)
        picked["qa_effective_score"] = float(best_score)
        selected.append(picked)
    return selected


def _compute_metrics(*, y_true: np.ndarray, y_pred: np.ndarray) -> QaMetrics:
    n = int(len(y_true))
    if n == 0:
        return QaMetrics(mae=0.0, rmse=0.0, r2=0.0, spearman=0.0, pearson=0.0, n_samples=0)
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
    return QaMetrics(mae=mae, rmse=rmse, r2=r2, spearman=sp, pearson=pear, n_samples=n)


def _split_groups(
    *,
    groups: np.ndarray,
    val_fraction: float,
    seed: int,
    location: str,
) -> tuple[np.ndarray, np.ndarray]:
    if val_fraction <= 0.0 or val_fraction >= 0.9:
        raise_error("QA", location, "val_fraction invalido (0,0.9)", impact="1", examples=[str(val_fraction)])
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise_error("QA", location, "grupos insuficientes para validacao", impact=str(len(uniq)), examples=[str(v) for v in uniq[:8]])
    rng = np.random.default_rng(int(seed))
    shuffled = np.array(uniq, dtype=object)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * float(val_fraction))))
    val_set = set(shuffled[:n_val].tolist())
    val_mask = np.array([g in val_set for g in groups], dtype=bool)
    train_mask = ~val_mask
    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
        raise_error(
            "QA",
            location,
            "split train/val invalido para QA",
            impact=f"train={int(train_mask.sum())} val={int(val_mask.sum())}",
            examples=[],
        )
    return train_mask, val_mask


def train_qa_ranker(
    *,
    candidates_path: Path,
    out_model_path: Path,
    label_col: str = "label",
    group_col: str = "target_id",
    feature_names: tuple[str, ...] = QA_FEATURE_NAMES,
    l2_lambda: float = 1.0,
    val_fraction: float = 0.2,
    seed: int = 123,
) -> dict:
    location = "src/rna3d_local/qa_ranker.py:train_qa_ranker"
    if not candidates_path.exists():
        raise_error("QA", location, "arquivo de candidatos ausente", impact="1", examples=[str(candidates_path)])
    if l2_lambda < 0.0:
        raise_error("QA", location, "l2_lambda invalido (>=0)", impact="1", examples=[str(l2_lambda)])
    if len(feature_names) == 0:
        raise_error("QA", location, "feature_names vazio", impact="0", examples=[])
    suffix = candidates_path.suffix.lower()
    if suffix == ".parquet":
        df = pl.read_parquet(candidates_path)
    elif suffix == ".csv":
        df = pl.read_csv(candidates_path, infer_schema_length=10_000)
    else:
        raise_error("QA", location, "formato nao suportado para treino QA", impact="1", examples=[str(candidates_path)])

    required = set(feature_names) | {str(label_col), str(group_col)}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise_error("QA", location, "tabela de treino QA sem coluna obrigatoria", impact=str(len(missing)), examples=missing[:8])
    if df.height < 10:
        raise_error("QA", location, "amostras insuficientes para treino QA", impact=str(int(df.height)), examples=[])

    sel = df.select(
        *[pl.col(c).cast(pl.Float64).alias(c) for c in feature_names],
        pl.col(label_col).cast(pl.Float64).alias(label_col),
        pl.col(group_col).cast(pl.Utf8).alias(group_col),
    )
    if int(sel.null_count().select(pl.sum_horizontal(pl.all()).alias("n")).item()) > 0:
        raise_error("QA", location, "dados de treino QA contem nulos", impact="1", examples=[])

    X = sel.select([pl.col(c) for c in feature_names]).to_numpy().astype(np.float64, copy=False)
    y = sel.get_column(label_col).to_numpy().astype(np.float64, copy=False)
    groups = sel.get_column(group_col).to_numpy()
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise_error("QA", location, "dados de treino QA contem nao-finito", impact="1", examples=["nan_or_inf"])

    train_mask, val_mask = _split_groups(groups=groups, val_fraction=float(val_fraction), seed=int(seed), location=location)
    X_train = X[train_mask, :]
    y_train = y[train_mask]
    X_val = X[val_mask, :]
    y_val = y[val_mask]

    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)
    stds = np.where(stds <= 1e-9, 1.0, stds)
    Xz = (X_train - means) / stds
    Xz_val = (X_val - means) / stds

    # Ridge closed-form (bias not regularized).
    ones = np.ones((Xz.shape[0], 1), dtype=np.float64)
    Xb = np.concatenate([ones, Xz], axis=1)
    reg = np.eye(Xb.shape[1], dtype=np.float64) * float(l2_lambda)
    reg[0, 0] = 0.0
    try:
        beta = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y_train)
    except np.linalg.LinAlgError as e:
        raise_error("QA", location, "falha numerica no ajuste ridge", impact="1", examples=[f"{type(e).__name__}:{e}"])
    bias = float(beta[0])
    weights = beta[1:]

    y_hat_train = bias + (Xz @ weights)
    y_hat_val = bias + (Xz_val @ weights)
    train_metrics = _compute_metrics(y_true=y_train, y_pred=y_hat_train)
    val_metrics = _compute_metrics(y_true=y_val, y_pred=y_hat_val)

    model = QaModel(
        version=1,
        feature_names=tuple(str(c) for c in feature_names),
        means=tuple(float(v) for v in means.tolist()),
        stds=tuple(float(v) for v in stds.tolist()),
        weights=tuple(float(v) for v in weights.tolist()),
        bias=float(bias),
        l2_lambda=float(l2_lambda),
        label_col=str(label_col),
        group_col=str(group_col),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    out_model_path.write_text(json.dumps(model.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "model_path": str(out_model_path),
        "sha256": sha256_file(out_model_path),
        "train_metrics": model.to_json_dict()["train_metrics"],
        "val_metrics": model.to_json_dict()["val_metrics"],
    }
