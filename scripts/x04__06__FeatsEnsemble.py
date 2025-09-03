# -*- coding: utf-8 -*-
"""
Logistic Regression Subspace Ensemble (Classification) — F01__LR_SubspaceEnsemble

- k_features_min, k_features_max の範囲から各モデルごとに特徴量数 k をランダムサンプリング
- 評価指標による重みづけ時に、ハイパーパラメータ `metrics_power` で「評価指標の p 乗」を使用（既定: 1.0）
- RealMLP モジュールと同じ I/O (fit, predict, predict_proba) に対応
"""

import os
from typing import Optional, Dict, Union, List, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from imblearn.under_sampling import RandomUnderSampler

# 並列抑制（環境依存でハング回避）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _is_binary(y: np.ndarray) -> bool:
    return np.unique(y).size == 2


def _numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    return X.select_dtypes(include=["float", "float64", "float32", "int", "int64", "int32"])


def _compute_metric(
    y_true: np.ndarray,
    proba: np.ndarray,
    metric: Union[str, Callable] = "roc_auc",
    average: str = "macro",
    threshold: float = 0.5,
) -> float:
    if callable(metric):
        return float(metric(y_true, proba))

    if metric == "roc_auc":
        if proba.shape[1] == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        else:
            return float(roc_auc_score(y_true, proba, multi_class="ovr", average=average))

    if metric == "f1":
        if proba.shape[1] == 2:
            y_pred = (proba[:, 1] >= threshold).astype(int)
            return float(f1_score(y_true, y_pred, average="binary"))
        else:
            y_pred = np.argmax(proba, axis=1)
            return float(f1_score(y_true, y_pred, average=average))

    if metric == "accuracy":
        if proba.shape[1] == 2:
            y_pred = (proba[:, 1] >= threshold).astype(int)
        else:
            y_pred = np.argmax(proba, axis=1)
        return float(accuracy_score(y_true, y_pred))

    raise ValueError(f"Unsupported metric: {metric}")


class F01__LR_SubspaceEnsemble:
    """
    __init__(hyperparameters: dict = None, handle_imbalance: bool = False)

    hyperparameters 例:
        - m_models: int = 20            # アンサンブルするモデル本数
        - k_features_min: int = 5       # 選択する特徴量の最小値
        - k_features_max: int = 15      # 選択する特徴量の最大値
        - metric: str = "roc_auc"       # "roc_auc", "f1", "accuracy" など
        - average: str = "macro"        # multi-class 用
        - random_state: int = 42
        - test_size: float = 0.2        # 内部 val データの割合
        - threshold: float = 0.5        # 予測時の確率閾値 (binary のみ有効)
        - metrics_power: float = 1.0    # 重みづけに使う評価指標に p 乗を適用
        - base_params: dict             # LogisticRegression のパラメータ
    """

    def __init__(self, hyperparameters: dict = None, handle_imbalance: bool = False):
        hp = {} if hyperparameters is None else dict(hyperparameters)

        self.m_models: int = int(hp.get("m_models", 20))
        self.k_features_min: Optional[int] = hp.get("k_features_min", None)
        self.k_features_max: Optional[int] = hp.get("k_features_max", None)
        self.metric: Union[str, Callable] = hp.get("metric", "roc_auc")
        self.average: str = hp.get("average", "macro")
        self.random_state: int = int(hp.get("random_state", 42))
        self.test_size: float = float(hp.get("test_size", 0.2))
        self.threshold: float = float(hp.get("threshold", 0.5))
        self.metrics_power: float = float(hp.get("metrics_power", 1.0))
        self.base_params: Dict = hp.get("base_params", {})

        self.handle_imbalance = handle_imbalance

        # components
        self.scaler = StandardScaler()
        self.sampler = RandomUnderSampler(random_state=self.random_state)

        # fitted artifacts
        self.models: List[LogisticRegression] = []
        self.subspaces: List[np.ndarray] = []
        self.weights: Optional[np.ndarray] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.is_binary_: Optional[bool] = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_columns_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        Xn = _numeric_df(X).copy()
        self.feature_columns_ = list(Xn.columns)

        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(np.asarray(y))
        self.classes_ = self.label_encoder.classes_
        self.is_binary_ = _is_binary(y_enc)

        if self.handle_imbalance:
            X_res, y_res = self.sampler.fit_resample(Xn, y_enc)
        else:
            X_res, y_res = Xn.values, y_enc

        X_res = self.scaler.fit_transform(X_res)

        strat = y_res if np.unique(y_res).size > 1 else None
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_res, y_res, test_size=self.test_size, random_state=self.random_state, stratify=strat
        )

        n_features = X_tr.shape[1]
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**32 - 1, size=self.m_models, endpoint=False)

        # k_features の範囲を決定
        if self.k_features_min is None or self.k_features_max is None:
            k_min = max(1, int(np.sqrt(n_features)))
            k_max = n_features
        else:
            k_min = max(1, min(int(self.k_features_min), n_features))
            k_max = max(k_min, min(int(self.k_features_max), n_features))

        val_scores = []
        self.models.clear()
        self.subspaces.clear()

        for i in range(self.m_models):
            k = rng.integers(k_min, k_max + 1)  # k_min 〜 k_max の整数をランダムに選ぶ
            cols = rng.choice(n_features, size=k, replace=False)

            model = LogisticRegression(
                random_state=int(seeds[i]),
                max_iter=1000,
                **self.base_params
            )
            model.fit(X_tr[:, cols], y_tr)
            proba_val = model.predict_proba(X_val[:, cols])
            score = _compute_metric(
                y_val, proba_val,
                metric=self.metric,
                average=self.average,
                threshold=self.threshold
            )

            self.models.append(model)
            self.subspaces.append(cols)
            val_scores.append(score)

        # --- 重み計算: 評価指標の p 乗を適用 ---
        val_scores = np.asarray(val_scores, dtype=float)
        val_scores = np.clip(val_scores, 0.0, None)          # 安全のため負値クリップ
        powered = np.power(val_scores, self.metrics_power)    # p 乗（p=1 で従来どおり）
        s = powered.sum()
        if s <= 1e-12:
            self.weights = np.ones_like(powered) / len(powered)
        else:
            self.weights = powered / s

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        Xn = _numeric_df(X)
        Xn_aligned = self._align_columns(Xn, self.feature_columns_)
        X_scaled = self.scaler.transform(Xn_aligned.values)

        n_classes = 2 if self.is_binary_ else len(self.classes_)
        agg = np.zeros((X_scaled.shape[0], n_classes), dtype=float)

        for w, model, cols in zip(self.weights, self.models, self.subspaces):
            p = model.predict_proba(X_scaled[:, cols])
            agg += w * p

        row_sum = agg.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        agg /= row_sum
        return agg

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        if self.is_binary_:
            y_pred = (proba[:, 1] >= self.threshold).astype(int)
        else:
            y_pred = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(y_pred)

    @staticmethod
    def _align_columns(X: pd.DataFrame, cols_ref: List[str]) -> pd.DataFrame:
        Xc = X.copy()
        for c in cols_ref:
            if c not in Xc.columns:
                Xc[c] = 0.0
        return Xc[cols_ref]
