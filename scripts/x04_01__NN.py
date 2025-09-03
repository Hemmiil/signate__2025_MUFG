import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RepeatedEditedNearestNeighbours, RandomUnderSampler
import os

# 環境変数で並列を制御（ハング回避）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")


class F01__MLP:
    def __init__(self, hyperparameters: dict = None, handle_imbalance: bool = False):
        self.hyperparameters = hyperparameters if hyperparameters is not None else {}
        self.model = MLPClassifier(
            random_state=42, **self.hyperparameters,
            early_stopping=True)
        # self.sampler = RepeatedEditedNearestNeighbours(n_neighbors=5)
        self.sampler = RandomUnderSampler(random_state=42)
        self.scaler = StandardScaler()
        self.handle_imbalance = handle_imbalance

    def fit(self, X, y, X_val=None, y_val=None):
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        X_scaled = self.scaler.fit_transform(X_resampled)
        self.model.fit(X_scaled, y_resampled)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)