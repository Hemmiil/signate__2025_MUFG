import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RepeatedEditedNearestNeighbours, RandomUnderSampler
import os
from umap import UMAP

# 環境変数で並列を制御（ハング回避）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

class F01__KNC:
    def __init__(self, hyperparameters: dict = None, handle_imbalance: bool = False):
        self.is_conv_log: bool = bool(hyperparameters.pop("is_conv_log", False))
        self.is_conv_emb: bool = bool(hyperparameters.pop("is_conv_emb", False))

        self.hyperparameters = hyperparameters if hyperparameters is not None else {}
        self.model = KNeighborsClassifier(
            **self.hyperparameters,
            )
        # self.sampler = RepeatedEditedNearestNeighbours(sampling_strategy="auto", n_neighbors=5)
        self.sampler = RandomUnderSampler(random_state = 42)
        self.scaler = StandardScaler()
        self.handle_imbalance = handle_imbalance

    def data_convs(
            self, X, is_log_conv = True, is_emb = True, is_train = False
    ):
        # 対数変換
        if is_log_conv:
            cols_pos = X.columns[X.min() > 0].tolist()
            non_target_log = ["f01__goal", "f02__product__goal_X_f04__datetime"]
            for col in non_target_log:
                cols_pos.remove(col)

            X.loc[:, cols_pos] = X.loc[:, cols_pos].astype(float)
            X[cols_pos] = X[cols_pos].apply(np.log)

        # テキストベクトルの圧縮
        if is_emb:
            if is_train:
                self.EmbeddingModel = UMAP(n_components=3)
                cols_sentence = [col for col in X.columns if "SentenceVec" in col]
                X_sentence_emb = pd.DataFrame(
                    self.EmbeddingModel.fit_transform(X[cols_sentence]),
                    columns=[f"f06__SentenceVec_Emb_{i+1}" for i in range(3)],
                    index=X.index,
                )
                X = pd.concat(
                    [X.drop(cols_sentence, axis="columns"), X_sentence_emb], axis="columns"
                )
            else:

                cols_sentence = [col for col in X.columns if "SentenceVec" in col]
                X_sentence_emb = pd.DataFrame(
                    self.EmbeddingModel.fit_transform(X[cols_sentence]),
                    columns=[f"f06__SentenceVec_Emb_{i+1}" for i in range(3)],
                    index=X.index,
                )
                X = pd.concat(
                    [X.drop(cols_sentence, axis="columns"), X_sentence_emb], axis="columns"
                )

        return X
    def fit(self, X, y, X_val=None, y_val=None):
        X_nu = X.select_dtypes("float")
        X_resampled, y_resampled = self.sampler.fit_resample(X_nu, y)
        X_conved = self.data_convs(X_resampled, is_emb=self.is_conv_emb, is_log_conv=self.is_conv_log, is_train=True)
        X_scaled = self.scaler.fit_transform(X_conved)
        self.model.fit(X_scaled, y_resampled)

    def predict(self, X):
        X_nu = X.select_dtypes("float")
        X_conved = self.data_convs(X_nu, is_emb=self.is_conv_emb, is_log_conv=self.is_conv_log, is_train=True)
        X_scaled = self.scaler.fit_transform(X_conved)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_nu = X.select_dtypes("float")
        X_conved = self.data_convs(X_nu, is_emb=self.is_conv_emb, is_log_conv=self.is_conv_log, is_train=True)
        X_scaled = self.scaler.fit_transform(X_conved)
        return self.model.predict_proba(X_scaled)