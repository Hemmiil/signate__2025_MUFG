from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

NUM_CLUSTERS = 6
class F01__ClusteringClassifier:
    def __init__(self, hyperparameters: dict = None, handle_imbalance: bool = False, NUM_CLUSTERS=6):
        self.NUM_CLUSTERS = NUM_CLUSTERS
        self.hyperparameters = hyperparameters if hyperparameters is not None else {}

        self.model_clustering = KMeans(n_clusters=self.NUM_CLUSTERS, random_state=42)
        self.scaler = StandardScaler()

        model_name = hyperparameters["model_name"]
        hyperparameter = hyperparameters["hyperparameter"]
        if model_name == "RFC":
            self.model_classifiers = [
                RandomForestClassifier(
                    **hyperparameter,
                    # class_weight={0: 1, 1: 2}
                    )
            ] * NUM_CLUSTERS
        elif model_name == "LightGBM":
            self.model_classifiers = [
                LGBMClassifier(**hyperparameter, is_unbalance=True, verbosity=-1)
            ] * NUM_CLUSTERS
        elif model_name == "XGBoost":
            self.model_classifiers = [
                XGBClassifier(**hyperparameter, scale_pos_weight=2)
            ] * NUM_CLUSTERS

        self.handle_imbalance = handle_imbalance
        self.text_vecs = []


    def fit(self, X, y):
        self.text_vecs = [col for col in X.columns if "SentenceVec" in col]
        X_text_vecs = X[self.text_vecs].copy()
        X_text_vecs_scaled = self.scaler.fit_transform(X_text_vecs)
        labels = self.model_clustering.fit_predict(X_text_vecs_scaled)

        # ラベルごとにデータを分割・学習
        for label in range(self.NUM_CLUSTERS):
            X_sub = X[labels == label]
            y_sub = y[labels == label]
            # 重みづけを更新
            if self.hyperparameters["model_name"] == "RFC":
                pos_scale = y_sub.sum() / (len(y_sub) - y_sub.sum())
                self.model_classifiers[label] = RandomForestClassifier(
                    **self.hyperparameters["hyperparameter"],
                    # class_weight={0: 1, 1: pos_scale}
                )
            elif self.hyperparameters["model_name"] == "XGBoost":
                pos_scale = y_sub.sum() / (len(y_sub) - y_sub.sum())
                self.model_classifiers[label] = XGBClassifier(
                    **self.hyperparameters["hyperparameter"], scale_pos_weight=pos_scale
                )

            self.model_classifiers[label].fit(X_sub, y_sub)

    def predict(self, X):
        y_return = np.array([-1]*len(X))
        labels = self.model_clustering.predict(
            self.scaler.transform(X[self.text_vecs])
        )
        for label in range(self.NUM_CLUSTERS):
            X_sub = X[labels == label]
            if len(X_sub) == 0:
                pass
            else:
                y_sub = self.model_classifiers[label].predict(X_sub)
                y_return[labels == label] = y_sub
        # print(np.unique(y_return, return_counts=True))
        return y_return

    def predict_proba(self, X):

        y_proba_return = np.zeros((len(X), 2))
        labels = self.model_clustering.predict(
            self.scaler.transform(X[self.text_vecs])
        )
        for label in range(self.NUM_CLUSTERS):
            X_sub = X[labels == label]
            if len(X_sub)==0:
                pass
            else:
                y_sub = self.model_classifiers[label].predict_proba(X_sub)
                y_proba_return[labels == label] = y_sub
        return y_proba_return
