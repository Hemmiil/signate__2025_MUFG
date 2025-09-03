import os
# ← ここは "他のimportより前" に置く
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONFAULTHANDLER", "1")


import pandas as pd
import numpy as np
import sys
import json
import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from x04_01__NN import F01__MLP
from x04__02__ClusteringClassifier import F01__ClusteringClassifier
from x04__03__KNN import F01__KNC
from x04__04__CatBoost import F01__CatBoost
from x04__05__PyTorchMLP import F01__NNClassifier
from x04__06__RealMLP import F01__RealMLP
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

class F01__Stacking():
    def __init__(self, timestamp, is_test, is_validation, configs, is_halftest):

        self.n_splits = 5
        self.is_validation = is_validation
        self.configs = configs
        self.timestamp = timestamp
        self.is_test = is_test
        self.is_halftest = is_halftest

        self.is_PseudoLabeling = self.configs.get("is_PseudoLabeling", False)

    def f01__get_data(self):
        if self.is_PseudoLabeling:
            self.f01__02__get_Pseudo_labeling_data()
        else:
            if self.is_halftest:
                data = pd.read_csv("output/05__01__SentenceTransformer_Raw/train.csv", nrows=10000)
                target_col = "final_status"
                self.X = data.drop(target_col, axis="columns")
                self.y = data[target_col]

                data_submit = pd.read_csv("output/05__01__SentenceTransformer_Raw/test.csv")
                target_col = "final_status"
                self.X_submit = data_submit.drop(target_col, axis="columns")
                self.y_submit = None

            else:
                if self.is_test:
                    data = pd.read_csv("output/05__01__SentenceTransformer_Raw/train_test.csv")
                    target_col = "final_status"
                    self.X = data.drop(target_col, axis="columns")
                    self.y = data[target_col]
                    data_submit = pd.read_csv("output/05__01__SentenceTransformer_Raw/test.csv")
                    target_col = "final_status"
                    self.X_submit = data_submit.drop(target_col, axis="columns")
                    self.y_submit = None

                else:
                    data = pd.read_csv("output/05__01__SentenceTransformer_Raw/train.csv")
                    target_col = "final_status"
                    self.X = data.drop(target_col, axis="columns")
                    self.y = data[target_col]

                    data_submit = pd.read_csv("output/05__01__SentenceTransformer_Raw/test.csv")
                    target_col = "final_status"
                    self.X_submit = data_submit.drop(target_col, axis="columns")
                    self.y_submit = None

    def f01__02__get_Pseudo_labeling_data(self):
        # テストデータ
        path_add = "output/11__PseudoLabeling2/train_add.csv"
        path_origin = "output/05__01__SentenceTransformer_Raw/train.csv"

        data_train_origin = pd.read_csv(path_origin, index_col=0)
        data_train_add = pd.read_csv(path_add, index_col=0)
        data = pd.concat(
            [data_train_origin, data_train_add], axis="index"
        )
        target_col = "final_status"
        self.X = data.drop(target_col, axis="columns")
        self.y = data[target_col]

        data_submit = pd.read_csv("output/05__01__SentenceTransformer_Raw/test.csv")
        target_col = "final_status"
        self.X_submit = data_submit.drop(target_col, axis="columns")
        self.y_submit = None

    def f02__data_split(self):
        if self.is_validation:
            # train, val 分割
            self.test_size = 0.2
            self.dataset = {
                k: v for k, v in zip(
                    ["X_train", "X_test", "y_train", "y_test"],
                    train_test_split(
                        self.X, self.y, random_state=42,
                        test_size=self.test_size, stratify=self.y
                    )
                )
            }
        else:
            self.dataset = {
                "X_train": self.X,
                "X_test": self.X_submit,
                "y_train": self.y,
                "y_test": None
            }


    def f03__return_model(self, model_name, hyperparameters):
        # 既存HPをコピーして上書き
        hp = dict(hyperparameters or {})

        if model_name == "LightGBM":
            # num_threads / n_jobs のいずれか（両方あってもOK、LightGBM側が解釈）
            hp.setdefault("num_threads", 1)
            hp.setdefault("n_jobs", 1)
            hp.setdefault("verbose", -1)
            return LGBMClassifier(**hp, is_unbalance=True, verbosity=-1)

        elif model_name == "XGBoost":
            # XGBoostは nthread or n_jobs のどちらも可（古参互換）
            hp.setdefault("nthread", 1)
            hp.setdefault("n_jobs", 1)
            hp.setdefault("tree_method", "hist")
            hp.setdefault("eval_metric", "logloss")
            # macOSでlabel encoder警告回避
            return XGBClassifier(**hp, use_label_encoder=False)

        elif model_name == "MLP":
            return F01__MLP(hyperparameters=hp)

        elif model_name == "CluCla":
            return F01__ClusteringClassifier(hyperparameters=hp)

        elif model_name == "KNC":
            return F01__KNC(hyperparameters=hp)

        elif model_name == "NB":
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB()
        elif model_name == "CatBoost":
            return F01__CatBoost(hyperparameters=hp)
        elif model_name == "MLP_PyT":
            return F01__NNClassifier(hyperparameters=hp)
        elif model_name == "MLP_Emb":
            from x04__08__PyTorchNN_Emb import F01__NNClassifier_Emb
            return F01__NNClassifier_Emb(hyperparameters=hp)
        elif model_name == "RealMLP":
            return F01__RealMLP(hyperparameters=hp)
        elif model_name == "FeatEns":
            from x04__06__FeatsEnsemble import F01__LR_SubspaceEnsemble
            return F01__LR_SubspaceEnsemble(hyperparameters=hp)


        else:
            raise ValueError(f"Unknown model_name: {model_name}")


    def f04__base_train(self, n_fold=5):

        configs__base_models = self.configs["base_models"]

        proba_base_models = {
            "train": pd.DataFrame(),
            "test": pd.DataFrame()
        }
        cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
        for I, config in enumerate(configs__base_models):
            print(f"------Base Model: {config['model_id']} Start...------")
            if config["is_skip"]:
                if self.is_validation:
                    print(f"------! Learning is skipped !------")
                    col_name = f"{config['model_id']}__proba_pos"
                    proba_base_models["train"][col_name] = pd.read_csv(
                        "output/07__Stack/02__MetaFeats_train.csv", usecols=[col_name])
                    proba_base_models["test"][col_name] = pd.read_csv(
                        "output/07__Stack/02__MetaFeats_test.csv", usecols=[col_name])
                else:
                    print(f"------! Learning is skipped !------")
                    col_name = f"{config['model_id']}__proba_pos"
                    proba_base_models["train"][col_name] = pd.read_csv(
                        "output/07__Stack/02__MetaFeats_Submit_train.csv", usecols=[col_name])
                    proba_base_models["test"][col_name] = pd.read_csv(
                        "output/07__Stack/02__MetaFeats_Submit_test.csv", usecols=[col_name])

            else:
                model = self.f03__return_model(
                    model_name=config["model_name"],
                    hyperparameters=config["hyperparameters"]
                )
                preds_oof_train = np.zeros(len(self.dataset["X_train"]))
                preds_oof_test = np.zeros((len(self.dataset["X_test"]), n_fold))

                weights = np.zeros(n_fold)
                for i, (trn_idx, val_idx) in enumerate(
                    cv.split(self.dataset["X_train"], self.dataset["y_train"])
                    ):
                    print(f"------CV {i+1}...------")
                    trn_x, trn_y = self.dataset["X_train"].iloc[trn_idx, :], self.dataset["y_train"].iloc[trn_idx]
                    val_x, val_y = self.dataset["X_train"].iloc[val_idx, :], self.dataset["y_train"].iloc[val_idx]
                    model.fit(
                            trn_x, trn_y,
                    )
                    preds_oof_train_ = model.predict(val_x)
                    weights[i] = f1_score(val_y, preds_oof_train_)
                    preds_oof_train[val_idx] = model.predict_proba(val_x)[:, 1]
                    preds_oof_test[:, i] = model.predict_proba(self.dataset["X_test"])[:, 1]
                print(len(proba_base_models["train"]))
                print(len(preds_oof_train))
                proba_base_models["train"][f"{config['model_id']}__proba_pos"] = preds_oof_train
                # proba_base_models["test"][f"{str(I).zfill(2)}__proba_pos"] = preds_oof_test.mean(axis=1)
                proba_base_models["test"][f"{config['model_id']}__proba_pos"] = np.average(
                    preds_oof_test, axis=1,
                    weights=weights,
                    # weights=[1]*n_fold
                )

            self.proba_base_models = proba_base_models

            # メタ特徴量の保存
            for train_test, data in proba_base_models.items():
                if self.is_PseudoLabeling:
                    if self.is_validation:
                        if self.is_test:
                            if os.path.exists(f"output/07__Stack/04__MetaFeats_PseudoLabeling/{train_test}_test.csv"):
                                data_old = pd.read_csv(
                                    f"output/07__Stack/04__MetaFeats_PseudoLabeling/{train_test}_test.csv", index_col=0
                                )
                            else:
                                data_old = pd.DataFrame()
                            data_old = data_old.loc[:, ~data_old.columns.isin(data.columns)]
                            data_new = pd.concat([data_old, data], axis="columns")
                            data_new.to_csv(f"output/07__Stack/04__MetaFeats_PseudoLabeling/{train_test}_test.csv")
                        else:
                            if os.path.exists(f"output/07__Stack/04__MetaFeats_PseudoLabeling/{train_test}_test.csv"):
                                data_old = pd.read_csv(
                                    f"output/07__Stack/04__MetaFeats_PseudoLabeling/{train_test}.csv", index_col=0
                                )
                            else:
                                data_old = pd.DataFrame()
                            data_old = data_old.loc[:, ~data_old.columns.isin(data.columns)]
                            data_new = pd.concat([data_old, data], axis="columns")
                            data_new.to_csv(f"output/07__Stack/04__MetaFeats_PseudoLabeling/{train_test}.csv")
                        print("------ Meta Feats saved. ------")
                    else:
                        path = f"output/07__Stack/04__MetaFeats_PseudoLabeling/{train_test}.csv"
                        if os.path.exists(path):
                            data_old = pd.read_csv(
                                path, index_col=0
                            )
                            data_old = data_old.loc[:, ~data_old.columns.isin(data.columns)]
                            data_new = pd.concat([data_old, data], axis="columns")
                            data_new.to_csv(path)
                        else:
                            data.to_csv(path)
                        print("------ Meta Feats for Submit saved. ------")


                else:
                    if self.is_validation:
                        if self.is_test:
                            data_old = pd.read_csv(
                                f"output/07__Stack/02__MetaFeats_{train_test}_test.csv", index_col=0
                            )
                            data_old = data_old.loc[:, ~data_old.columns.isin(data.columns)]
                            data_new = pd.concat([data_old, data], axis="columns")
                            data_new.to_csv(f"output/07__Stack/02__MetaFeats_{train_test}_test.csv")
                        else:
                            data_old = pd.read_csv(
                                f"output/07__Stack/02__MetaFeats_{train_test}.csv", index_col=0
                            )
                            data_old = data_old.loc[:, ~data_old.columns.isin(data.columns)]
                            data_new = pd.concat([data_old, data], axis="columns")
                            data_new.to_csv(f"output/07__Stack/02__MetaFeats_{train_test}.csv")
                        print("------ Meta Feats saved. ------")
                    else:
                        path = f"output/07__Stack/02__MetaFeats_Submit_{train_test}.csv"
                        if os.path.exists(path):
                            data_old = pd.read_csv(
                                path, index_col=0
                            )
                            data_old = data_old.loc[:, ~data_old.columns.isin(data.columns)]
                            data_new = pd.concat([data_old, data], axis="columns")
                            data_new.to_csv(path)
                        else:
                            data.to_csv(path)
                        print("------ Meta Feats for Submit saved. ------")


    def f05__meta_fit_predict(self):
        X_train, X_test = self.f05__01__concat_meta_feats()
        meta_config = self.configs["meta_model"]
        meta_model = self.f03__return_model(
            model_name=meta_config["model_name"],
            hyperparameters=meta_config["hyperparameters"]
        )

        meta_model.fit(
            X_train, self.dataset["y_train"]
        )

        y_pred = meta_model.predict(X_test)

        if is_validation:
            feats__imp = {
                k: v for k, v in zip(
                    X_train.columns,
                    meta_model.feature_importances_.astype(float)
                )
            }

            score = f1_score(
                self.dataset["y_test"], y_pred
            )

            return {
                "y_pred": y_pred,
                "f1_score": score,
                "feat__imp": feats__imp
            }
        else:
            return y_pred

    def f05__meta_fit_predict_2(self):
        # メタモデルを単一モデル、全領域を学習 → OOF
        X_train, X_test = self.f05__02__concat_meta_pc()
        meta_config = self.configs["meta_model"]

        from sklearn.model_selection import StratifiedKFold
        n_splits = 5
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        y_pred_proba = np.zeros((len(self.dataset["X_test"]), n_splits))
        weights = []
        for i, (trn_idx, val_idx) in enumerate(
                        cv.split(X_train, self.dataset["y_train"])
                        ):
            print(f"------MetaModel CV {i+1} Start...------")
            model = self.f03__return_model(
                model_name = meta_config["model_name"],
                hyperparameters=meta_config["hyperparameters"]
            )
            model.fit(X_train.iloc[trn_idx], self.dataset["y_train"].iloc[trn_idx])
            y_pred_proba_ = model.predict_proba(X_test)[:, 1]
            y_pred_proba[:, i] = y_pred_proba_
            weights.append(
                f1_score(
                    model.predict(X_train.iloc[val_idx]), self.dataset["y_train"].iloc[val_idx]
                    )
                )

        y_pred_proba_weighted = np.average(y_pred_proba, weights=weights, axis=1)
        y_pred = np.array([1 if v > 0.5 else 0 for v in y_pred_proba_weighted])

        if is_validation:

            score = f1_score(
                self.dataset["y_test"], y_pred
            )

            return {
                "y_pred": y_pred,
                "f1_score": score,
            }
        else:
            return y_pred


    def f05__meta_fit_predict_3(self, is_validation: bool = None, n_splits: int = 5, threshold: float = 0.5):
        """
        複数メタモデルのCV加重平均 → モデル間加重平均で最終予測を返す。
        self.configs["meta_model"] は以下のリストを想定：
        [
            {"model_name": "lightgbm", "hyperparameters": {...}},
            {"model_name": "logreg",   "hyperparameters": {...}},
            ...
        ]
        """
        import numpy as np
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score

        # ===== データ取得 =====
        # X_train, X_test = self.f05__01__concat_meta_feats()
        X_train, X_test = self.f05__02__concat_meta_pc()
        y_train = self.dataset["y_train"]

        # is_validation が未指定なら、self.is_validation または y_test の有無で推定
        if is_validation is None:
            is_validation = bool(getattr(self, "is_validation", "y_test" in self.dataset))

        # ===== メタモデル設定（複数） =====
        meta_configs = self.configs.get("meta_model", None)
        if meta_configs is None or len(meta_configs) == 0:
            raise ValueError("self.configs['meta_model'] にメタモデル設定（リスト）がありません。")

        if isinstance(meta_configs, dict):
            # 後方互換：単一辞書だったらリスト化
            meta_configs = [meta_configs]

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # ===== 各モデルでCVしつつ X_test へ確率出力 → fold F1で加重平均 =====
        model_level_probas = []   # shape: (n_samples_test,) をモデル数だけ
        model_level_weights = []  # 各モデルの重み（= fold F1 の平均）
        feat__imp = []

        def _predict_proba_1d(clf, X):
            # predict_proba が無いモデル対策
            if hasattr(clf, "predict_proba"):
                p = clf.predict_proba(X)
                return p[:, 1] if p.ndim == 2 else p.ravel()
            elif hasattr(clf, "decision_function"):
                from sklearn.preprocessing import MinMaxScaler
                s = clf.decision_function(X).reshape(-1, 1)
                return MinMaxScaler().fit_transform(s).ravel()
            else:
                # 0/1 予測しか返せない場合は確率っぽく扱う（要注意）
                return clf.predict(X).astype(float)

        for mi, cfg in enumerate(meta_configs):
            model_name = cfg["model_name"]
            hyperparams = cfg.get("hyperparameters", {})

            # foldごとの test 確率と重み（val F1）
            fold_probas = np.zeros((len(X_test), n_splits), dtype=float)
            fold_weights = []

            feat__imp_ = []

            for fi, (trn_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                print(f"[MetaModel {mi+1}/{len(meta_configs)}: {model_name}] CV {fi+1}/{n_splits} ...")

                # モデル作成＆学習
                model = self.f03__return_model(model_name=model_name, hyperparameters=hyperparams)
                model.fit(X_train.iloc[trn_idx], y_train.iloc[trn_idx])

                # X_test へ確率出力
                fold_probas[:, fi] = _predict_proba_1d(model, X_test)

                # fold の検証スコア（重み）
                y_val_pred = model.predict(X_train.iloc[val_idx])
                f1 = f1_score(y_train.iloc[val_idx], y_val_pred)
                fold_weights.append(max(f1, 1e-6))  # 0割防止のため微小値で下駄

                # 特徴量重要度
                if hasattr(model, "feature_importances_"):
                    feat__imp_.append(
                        {
                            k: v for k, v in zip(
                                X_train.columns,
                                model.feature_importances_.astype("float")
                            )
                        }
                    )
                else:
                    feat__imp_.append({})

            # fold 加重平均で当該モデルの test 確率を1本化
            fold_weights = np.asarray(fold_weights, dtype=float)
            fold_weights = fold_weights / (fold_weights.sum() + 1e-12)
            proba_model = np.average(fold_probas, weights=fold_weights, axis=1)

            model_level_probas.append(proba_model)
            model_level_weights.append(float(fold_weights.mean()))  # モデル間重み＝fold重みの平均（簡便）

            # fold-weight が最も大きなモデルを代表してfeat__imp に追加
            feat__imp.append(
                feat__imp_[np.argmax(fold_weights)]
            )

        # ===== モデル間アンサンブル（加重平均） =====
        model_level_probas = np.vstack(model_level_probas).T  # shape: (n_samples_test, n_models)
        model_level_weights = np.asarray(model_level_weights, dtype=float)
        if model_level_weights.sum() <= 0:
            # すべて0近傍なら均等重み
            model_level_weights = np.ones_like(model_level_weights) / len(model_level_weights)
        else:
            model_level_weights = model_level_weights / model_level_weights.sum()

        y_pred_proba = np.average(model_level_probas, weights=model_level_weights, axis=1)
        y_pred = (y_pred_proba > threshold).astype(int)

        # ===== 検証（任意） =====
        if is_validation and ("y_test" in self.dataset):
            from sklearn.metrics import f1_score
            score = f1_score(self.dataset["y_test"], y_pred)
            return {
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba,
                "f1_score": score,
                "model_weights": model_level_weights.tolist(),
                "feat__imp": feat__imp
            }
        else:
            return y_pred

    def f05__meta_fit_predict_4(self, n_splits: int = 5, threshold: float = 0.5, top_k_feats: int = 100):
        """
        configs['meta_model'] の各要素に 'is_feats_select' を受け付ける新バージョン。
        is_feats_select が True の場合は次を実施:
        1) 従来通り model_name/hyperparameters で k-fold OOF を実行し、
            model_level_probas / model_level_weights に追加（ベースモデル）
        2) 1) の fold_weight が最大の fold の特徴量重要度 feat__imp_ に準拠し、
            上位 top_k_feats を選択して再度 k-fold OOF を実行（feats-select-model）
        3) その結果（予測と重み）も model_level_probas / model_level_weights に追加

        返り値（is_validation=True かつ 'y_test' がある場合）:
            {
            "y_pred": np.ndarray[int],                 # (n_samples_test,)
            "y_pred_proba": np.ndarray[float],         # (n_samples_test,)
            "f1_score": float,
            "model_weights": list[float],              # モデル間重み（正規化後）
            "feat__imp": list[dict[str, float]],       # 各メタモデルの代表foldの重要度辞書
            }
        それ以外の場合: y_pred (np.ndarray[int])
        """
        import numpy as np
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score

        # ===== データ取得 =====
        # X_train, X_test = self.f05__01__concat_meta_feats()
        X_train, X_test = self.f05__02__concat_meta_pc()
        y_train = self.dataset["y_train"]


        # ===== メタモデル設定（複数） =====
        meta_configs = self.configs.get("meta_model", None)
        if meta_configs is None or len(meta_configs) == 0:
            raise ValueError("self.configs['meta_model'] にメタモデル設定（リスト）がありません。")
        if isinstance(meta_configs, dict):
            meta_configs = [meta_configs]

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # ===== 出力バッファ =====
        model_level_probas = pd.DataFrame()   # 各モデル（＆feats-select-model）の test 確率ベクトル
        model_level_weights = []  # 各モデル（＆feats-select-model）の重み（fold F1の平均）
        feat__imp = []            # 各メタモデルの「代表fold（最大fold_weight）」の重要度辞書

        def _predict_proba_1d(clf, X):
            # predict_proba が無いモデル対策
            if hasattr(clf, "predict_proba"):
                p = clf.predict_proba(X)
                return p[:, 1] if p.ndim == 2 else p.ravel()
            elif hasattr(clf, "decision_function"):
                from sklearn.preprocessing import MinMaxScaler
                s = clf.decision_function(X).reshape(-1, 1)
                return MinMaxScaler().fit_transform(s).ravel()
            else:
                # 0/1 予測しか返せない場合は確率っぽく扱う（要注意）
                return clf.predict(X).astype(float)

        def _run_kfold_oof(model_name, hyperparams, X_tr, X_te, y_tr):
            """与えられた (model_name, hyperparams) と (X_tr, X_te, y_tr) で KFold OOF を実行。
            戻り値:
                proba_model: np.ndarray, shape (n_samples_test,)
                weight_mean: float,      fold重み平均
                fold_weights: np.ndarray, shape (n_splits,)
                feat_imp_list: list[dict], fold毎の重要度辞書（なければ {}）
            """
            fold_probas = np.zeros((len(X_te), n_splits), dtype=float)
            fold_weights = []
            feat_imp_list = []

            for fi, (trn_idx, val_idx) in enumerate(cv.split(X_tr, y_tr)):
                print(f"[MetaModel {model_name}] CV {fi+1}/{n_splits} ...")

                model = self.f03__return_model(model_name=model_name, hyperparameters=hyperparams)
                model.fit(X_tr.iloc[trn_idx], y_tr.iloc[trn_idx])

                # 予測（test）
                fold_probas[:, fi] = _predict_proba_1d(model, X_te)

                # 重み（val F1）
                y_val_pred = model.predict(X_tr.iloc[val_idx])
                f1 = f1_score(y_tr.iloc[val_idx], y_val_pred)
                fold_weights.append(max(f1, 1e-6))

                # 重要度
                if hasattr(model, "feature_importances_"):
                    feat_imp_list.append(
                        {k: float(v) for k, v in zip(X_tr.columns, model.feature_importances_.astype("float"))}
                    )
                else:
                    feat_imp_list.append({})

            fold_weights = np.asarray(fold_weights, dtype=float)
            w = fold_weights / (fold_weights.sum() + 1e-12)
            proba_model = np.average(fold_probas, weights=w, axis=1)
            weight_mean = float(w.mean())
            return proba_model, weight_mean, fold_weights, feat_imp_list

        def _compute_weights(f1_scores, method="naive", metrics_power=4, temperature=1):
            scores = np.array(f1_scores, dtype=float)
            if method == "naive":
                w = scores
            elif method == "power":
                w = scores ** metrics_power
            elif method == "softmax":
                w = np.exp(scores / temperature)
            elif method == "minmax":
                s = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                w = s ** metrics_power
            elif method == "rank":
                ranks = len(scores) - np.argsort(np.argsort(scores))
                w = ranks
            else:
                raise ValueError("Unknown method")
            return w / w.sum()

        for cfg in meta_configs:
            model_name = cfg["model_name"]
            model_id = cfg["model_id"]
            hyperparams = cfg.get("hyperparameters", {})
            is_feats_select = bool(cfg.get("is_feats_select", False))

            # ==== 1) ベースモデル: 通常の OOF ====
            proba_base, w_base, fold_w_base, feat_imp_list = _run_kfold_oof(
                model_name=model_name,
                hyperparams=hyperparams,
                X_tr=X_train,
                X_te=X_test,
                y_tr=y_train
            )
            model_level_probas[model_id] = proba_base
            model_level_weights.append(w_base)

            # 代表fold（fold_weight 最大）の重要度を保存
            rep_idx = int(np.argmax(fold_w_base)) if len(fold_w_base) > 0 else 0
            rep_imp_dict = feat_imp_list[rep_idx] if len(feat_imp_list) > 0 else {}
            feat__imp.append(rep_imp_dict)

            # ==== 2) is_feats_select=True のとき、重要度上位 top_k_feats で再学習 ====
            if is_feats_select and len(rep_imp_dict) > 0:
                # 上位特徴量を抽出
                sorted_feats = sorted(rep_imp_dict.items(), key=lambda x: x[1], reverse=True)
                top_feats = [k for k, _ in sorted_feats[:min(top_k_feats, len(sorted_feats))]]

                # サブセット作成（存在カラムのみ）
                top_feats = [c for c in top_feats if c in X_train.columns]
                if len(top_feats) > 0:
                    X_train_sel = X_train[top_feats]
                    X_test_sel = X_test[top_feats]

                    # ==== 3) feats-select-model の OOF ====
                    proba_fs, w_fs, _, _ = _run_kfold_oof(
                        model_name=model_name,
                        hyperparams=hyperparams,
                        X_tr=X_train_sel,
                        X_te=X_test_sel,
                        y_tr=y_train
                    )
                    model_level_probas.append(proba_fs)
                    model_level_weights.append(w_fs)
                else:
                    # top_feats が空になる保険（稀）
                    print(f"[MetaModel {model_name}] Feature selection skipped (no valid top features).")

        # ===== アンサンブル前予測値の保存 =====
        if self.is_PseudoLabeling:
            os.makedirs(f"output/07__Stack/03__Ensemble/PseudoLabeling/{self.timestamp}", exist_ok=True)
            if self.is_validation:
                model_level_probas.to_csv(f"output/07__Stack/03__Ensemble/PseudoLabeling/{self.timestamp}/model_level_probas.csv")
                self.dataset["y_test"].to_csv(f"output/07__Stack/03__Ensemble/PseudoLabeling/{self.timestamp}/y_test.csv")
            else:
                model_level_probas.to_csv(f"output/07__Stack/03__Ensemble/PseudoLabeling/{self.timestamp}/model_level_probas_is_submit.csv")

        else:
            os.makedirs(f"output/07__Stack/03__Ensemble/{self.timestamp}", exist_ok=True)
            if self.is_validation:
                model_level_probas.to_csv(f"output/07__Stack/03__Ensemble/{self.timestamp}/model_level_probas.csv")
                self.dataset["y_test"].to_csv(f"output/07__Stack/03__Ensemble/{self.timestamp}/y_test.csv")
            else:
                model_level_probas.to_csv(f"output/07__Stack/03__Ensemble/{self.timestamp}/model_level_probas_is_submit.csv")

        # ===== モデル間アンサンブル（加重平均） =====
        model_level_weights_computed = _compute_weights(
            f1_scores = np.asarray(model_level_weights, dtype=float),
            method=self.configs.get("weight_method", "naive")
            )
        if model_level_weights_computed.sum() <= 0:
            model_level_weights_computed = np.ones_like(model_level_weights_computed) / len(model_level_weights_computed)
        else:
            model_level_weights_computed = model_level_weights_computed / model_level_weights_computed.sum()

        y_pred_proba = model_level_probas.dot(model_level_weights_computed)
        y_pred = (y_pred_proba > threshold).astype(int)

        # ===== 検証（任意） =====
        if self.is_validation:
            score = f1_score(self.dataset["y_test"], y_pred)
            return {
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba,
                "f1_score": score,
                "model_weights": {
                    k: v  for k, v in zip(
                        [v["model_id"] for v in meta_configs],
                        model_level_weights
                    )
                },
                "feat__imp": feat__imp
            }
        else:
            return y_pred

    def f05__01__concat_meta_feats(self):
        # --- X の作成 ---
        X_train = pd.concat(
            [
                self.dataset["X_train"].reset_index(drop=True),
                self.proba_base_models["train"].reset_index(drop=True)
            ],
            axis="columns"
        )
        X_test = pd.concat(
            [
                self.dataset["X_test"].reset_index(drop=True),
                self.proba_base_models["test"].reset_index(drop=True)
            ],
            axis="columns"
        )

        print("meta_train_X.shape:", X_train.shape)
        print("meta_train_y.shape:", X_test.shape)

        return X_train, X_test

    def f05__02__concat_meta_pc(self):
        # --- X の作成 ---
        print("------ Meta Feats converting to PC... ------")
        from sklearn.decomposition import PCA
        pca = PCA(random_state=42)
        meta_train = self.proba_base_models["train"].reset_index(drop=True)
        X_train = pd.concat(
            [
                self.dataset["X_train"].reset_index(drop=True),
                pd.DataFrame(
                    pca.fit_transform(meta_train),
                    columns=[f"MetaFeatsPC_{1+i}" for i in range(len(meta_train.columns))],
                    index = meta_train.index
                )
            ],
            axis="columns"
        )
        meta_test = self.proba_base_models["test"].reset_index(drop=True)
        X_test = pd.concat(
            [
                self.dataset["X_test"].reset_index(drop=True),
                pd.DataFrame(
                    pca.transform(meta_test),
                    columns=[f"MetaFeatsPC_{1+i}" for i in range(len(meta_test.columns))],
                    index = meta_test.index
                )
            ],
            axis="columns"
        )

        print("meta_train_X.shape:", X_train.shape)
        print("meta_train_y.shape:", X_test.shape)

        return X_train, X_test

    def f06__save(self, result):
        """最終予測とメタモデルの設定を保存する。"""
        output_dir = "output/07__Stacking"
        os.makedirs(output_dir, exist_ok=True)

        if self.is_validation:
            validation_output_dir = os.path.join(output_dir, "02__validation", self.timestamp)
            os.makedirs(validation_output_dir, exist_ok=True)

            prediction_filepath = os.path.join(validation_output_dir, "01__y_pred.npy")
            np.save(prediction_filepath, result["y_pred"])
            print(f"検証予測が {prediction_filepath} に保存されました。" )


            config_path = "output/07__Stack/01__configs_CV.json"
            with open(config_path, "r") as f:
                config_exp = json.load(f)

            configs = {
                "HP": config_exp,
                "is_validation": self.is_validation,
                "is_test": self.is_test,
                "f1_score": result["f1_score"],
                "feat__imp": result["feat__imp"],
                "model_weights": result["model_weights"]
            }
            configs_filepath = os.path.join(validation_output_dir, "configs.json")
            with open(configs_filepath, 'w') as f:
                json.dump(configs, f, indent=2)
            print(f"F1スコア: {result['f1_score']}")
            print(f"検証設定が {configs_filepath} に保存されました。" )
        else:
            config_path = "output/07__Stack/01__configs_CV.json"
            with open(config_path, "r") as f:
                config_exp = json.load(f)

            submission_output_dir = os.path.join(output_dir, "03__submission", self.timestamp)
            os.makedirs(submission_output_dir, exist_ok=True)

            prediction_filepath = os.path.join(submission_output_dir, "01__y_pred.npy")
            np.save(prediction_filepath, result)
            print(f"提出予測が {prediction_filepath} に保存されました。" )

            configs = {
                "HP": config_exp,
                "is_validation": self.is_validation,
                "is_test": self.is_test,
            }
            configs_filepath = os.path.join(submission_output_dir, "configs.json")
            with open(configs_filepath, 'w') as f:
                json.dump(configs, f, indent=2)
            print(f"提出設定が {configs_filepath} に保存されました。" )




if __name__ == "__main__":
    is_validation = '--validation' in sys.argv
    is_test = '--test' in sys.argv
    is_halftest = '--halftest' in sys.argv

    print(f"スタッキングを実行中: validation={is_validation}, test_mode={is_test}, half_test_mode={is_halftest}" )

    config_path = "output/07__Stack/01__configs_CV.json"
    try:
        with open(config_path, 'r') as f:
            configs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"スタッキング設定ファイル '{config_path}' の読み込みエラー: {e}")
        sys.exit(1)

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    stacker = F01__Stacking(
        is_validation=is_validation,
        is_test=is_test,
        is_halftest=is_halftest,
        configs=configs,
        timestamp=timestamp
    )

    stacker.f01__get_data()
    stacker.f02__data_split()
    stacker.f04__base_train(n_fold=5)
    # result = stacker.f05__meta_fit_predict()

    ## 単一メタモデル
    # result = stacker.f05__meta_fit_predict_2()

    ## メタモデルのアンサンブル化
    result = stacker.f05__meta_fit_predict_4(n_splits=5)
    stacker.f06__save(result=result)

    print("スタッキングプロセスが完了しました。" )