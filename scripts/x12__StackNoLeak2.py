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
    def __init__(self, timestamp, is_validation, configs):

        self.n_splits = 5
        self.is_validation = is_validation
        self.configs = configs
        self.timestamp = timestamp

        self.is_add = True

    def f01__get_data(self):
        self.f01__02__get_Pseudo_labeling_data()

    def f01__02__get_Pseudo_labeling_data(self):
        # テストデータ
        path_add = "output/11__PseudoLabeling2/train_add.csv"
        path_origin = "output/05__01__SentenceTransformer_Raw/train.csv"

        data = pd.read_csv(path_origin)
        target_col = "final_status"
        self.X = data.drop(target_col, axis="columns")
        self.y = data[target_col]
        print("train done")

        data_add = pd.read_csv(path_add)
        target_col = "final_status"
        self.X_add = data_add.drop(target_col, axis="columns")
        self.y_add = data_add[target_col]
        print("add done")
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
        self.dataset["X_add"] = self.X_add
        self.dataset["y_add"] = self.y_add


        # add を追加した場合のpos_neg_ratio を計算する
        y_total = pd.concat([self.dataset["y_train"], self.dataset["y_add"]])
        ratio = y_total.sum() / len(y_total)
        self.pos_scale_weight = (1 - ratio) / ratio



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
            return XGBClassifier(**hp, use_label_encoder=False, scale_pos_weight=self.pos_scale_weight)

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
            "test": pd.DataFrame(),
            "add": pd.DataFrame(),
        }
        cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
        for I, config in enumerate(configs__base_models):
            print(f"------Base Model: {config['model_id']} Start...------")


            model = self.f03__return_model(
                model_name=config["model_name"],
                hyperparameters=config["hyperparameters"]
            )
            preds_oof_train = np.zeros(len(self.dataset["X_train"]))
            preds_oof_test = np.zeros((len(self.dataset["X_test"]), n_fold))
            preds_oof_add = np.zeros(len(self.X_add))

            weights = np.zeros(n_fold)
            for i, ((trn_idx_train, val_idx_train), (trn_idx_add, val_idx_add)) in enumerate(
                zip(
                    cv.split(self.dataset["X_train"], self.dataset["y_train"]),
                    cv.split(self.dataset["X_add"], self.dataset["y_add"])
                )
                ):
                print(f"------CV {i+1}...------")
                trn_x, trn_y = self.dataset["X_train"].iloc[trn_idx_train, :], self.dataset["y_train"].iloc[trn_idx_train]
                val_x, val_y = self.dataset["X_train"].iloc[val_idx_train, :], self.dataset["y_train"].iloc[val_idx_train]
                add_x, add_y = self.dataset["X_add"].iloc[trn_idx_add, :], self.dataset["y_add"].iloc[trn_idx_add]


                print(f"data lengthes: {list(map(lambda x: x.shape, [trn_x, val_x, add_x]))}")

                # trn_x, trn_yへの擬似ラベルの追加

                fit_x = pd.concat(
                    [trn_x, add_x], axis="index"
                ).reset_index(drop=True)

                fit_y = pd.concat(
                    [trn_y, add_y],
                ).reset_index(drop=True)

                model.fit(
                        fit_x, fit_y
                )
                preds_oof_train_ = model.predict(val_x)
                weights[i] = f1_score(val_y, preds_oof_train_)
                preds_oof_train[val_idx_train] = model.predict_proba(val_x)[:, 1]
                preds_oof_test[:, i] = model.predict_proba(self.dataset["X_test"])[:, 1]


                preds_oof_add[val_idx_add] = model.predict_proba(
                    self.dataset["X_add"].iloc[val_idx_add]
                )[:, 1]
            print(len(proba_base_models["train"]))
            print(len(preds_oof_train))
            proba_base_models["train"][f"{config['model_id']}__proba_pos"] = preds_oof_train
            # proba_base_models["test"][f"{str(I).zfill(2)}__proba_pos"] = preds_oof_test.mean(axis=1)
            proba_base_models["test"][f"{config['model_id']}__proba_pos"] = np.average(
                preds_oof_test, axis=1,
                weights=weights,
                # weights=[1]*n_fold
            )
            proba_base_models["add"][f"{config['model_id']}__proba_pos"] = preds_oof_add

            self.proba_base_models = proba_base_models

    def f05__meta_fit_predict_4(self, n_splits: int = 5, threshold: float = 0.5, top_k_feats: int = 100, is_skipped=False):
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
        if is_skipped:
            X_train, X_test, X_add = self.f05__03__concat_meta_BL_skip()
        else:
            X_train, X_test, X_add = self.f05__02__concat_meta_pc()

        y_train = self.dataset["y_train"]
        y_add = self.dataset["y_add"]


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

        def _run_kfold_oof(model_name, hyperparams, X_tr, X_te, y_tr, X_ad, y_ad):
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

            for fi, ((trn_idx_train, val_idx_train), (trn_idx_add, val_idx_add)) in enumerate(zip(
                cv.split(X_tr, y_tr), cv.split(X_ad, y_ad)
            )):
                print(f"[MetaModel {model_name}] CV {fi+1}/{n_splits} ...")

                model = self.f03__return_model(model_name=model_name, hyperparameters=hyperparams)

                X_fit = pd.concat(
                    [X_tr.iloc[trn_idx_train], X_ad.iloc[trn_idx_add]], axis="index"
                )
                y_fit = pd.concat(
                    [y_tr.iloc[trn_idx_train], y_ad.iloc[trn_idx_add]], axis="index"
                )
                model.fit(X_fit, y_fit)

                # 予測（test）
                fold_probas[:, fi] = _predict_proba_1d(model, X_te)

                # 重み（val F1）
                y_val_pred = model.predict(X_tr.iloc[val_idx_train])
                f1 = f1_score(y_tr.iloc[val_idx_train], y_val_pred)
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
                y_tr=y_train,
                X_ad=X_add,
                y_ad=y_add
            )
            model_level_probas[model_id] = proba_base
            model_level_weights.append(w_base)

            # 代表fold（fold_weight 最大）の重要度を保存
            rep_idx = int(np.argmax(fold_w_base)) if len(fold_w_base) > 0 else 0
            rep_imp_dict = feat_imp_list[rep_idx] if len(feat_imp_list) > 0 else {}
            feat__imp.append(rep_imp_dict)

        # ===== アンサンブル前予測値の保存 =====

        os.makedirs(f"output/11__Stack/03__Ensemble/{self.timestamp}", exist_ok=True)
        if self.is_validation:
            model_level_probas.to_csv(f"output/11__Stack/03__Ensemble/{self.timestamp}/model_level_probas.csv")
            self.dataset["y_test"].to_csv(f"output/11__Stack/03__Ensemble/{self.timestamp}/y_test.csv")
        else:
            model_level_probas.to_csv(f"output/11__Stack/03__Ensemble/{self.timestamp}/model_level_probas_is_submit.csv")

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
            return {
                "y_pred": y_pred,
                "model_weights": {
                    k: v  for k, v in zip(
                        [v["model_id"] for v in meta_configs],
                        model_level_weights
                    )
                },
            }

    def f05__02__concat_meta_pc(self):
        # --- X の作成 ---
        print("------ Meta Feats converting to PC... ------")
        from sklearn.decomposition import PCA
        pca = PCA(random_state=42)
        meta_train = self.proba_base_models["train"].reset_index(drop=True)
        meta_add = self.proba_base_models["add"].reset_index(drop=True)
        meta_train_add = pd.concat([meta_train, meta_add], axis="index").reset_index(drop=True)
        pca.fit(meta_train_add)
        X_train = pd.concat(
            [
                self.dataset["X_train"].reset_index(drop=True),
                pd.DataFrame(
                    pca.transform(meta_train),
                    columns=[f"MetaFeatsPC_{1+i}" for i in range(len(meta_train.columns))],
                    index = meta_train.index
                )
            ],
            axis="columns"
        )
        X_add = pd.concat(
            [
                self.dataset["X_add"].reset_index(drop=True),
                pd.DataFrame(
                    pca.transform(meta_add),
                    columns=[f"MetaFeatsPC_{1+i}" for i in range(len(meta_add.columns))],
                    index = meta_add.index
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
        print("meta_add_X.shape:", X_add.shape)
        print("meta_test_X.shape:", X_test.shape)

        # メタ特徴量こみのデータ保存
        os.makedirs("output/11__Stack/04__dataset_with_Meta/", exist_ok=True)
        for name, data in zip(
            ["train", "add", "test"],
            [X_train, X_add, X_test],
        ):
            data.to_csv(f"output/11__Stack/04__dataset_with_Meta/X_{name}.csv")

            if not self.is_validation and name=="test":
                pass
            else:
                self.dataset[f"y_{name}"].to_csv(
                    f"output/11__Stack/04__dataset_with_Meta/y_{name}.csv"
                )


        return X_train, X_test, X_add

    def f05__03__concat_meta_BL_skip(self):
        X_train = pd.read_csv("output/11__Stack/04__dataset_with_Meta/X_train.csv", index_col=0)
        X_add = pd.read_csv("output/11__Stack/04__dataset_with_Meta/X_add.csv", index_col=0)
        X_test = pd.read_csv("output/11__Stack/04__dataset_with_Meta/X_test.csv", index_col=0)

        return X_train, X_test, X_add

    def f06__save(self, result):
        """最終予測とメタモデルの設定を保存する。"""
        output_dir = "output/11__Stack"
        os.makedirs(output_dir, exist_ok=True)

        if self.is_validation:
            validation_output_dir = os.path.join(output_dir, "02__validation", self.timestamp)
            os.makedirs(validation_output_dir, exist_ok=True)

            prediction_filepath = os.path.join(validation_output_dir, "01__y_pred.npy")
            np.save(prediction_filepath, result["y_pred"])
            print(f"検証予測が {prediction_filepath} に保存されました。" )


            config_path = "output/11__Stack/01__configs_CV.json"
            with open(config_path, "r") as f:
                config_exp = json.load(f)

            configs = {
                "HP": config_exp,
                "is_validation": self.is_validation,
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
            config_path = "output/11__Stack/01__configs_CV.json"
            with open(config_path, "r") as f:
                config_exp = json.load(f)

            submission_output_dir = os.path.join(output_dir, "03__submission", self.timestamp)
            os.makedirs(submission_output_dir, exist_ok=True)

            prediction_filepath = os.path.join(submission_output_dir, "01__y_pred.npy")
            np.save(prediction_filepath, result["y_pred"])
            print(f"提出予測が {prediction_filepath} に保存されました。" )

            configs = {
                "HP": config_exp,
                "is_validation": self.is_validation,
                "model_weights": result["model_weights"]
            }
            configs_filepath = os.path.join(submission_output_dir, "configs.json")
            with open(configs_filepath, 'w') as f:
                json.dump(configs, f, indent=2)
            print(f"提出設定が {configs_filepath} に保存されました。" )




if __name__ == "__main__":
    is_validation = '--validation' in sys.argv

    print(f"スタッキングを実行中: validation={is_validation}" )

    config_path = "output/11__Stack/01__configs_CV.json"
    try:
        with open(config_path, 'r') as f:
            configs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"スタッキング設定ファイル '{config_path}' の読み込みエラー: {e}")
        sys.exit(1)

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    stacker = F01__Stacking(
        is_validation=is_validation,
        configs=configs,
        timestamp=timestamp
    )

    stacker.f01__get_data()
    stacker.f02__data_split()

    is_skipped = False

    if is_skipped:
        pass
    else:
        stacker.f04__base_train(n_fold=5)

    ## メタモデルのアンサンブル化
    result = stacker.f05__meta_fit_predict_4(n_splits=5, is_skipped=is_skipped)
    stacker.f06__save(result=result)

    print("スタッキングプロセスが完了しました。" )