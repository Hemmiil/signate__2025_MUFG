# データダウンロード
from sklearn.model_selection import validation_curve
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from x04_01__NN import F01__MLP


class F01__HPT():
    def __init__(self):
        self.nrows = 1000
        self.params_lightGBM = {
            'random_state': 42,  # 乱数シード
            # 'n_estimators': 50,  # 最大学習サイクル数。early_stopping使用時は大きな値を入力
            'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
            # 'is_unbalance': True,
            'reg_alpha': 1e2,
            'reg_lambda': 1e2,
            #'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0],
            #'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],
            #'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
            'n_jobs': -1,
            'data_sample_strategy': 'bagging',
            "scale_pos_weight": 2,
            "max_depth": 8,
            "n_estimators": 100,
            "num_leaves": 64
            }
        self.params_XGBoost = {
            "n_estimators": 50,
            "n_jobs": -1
        }
        self.params_MLP = {
            'random_state': 42,
            'max_iter': 300,
        }

        self.params_CatBoost = {
            "iterations": 50,          # 学習する木の数（boosting の繰り返し回数）
            # "learning_rate": 0.03,       # 学習率（小さくすると安定だが時間がかかる）
            #"depth": 6,                  # 決定木の深さ
            #"l2_leaf_reg": 3.0,          # L2 正則化係数
            #"border_count": 254,         # 連続値特徴量の分割数（CatBoost 独自）
            #"loss_function": "Logloss",  # バイナリ分類用の損失関数
            #"eval_metric": "AUC",        # 評価指標
            # "random_seed": 42,           # 乱数シード
            #"logging_level": "Silent",   # 学習ログの出力レベル
            #"use_best_model": True,      # early_stopping時にベストモデルを保存
            #"task_type": "CPU",          # "GPU"も選択可
        }

    def f01__get_data(self):
        self.dataset = {
            train_test: pd.read_csv(
                f"output/05__SentenceTransformer/{train_test}.csv",  nrows=self.nrows
            ) for train_test in ["train", "test"]
        }

    def f01__get_data2(self):
        feats__meta = pd.read_csv("output/07__Stacking/04__MetaModelTrain/train.csv", index_col=0)
        feats__base = pd.read_csv("output/05__01__SentenceTransformer_Raw/train.csv")

        feats__concat = pd.concat(
            [feats__base, feats__meta], axis="columns"
        )
        self.dataset = {
            "train": feats__concat
        }

    def f02__split_data(self):
        # データ整形
        target_col = "final_status"
        X, y = self.dataset["train"].drop(target_col, axis="columns").copy(), self.dataset["train"][target_col].copy()

        from sklearn.model_selection import train_test_split

        X_cv, X_eval, y_cv, y_eval = train_test_split(
            X.values, y.values, stratify=y.values,
            test_size=0.25, random_state=42)

        return X_cv, X_eval, y_cv, y_eval

    def f03__HPT_lightGBM(self):
        cv_params = {
            "num_leaves": [
                64, 128, 256
            ]
        }
        param_scales = {
            "num_leaves": "log",
        }

        scoring = "f1"

        params = self.params_lightGBM

        X_cv, X_eval, y_cv, y_eval = self.f02__split_data()
        # 検証曲線のプロット（パラメータ毎にプロット）
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.params_lightGBM["random_state"])  # KFoldでクロスバリデーション分割指定
        model = LGBMClassifier(**params)
        # 学習時fitパラメータ指定 (early_stopping用のデータeval_setを渡す)
        fit_params = {
            'eval_set': [(X_eval, y_eval)],
            # 'early_stopping_rounds': 10,
            }
        print(f"------ Exp Start... ------")
        for i, (k, v) in enumerate(cv_params.items()):
            train_scores, valid_scores = validation_curve(estimator=model,
                                                        X=X_cv, y=y_cv,
                                                        param_name=k,
                                                        param_range=v,
                                                        fit_params=fit_params,
                                                        cv=cv, scoring=scoring,
                                                        n_jobs=-1)
            # 学習データに対するスコアの平均±標準偏差を算出
            train_mean = np.mean(train_scores, axis=1)
            train_std  = np.std(train_scores, axis=1)
            train_center = train_mean
            train_high = train_mean + train_std
            train_low = train_mean - train_std
            # テストデータに対するスコアの平均±標準偏差を算出
            valid_mean = np.mean(valid_scores, axis=1)
            valid_std  = np.std(valid_scores, axis=1)
            valid_center = valid_mean
            valid_high = valid_mean + valid_std
            valid_low = valid_mean - valid_std
            # training_scoresをプロット
            plt.plot(v, train_center, color='blue', marker='o', markersize=5, label='training score')
            plt.fill_between(v, train_high, train_low, alpha=0.15, color='blue')
            # validation_scoresをプロット
            plt.plot(v, valid_center, color='green', linestyle='--', marker='o', markersize=5, label='validation score')
            plt.fill_between(v, valid_high, valid_low, alpha=0.15, color='green')
            # スケールをparam_scalesに合わせて変更
            plt.xscale(param_scales[k])
            # 軸ラベルおよび凡例の指定
            plt.xlabel(k)  # パラメータ名を横軸ラベルに
            plt.ylabel(scoring)  # スコア名を縦軸ラベルに
            plt.legend(loc='lower right')  # 凡例
            # グラフを描画
            import os
            os.makedirs(
                f"output/98__HPT/0827/LightGBM", exist_ok=True
            )
            plt.savefig(
                f"output/98__HPT/0827/LightGBM/{k}.png"
            )
            plt.close()
            print(f"{k} done.")

    def f04__HPT_XGBoost(self):
        from xgboost import XGBClassifier

        cv_params = {
            'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],
            'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0],
            'reg_alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10],
            'reg_lambda': [0.0001, 0.001, 0.01,  0.1, 1.0, 10],
            'learning_rate': [0.0001, 0.001, 0.01, 0.1],
            'min_child_weight': [1, 3, 5, 7, 9, 11, 13, 15],
            'max_depth': [2, 4, 6, 8, 10, 12],
            'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0]}

        # cv_params = {"eval_metric": ["auc", "aucpr", "logloss", "error"]}
        param_scales = {"eval_metric": "linear"}
        X, y = self.dataset["train"].drop("final_status", axis="columns").copy().values, self.dataset["train"]["final_status"].values

        # 乱数シード
        seed = 42
        # モデル作成
        model = XGBClassifier(
            **self.params_XGBoost,
            scale_pos_weight = 2
            )  # チューニング前のモデル
        # 学習時fitパラメータ指定
        fit_params = {'verbose': -1,  # 学習中のコマンドライン出力
                    # 'eval_set': [(X, y)]  # early_stopping_roundsの評価指標算出用データ
                    }
        # クロスバリデーションして決定境界を可視化
        seed = 42  # 乱数シード
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)  # KFoldでクロスバリデーション分割指定
        scoring = "f1"

        # 検証曲線のプロット（パラメータ毎にプロット）
        for i, (k, v) in enumerate(cv_params.items()):
            train_scores, valid_scores = validation_curve(estimator=model,
                                                        X=X, y=y,
                                                        param_name=k,
                                                        param_range=v,
                                                        # fit_params=fit_params,
                                                        cv=cv, scoring=scoring,
                                                        n_jobs=-1)
            # 学習データに対するスコアの平均±標準偏差を算出
            train_mean = np.mean(train_scores, axis=1)
            train_std  = np.std(train_scores, axis=1)
            train_center = train_mean
            train_high = train_mean + train_std
            train_low = train_mean - train_std
            # テストデータに対するスコアの平均±標準偏差を算出
            valid_mean = np.mean(valid_scores, axis=1)
            valid_std  = np.std(valid_scores, axis=1)
            valid_center = valid_mean
            valid_high = valid_mean + valid_std
            valid_low = valid_mean - valid_std
            # training_scoresをプロット
            plt.plot(v, train_center, color='blue', marker='o', markersize=5, label='training score')
            plt.fill_between(v, train_high, train_low, alpha=0.15, color='blue')
            # validation_scoresをプロット
            plt.plot(v, valid_center, color='green', linestyle='--', marker='o', markersize=5, label='validation score')
            plt.fill_between(v, valid_high, valid_low, alpha=0.15, color='green')
            # スケールをparam_scalesに合わせて変更
            plt.xscale(param_scales[k])
            # 軸ラベルおよび凡例の指定
            plt.xlabel(k)  # パラメータ名を横軸ラベルに
            plt.ylabel(scoring)  # スコア名を縦軸ラベルに
            plt.legend(loc='lower right')  # 凡例
            plt.xticks(
                [i for i in range(len(v))], v
            )
            # グラフを描画
            import os
            os.makedirs(
                f"output/98__HPT/0827/XGBoost", exist_ok=True
            )
            plt.savefig(
                f"output/98__HPT/0827/XGBoost/{k}.png"
            )
            plt.close()
            print(f"{k} done.")

    def f05__HPT_MLP(self):
        self.params_MLP = {
            'random_state': 42,
            'max_iter': 300,
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate_init': 0.01,
            'alpha': 0.1,
        }
        cv_params = {
            'mlp__hidden_layer_sizes': [
                (16), (32), (64), (128), (16, 16), (16, 16, 16), (32, 32), (32, 32, 32)
                # (16), (32),
            ],
        }
        param_scales = {
            'mlp__hidden_layer_sizes': 'linear',
        }

        X, _, y, _ = self.f02__split_data()

        scoring = "f1"
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.params_MLP["random_state"])

        from sklearn.neural_network import MLPClassifier

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(**self.params_MLP))
        ])

        for i, (k, v) in enumerate(cv_params.items()):
            train_scores, valid_scores = validation_curve(estimator=model,
                                                        X=X, y=y,
                                                        param_name=k,
                                                        param_range=v,
                                                        # fit_params=fit_params,
                                                        cv=cv, scoring=scoring,
                                                        n_jobs=1)
            print(train_scores, valid_scores)
            # 学習データに対するスコアの平均±標準偏差を算出
            train_mean = np.mean(train_scores, axis=1)
            train_std  = np.std(train_scores, axis=1)
            train_center = train_mean
            train_high = train_mean + train_std
            train_low = train_mean - train_std
            # テストデータに対するスコアの平均±標準偏差を算出
            valid_mean = np.mean(valid_scores, axis=1)
            valid_std  = np.std(valid_scores, axis=1)
            valid_center = valid_mean
            valid_high = valid_mean + valid_std
            valid_low = valid_mean - valid_std
            # training_scoresをプロット
            print(train_center)
            plt.plot(
                [str(v_) for v_ in v],
                train_center, color='blue', marker='o', markersize=5, label='training score')
            plt.fill_between(
                [str(v_) for v_ in v],
                train_high, train_low, alpha=0.15, color='blue')
            # validation_scoresをプロット
            plt.plot(
                [str(v_) for v_ in v],
                valid_center, color='green', linestyle='--', marker='o', markersize=5, label='validation score')
            plt.fill_between(
                [str(v_) for v_ in v],
                valid_high, valid_low, alpha=0.15, color='green')
            # スケールをparam_scalesに合わせて変更
            plt.xscale(param_scales[k])
            # 軸ラベルおよび凡例の指定
            plt.xlabel(k)  # パラメータ名を横軸ラベルに
            if k == "mlp__hidden_layer_sizes":
                plt.xticks(
                    # [i for i in range(len(v))],
                    [str(v_) for v_ in v],
                    [str(v_) for v_ in v]
                )
            plt.ylabel(scoring)  # スコア名を縦軸ラベルに
            plt.legend(loc='lower right')  # 凡例
            # グラフを描画
            import os
            os.makedirs(
                f"output/98__HPT/0820/MLP/tmp", exist_ok=True
            )
            plt.savefig(
                f"output/98__HPT/0820/MLP/tmp/{k}.png"
            )
            plt.close()
            print(f"{k} done.")

    def f06__HPT_CatBoost(self):
        from catboost import CatBoostClassifier
        from sklearn.model_selection import StratifiedKFold, validation_curve
        import numpy as np
        import matplotlib.pyplot as plt
        import os

        # チューニング対象のハイパーパラメータ
        cv_params_cat = {
            'subsample': [0.5, 0.7, 0.9, 1.0],   # CatBoost の sampling_frequency に相当
            'depth': [2, 4, 6, 8, 10],           # 木の深さ
            'l2_leaf_reg': [0.0001, 0.001, 0.01, 0.1, 1.0, 10],  # L2正則化
            'learning_rate': [0.0001, 0.001, 0.01, 0.1],
            'iterations': [100, 300, 500, 700]   # boosting 回数
        }

        # 軸スケール指定（ログスケールが適切なものはlogに設定）
        param_scales_cat = {
            'subsample': "linear",
            'depth': "linear",
            'l2_leaf_reg': "log",
            'learning_rate': "log",
            'iterations': "linear"
        }

        # データ取得
        X, y = self.dataset["train"].drop("final_status", axis="columns").copy().values, \
            self.dataset["train"]["final_status"].values

        # 乱数シード
        seed = 42
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        scoring = "f1"

        # モデル作成（初期値）
        base_model_cat = CatBoostClassifier(
            random_seed=seed,
            verbose=0,
            eval_metric="F1"  # CatBoost内部の評価指標
        )

        # 検証曲線のプロット
        for i, (k, v) in enumerate(cv_params_cat.items()):
            print(f"Processing {k}...")
            train_scores, valid_scores = validation_curve(
                estimator=base_model_cat,
                X=X, y=y,
                param_name=k,
                param_range=v,
                cv=cv, scoring=scoring,
                n_jobs=-1
            )

            # 学習データの平均±標準偏差
            train_mean = np.mean(train_scores, axis=1)
            train_std  = np.std(train_scores, axis=1)
            train_high = train_mean + train_std
            train_low  = train_mean - train_std

            # 検証データの平均±標準偏差
            valid_mean = np.mean(valid_scores, axis=1)
            valid_std  = np.std(valid_scores, axis=1)
            valid_high = valid_mean + valid_std
            valid_low  = valid_mean - valid_std

            # プロット
            plt.plot(v, train_mean, color='blue', marker='o', markersize=5, label='training score')
            plt.fill_between(v, train_high, train_low, alpha=0.15, color='blue')

            plt.plot(v, valid_mean, color='green', linestyle='--', marker='o', markersize=5, label='validation score')
            plt.fill_between(v, valid_high, valid_low, alpha=0.15, color='green')

            # スケール調整
            plt.xscale(param_scales_cat[k])

            # 軸ラベルや凡例
            plt.xlabel(k)
            plt.ylabel(scoring)
            plt.legend(loc='lower right')

            # 保存先ディレクトリ作成
            os.makedirs("output/98__HPT/0827/CatBoost", exist_ok=True)
            plt.savefig(f"output/98__HPT/0827/CatBoost/{k}.png")
            plt.close()

            print(f"{k} done.")



if __name__ == "__main__":
    instance = F01__HPT()
    instance.f01__get_data2()
    # instance.f03__HPT_lightGBM()
    # instance.f04__HPT_XGBoost()
    #instance.f05__HPT_MLP()
    instance.f06__HPT_CatBoost()
