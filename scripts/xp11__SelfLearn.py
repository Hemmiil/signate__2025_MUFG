import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

# 定数
target_col = "final_status"
NEG_POS_RATIO = 4
threshold_pos = 0.8
threshold_neg = 0.1
PATIENCE = 1
NUM_STEPS = 10
RANDOM_STATE = 42
alpha = 1.1

class F01_SelfLearn():
    def __init__(self):
        # 早期停止・追加停止用カウンタはインスタンス変数へ
        self.score_old = 0.0
        self.cnt = 0
        self.cnt_len_data_add = 0
        self.data_add = pd.DataFrame()

        # ログ用
        self.logs = {
            "meta": {
                "NEG_POS_RATIO": NEG_POS_RATIO,
                "threshold_pos": threshold_pos,
                "threshold_neg": threshold_neg,
                "PATIENCE": PATIENCE,
                "NUM_STEPS": NUM_STEPS,
                "RANDOM_STATE": RANDOM_STATE,
            },
            "base_models": [],
            "steps": []
        }
        self._base_models_logged = False  # 一度だけベースモデル設定を記録するフラグ

        # ログファイルのパス作成
        self.out_dir = "output/11__PseudoLabeling2"
        self.logs_dir = os.path.join(self.out_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)  # ディレクトリ作成（exist_ok=True）
        ts = datetime.now().strftime("%m%d_%H%M")
        self.log_path = os.path.join(self.logs_dir, f"{ts}.json")

    # ヘルパ
    def split_X_y(self, data: pd.DataFrame):
        return data.drop(columns=[target_col]), data[target_col]

    def f01__get_data(self):
        self.data_cv = pd.read_csv("output/05__01__SentenceTransformer_Raw/train.csv")
        self.data_test = pd.read_csv("output/05__01__SentenceTransformer_Raw/test.csv")

    def f02__voting_model(self, pos_scale_weight=2):
        from sklearn.ensemble import VotingClassifier
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier

        # ベースモデル定義
        lgbm_gbdt = LGBMClassifier(
            n_estimators=200,
            num_leaves=10,
            boosting_type="gbdt",
            min_child_samples=90,
            reg_alpha=10,
            data_sample_strategy="bagging",
            random_state=RANDOM_STATE,
            verbosity=-1,
            is_unbalance=True
        )
        lgbm_dart = LGBMClassifier(
            n_estimators=200,
            num_leaves=32,
            boosting_type="dart",
            reg_alpha=10,
            reg_lambda=0,
            max_depth=10,
            drop_seed=RANDOM_STATE,
            random_state=RANDOM_STATE,
            is_unbalance=True,
            verbosity=-1,
        )
        xgb = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=2,
            subsample=1.0,
            scale_pos_weight=pos_scale_weight,
            reg_lambda=10,
            objective="binary:logistic",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0
        )

        estimators = [
            ("LightGBM_GBDT", lgbm_gbdt),
            ("LightGBM_DART", lgbm_dart),
            ("XGBoost", xgb),
        ]

        # ベースモデル設定を一度だけログ化
        if not self._base_models_logged:
            self.logs["base_models"] = [
                {"model_id": name, "params": model.get_params()}
                for name, model in estimators
            ]
            self._base_models_logged = True

        return VotingClassifier(estimators=estimators, voting="soft")

    def f03__main(self):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score

        data_cv = self.data_cv.copy()
        data_test = self.data_test.copy()

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        is_added = pd.Series(False, index=data_test.index)  # ラベルベースで管理
        len_data_add_old = 0

        for step in range(NUM_STEPS):
            # 予測を蓄積
            y_cv_pred_proba = np.zeros(len(data_cv), dtype=float)
            y_test_pred_proba = np.zeros((len(data_test), cv.get_n_splits()), dtype=float)

            X_cv, y_cv = self.split_X_y(data_cv)

            # CV ループ
            for fold_id, (idx_train, idx_val) in enumerate(cv.split(X_cv, y_cv)):
                data_train = data_cv.iloc[idx_train]
                data_val   = data_cv.iloc[idx_val]

                # 学習データ = 元学習 + これまでの擬似ラベル付与データ
                data_train_with_pseudo = pd.concat(
                    [data_train, self.data_add], axis="index"
                )

                X_train, y_train = self.split_X_y(data_train_with_pseudo)

                # クラス不均衡に応じて XGBoost の scale_pos_weight を更新
                pos_weight = float((len(y_train) - int(y_train.sum())) / max(int(y_train.sum()), 1))
                model = self.f02__voting_model(pos_scale_weight=pos_weight)
                model.fit(X_train, y_train)

                # 検証予測
                X_val, y_val = self.split_X_y(data_val)
                y_cv_pred_proba[idx_val] = model.predict_proba(X_val)[:, 1]

                # テスト予測（test には目的変数が無い前提）
                X_test = data_test.drop(columns=[target_col], errors="ignore")
                y_test_pred_proba[:, fold_id] = model.predict_proba(X_test)[:, 1]

            # CV 評価（閾値 0.5）
            y_cv_pred = (y_cv_pred_proba > 0.5).astype(int)
            score_new = f1_score(y_cv, y_cv_pred)

            # 早期停止判定
            if score_new <= self.score_old:
                self.cnt += 1
            else:
                self.cnt = 0
            print(f"step {step} score: {score_new:.5f} cnt: {self.cnt}")

            # ===== 擬似ラベル選別 =====
            proba_mean = y_test_pred_proba.mean(axis=1)

            # まだ未追加で、閾値を超えるものを抽出
            remain_mask = ~is_added
            pos_mask = (proba_mean > (threshold_pos * (alpha ** step))) & remain_mask
            neg_mask = (proba_mean < threshold_neg) & remain_mask

            data_add_pos = data_test.loc[pos_mask].copy()
            data_add_neg = data_test.loc[neg_mask].copy()

            # 件数調整（ネガは多すぎる場合に信頼度が高い方を優先）
            if len(data_add_pos) > 0 and len(data_add_neg) > NEG_POS_RATIO * len(data_add_pos):
                neg_scores = proba_mean[neg_mask]  # 長さ = len(data_add_neg)
                keep_k = NEG_POS_RATIO * len(data_add_pos)
                keep_idx = np.argsort(neg_scores)[:keep_k]  # 0 に近い順
                data_add_neg = data_add_neg.iloc[keep_idx]

            # ===== ラベル付与して束ねる =====
            pos_added = int(len(data_add_pos))
            neg_added = int(len(data_add_neg))

            if pos_added > 0:
                data_add_pos[target_col] = 1
            if neg_added > 0:
                data_add_neg[target_col] = 0

            newly_added = pd.concat([data_add_pos, data_add_neg], axis="index")
            if len(newly_added) > 0:
                # ここで self.data_add に追記
                self.data_add = pd.concat([self.data_add, newly_added], axis="index")
                # 既に追加済みとしてフラグを立てる（ラベルベースで安全に）
                is_added.loc[newly_added.index] = True

            cum_added = int(len(self.data_add))
            print(f"add data num (cumulative): {cum_added}")

            # ログ：ステップごとの情報を記録
            self.logs["steps"].append({
                "step": int(step),
                "cv_f1": float(score_new),
                "added_pos": pos_added,
                "added_neg": neg_added,
                "added_cumulative": cum_added,
                "threshold_pos_used": float(threshold_pos * (1.01 ** step)),
                "threshold_neg_used": float(threshold_neg)
            })

            # 追加が止まったら終了（2 回連続で増えない）
            if len_data_add_old == len(self.data_add):
                self.cnt_len_data_add += 1
            else:
                self.cnt_len_data_add = 0

            # 早期停止：改善なし
            if self.cnt >= PATIENCE:
                print("--- Break (no improvement) ---")
                self._flush_logs()
                break

            # 早期停止：追加が増えない
            if self.cnt_len_data_add >= 2:
                print("--- Break (no more additions) ---")
                self._flush_logs()
                break
            else:
                len_data_add_old = len(self.data_add)
                self.score_old = score_new  # ここで更新

        # ループ完了時もログを保存
        self._flush_logs()

    def _flush_logs(self):
        # JSON でログ書き出し
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)
        print(f"Log saved: {self.log_path}")

    def f04__save(self):
        os.makedirs(self.out_dir, exist_ok=True)
        out_path = os.path.join(self.out_dir, "train_add.csv")
        self.data_add.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    learner = F01_SelfLearn()
    learner.f01__get_data()   # データ読込
    learner.f03__main()       # 自己学習ループ実行（ログも保存）
    learner.f04__save()       # 擬似ラベル付与データを保存
