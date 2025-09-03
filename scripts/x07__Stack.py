import pandas as pd
import numpy as np
import os
import sys
import json
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from x04__ML import F01__MLModel
from xgboost import XGBClassifier

global_data_pathes = {
    "is_test": [
        "output/05__01__SentenceTransformer_Raw/train_test.csv",
        "output/05__01__SentenceTransformer_Raw/test_test.csv"
        ],
    "non_test": [
        "output/05__01__SentenceTransformer_Raw/train.csv",
        "output/05__01__SentenceTransformer_Raw/test.csv"
        ],
}

class F01__StackModel:
    """
    MLモデルをスタッキングするためのクラス。
    """
    def __init__(self, is_validation: bool = False, is_test: bool = False, meta_model_name: str = "LogisticRegression", meta_hyperparameters: dict = None, handle_imbalance: bool = False):
        self.is_validation = is_validation
        self.is_test = is_test
        self.meta_model_name = meta_model_name
        self.meta_hyperparameters = meta_hyperparameters if meta_hyperparameters is not None else {}
        self.handle_imbalance = handle_imbalance
        self.base_model_oof_preds = None
        self.base_model_test_preds = None
        self.meta_model = None
        self.base_models_config = None
        self.common_timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

        self.num__base_meta_validation_split = 10
        self.num__base = 5
        self.num__meta = 4
        self.num__validation = 1

        if is_test:
            self.data_pathes = global_data_pathes["is_test"]
        else:
            self.data_pathes = global_data_pathes["non_test"]

    def f01__train_base_models_and_get_predictions(self):
        """ベースモデルを訓練し、予測を取得する。"""
        print("ベースモデルの訓練と予測を開始..." )

        config_path = "output/07__Stack/01__configs_CV.json"
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            base_models_config = config.get("base_models", [])
            handle_imbalance_base = config.get("handle_imbalance", False)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"スタッキング設定ファイル '{config_path}' の読み込みエラー: {e}")
            sys.exit(1)

        if not base_models_config:
            print("設定ファイルにベースモデルが定義されていません。")
            sys.exit(1)

        oof_preds_list = []
        test_preds_list = []

        for i, model_config in enumerate(base_models_config):
            print(f"--- ベースモデル {i+1}/{len(base_models_config)} を訓練中 ---")
            model_name = model_config.get("model_name")
            hyperparameters = model_config.get("hyperparameters", {})

            print(f"モデル: {model_name}, ハイパーパラメータ: {hyperparameters}")

            # x04__ML.pyのF01__MLModelを使用してベースモデルを訓練
            base_model_trainer = F01__MLModel(
                is_validation=self.is_validation,
                is_test=self.is_test,
                handle_imbalance=handle_imbalance_base,
                model_name=model_name,
                hyperparameters=hyperparameters,
                experiment_id=i + 1
            )
            base_model_trainer.f01__get_data(
                pathes=self.data_pathes
                )
            base_model_trainer.f02__split_data()
            base_model_trainer.f03__fit_predict()

            # 予測を取得
            oof_preds_proba = base_model_trainer.oof_preds_proba
            test_preds_proba = base_model_trainer.test_preds_proba

            if oof_preds_proba is None or test_preds_proba is None:
                print(f"モデル {model_name} の予測を取得できませんでした。")
                continue

            # カラム名を f99__stack_model_{i}__cls_{j} の形式にする
            oof_preds_df = pd.DataFrame(
                oof_preds_proba[:, 1],
                columns=[f"f99__stack_model_{i+1}__pos"]
                )
            test_preds_df = pd.DataFrame(
                test_preds_proba[:, 1],
                columns=[f"f99__stack_model_{i+1}__pos"]
                )

            oof_preds_list.append(oof_preds_df)
            test_preds_list.append(test_preds_df)

        if not oof_preds_list:
            print("有効なベースモデルの予測がありません。")
            sys.exit(1)

        self.base_model_oof_preds = pd.concat(oof_preds_list, axis=1)
        if self.is_validation:
            # 検証モードではテスト予測はOOFから作られるため、OOFと同じものを使う
            self.base_model_test_preds = pd.concat(oof_preds_list, axis=1)
        else:
            self.base_model_test_preds = pd.concat(test_preds_list, axis=1)

        # 元の特徴量を保持
        self.original_X_train = base_model_trainer.dataset["X_train"]
        self.original_X_test = base_model_trainer.dataset["X_test"]

        print("すべてのベースモデルの訓練と予測が完了しました。")


    def f02__prepare_meta_features(self, y_train_original):
        """分割戦略に基づいてメタモデルの特徴量を準備する。"""
        print("メタ特徴量を準備中..." )

        # 元の特徴量とベースモデルの予測を結合
        meta_features_oof = pd.concat([self.original_X_train.reset_index(drop=True), self.base_model_oof_preds.reset_index(drop=True)], axis=1)
        if not self.is_validation:
            meta_features_test = pd.concat([self.original_X_test.reset_index(drop=True), self.base_model_test_preds.reset_index(drop=True)], axis=1)

        if self.is_validation:
            skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            meta_train_indices = []
            meta_val_indices = []

            for i, (train_idx, val_idx) in enumerate(skf_meta.split(meta_features_oof, y_train_original)):
                if i < 3:
                    meta_train_indices.extend(train_idx)
                else:
                    meta_val_indices.extend(val_idx)

            meta_train_indices = sorted(list(set(meta_train_indices)))
            meta_val_indices = sorted(list(set(meta_val_indices)))

            self.meta_X_train = meta_features_oof.iloc[meta_train_indices]
            self.meta_y_train = y_train_original.iloc[meta_train_indices]
            self.meta_X_test_for_f1 = meta_features_oof.iloc[meta_val_indices]
            self.meta_y_test_for_f1 = y_train_original.iloc[meta_val_indices]
            self.meta_X_test = None

        else:
            self.meta_X_train = meta_features_oof
            self.meta_y_train = y_train_original
            self.meta_X_test = meta_features_test
            self.meta_X_test_for_f1 = None
            self.meta_y_test_for_f1 = None

        print("メタ特徴量が準備されました。" )

    def f03__fit_predict_meta_model(self):
        """メタモデルを適合させ、予測を行う。"""
        print(f"メタモデル ({self.meta_model_name}) を訓練し、予測を作成中..." )

        meta_hyperparams = self.meta_hyperparameters.copy()
        if self.handle_imbalance:
            if self.meta_model_name == "LogisticRegression":
                meta_hyperparams['class_weight'] = 'balanced'
            elif self.meta_model_name == "LightGBM":
                meta_hyperparams['is_unbalance'] = True
            elif self.meta_model_name == "XGBoost":
                meta_hyperparams['scale_pos_weight'] = sum(self.meta_y_train==0)/sum(self.meta_y_train==1)

        if self.meta_model_name == "LogisticRegression":
            self.meta_model = LogisticRegression(random_state=42)
        elif self.meta_model_name == "LightGBM":
            from lightgbm import LGBMClassifier
            self.meta_model = LGBMClassifier(random_state=42, **meta_hyperparams)
        elif self.meta_model_name == "XGBoost":
            self.meta_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **meta_hyperparams)
        else:
            print(f"サポートされていないメタモデル: {self.meta_model_name}" )
            sys.exit(1)

        self.meta_model.fit(self.meta_X_train, self.meta_y_train)
        feat__imp = {
            k: v for k, v in zip(
                self.meta_model.feature_names_in_,
                self.meta_model.feature_importances_.astype(float)
            )
        }

        if self.is_validation:
            final_predictions = self.meta_model.predict(self.meta_X_test_for_f1)
            f1 = f1_score(self.meta_y_test_for_f1, final_predictions)
            print(f"メタモデルの検証F1スコア: {f1:.4f}" )
            return final_predictions, feat__imp
        else:
            final_predictions = self.meta_model.predict(self.meta_X_test)
            print("最終予測が作成されました。" )
            return final_predictions, feat__imp

    def f04__save_final_predictions(self, final_predictions: np.ndarray, feat__imp):
        """最終予測とメタモデルの設定を保存する。"""
        output_dir = "output/07__Stacking"
        os.makedirs(output_dir, exist_ok=True)

        if self.is_validation:
            validation_output_dir = os.path.join(output_dir, "02__validation", self.common_timestamp)
            os.makedirs(validation_output_dir, exist_ok=True)

            prediction_filepath = os.path.join(validation_output_dir, "01__y_pred.npy")
            np.save(prediction_filepath, final_predictions)
            print(f"検証予測が {prediction_filepath} に保存されました。" )


            config_path = "output/07__Stack/01__configs_CV.json"
            with open(config_path, "r") as f:
                config_exp = json.load(f)

            configs = {
                "HP": config_exp,
                "is_validation": self.is_validation,
                "is_test": self.is_test,
                "f1_score": f1_score(self.meta_y_test_for_f1, final_predictions),
                "feat__imp": feat__imp
            }

            configs_filepath = os.path.join(validation_output_dir, "configs.json")
            with open(configs_filepath, 'w') as f:
                json.dump(configs, f, indent=2)
            print(f"検証設定が {configs_filepath} に保存されました。" )
        else:
            submission_output_dir = os.path.join(output_dir, "03__submission", self.common_timestamp)
            os.makedirs(submission_output_dir, exist_ok=True)

            prediction_filepath = os.path.join(submission_output_dir, "01__y_pred.npy")
            np.save(prediction_filepath, final_predictions)
            print(f"提出予測が {prediction_filepath} に保存されました。" )

            configs = {
                "meta_model_name": self.meta_model_name,
                "meta_hyperparameters": self.meta_hyperparameters,
                "is_validation": self.is_validation,
                "is_test": self.is_test,
                "feat__imp": feat__imp
            }
            configs_filepath = os.path.join(submission_output_dir, "configs.json")
            with open(configs_filepath, 'w') as f:
                json.dump(configs, f, indent=2)
            print(f"提出設定が {configs_filepath} に保存されました。" )


if __name__ == '__main__':
    is_validation = '--validation' in sys.argv
    is_test = '--test' in sys.argv

    print(f"スタッキングを実行中: validation={is_validation}, test_mode={is_test}" )

    # Load the correct dataset based on test mode
    if is_test:
        train_data_path = global_data_pathes["is_test"][0]
    else:
        train_data_path = global_data_pathes["non_test"][0]

    try:
        original_train_df = pd.read_csv(train_data_path)
    except FileNotFoundError:
        print(f"{train_data_path} が見つかりません。続行できません。" )
        sys.exit(1)

    # If in validation mode, we need to get the correct y_train_original that corresponds to the OOF predictions
    if is_validation:
        # This split needs to be consistent with the split in x04__ML.py's f02__split_data
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # We only take the first split, just like in x04__ML.py
        train_idx, _ = next(iter(skf.split(original_train_df.drop(columns=['final_status']), original_train_df['final_status'])))
        y_train_original = original_train_df['final_status'].iloc[train_idx]
    else:
        y_train_original = original_train_df['final_status']


    config_path = "output/07__Stack/01__configs_CV.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        meta_model_config = config.get("meta_model", {})
        handle_imbalance_meta = meta_model_config.get("handle_imbalance", False)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"スタッキング設定ファイル '{config_path}' の読み込みエラー: {e}")
        sys.exit(1)

    meta_model_name = meta_model_config.get("model_name", "LogisticRegression")
    meta_hyperparameters = meta_model_config.get("hyperparameters", {})

    stacker = F01__StackModel(
        is_validation=is_validation,
        is_test=is_test,
        meta_model_name=meta_model_name,
        meta_hyperparameters=meta_hyperparameters,
        handle_imbalance=handle_imbalance_meta
    )

    stacker.f01__train_base_models_and_get_predictions()
    stacker.f02__prepare_meta_features(y_train_original)
    final_predictions, feat__imp = stacker.f03__fit_predict_meta_model()
    stacker.f04__save_final_predictions(final_predictions, feat__imp)

    print("スタッキングプロセスが完了しました。" )