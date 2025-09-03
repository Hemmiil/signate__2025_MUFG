import pandas as pd
import numpy as np
import os
import sys
import json
import datetime
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score

# Import models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from x04_01__NN import F01__MLP, F02__TabM
from x04__02__ClusteringClassifier import F01__ClusteringClassifier

class F01__MLModel:
    """
    A class for ML models based on scripts/README.md.
    """
    def __init__(self, is_validation: bool = False, is_test: bool = False, handle_imbalance: bool = False, model_name: str = "LightGBM", hyperparameters: dict = None, experiment_id: int = 0, timestamp: str=None):
        self.is_validation = is_validation
        self.is_test = is_test
        self.handle_imbalance = handle_imbalance
        self.dataset = {
            "X_train": None,
            "y_train": None,
            "X_test": None,
            "y_test": None
        }
        self.models = []
        self.test_predictions = []
        self.oof_preds = None
        self.oof_preds_proba = None
        self.test_preds_proba = None
        self.model_name = model_name
        self.hyperparameters = hyperparameters if hyperparameters is not None else {}
        self.timestamp = timestamp if timestamp != None else self._generate_timestamp()
        self.experiment_id = experiment_id

    def _generate_timestamp(self) -> str:
        """Generates a timestamp in mmdd_hhMM format."""
        return datetime.datetime.now().strftime("%m%d_%H%M")

    def _load_experiments_from_config(self):
        config_path = "output/04__ML/01__config_tmp.json"
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get("experiments", []) # Default to empty list
        except FileNotFoundError:
            print(f"Config file not found at {config_path}. No experiments to run.")
            return []
        except json.JSONDecodeError:
            print(f"Error decoding config file at {config_path}. No experiments to run.")
            return []

    def _merge_text_label(self, df_train, df_test):
        if self.is_test:
            train_path = "output/04__EDA/0814__text_clustering/trial_002/train_test.csv"
            test_path = "output/04__EDA/0814__text_clustering/trial_002/test_test.csv"

        else:
            train_path = "output/04__EDA/0814__text_clustering/trial_002/train.csv"
            test_path = "output/04__EDA/0814__text_clustering/trial_002/test.csv"
        df_train_label = pd.read_csv(train_path, index_col=0)
        df_test_label = pd.read_csv(test_path, index_col=0)
        df_train = pd.concat([df_train, df_train_label], axis="columns")
        df_test = pd.concat([df_test, df_test_label], axis="columns")

        return df_train, df_test


    def f01__get_data(self, pathes=None):
        """Loads the preprocessed data."""
        """8/13編集：テキストベクトルを追加。メディアの違い等が反映されていると思われる"""
        """8/15編集：テキストベクトルをクラスタリングしたラベルを追加。カラム名は`f99__text_label__{label_id}`。編集箇所はline 92"""

        if pathes == None:
            # train_path = "output/02__preprocessed/train_test.csv" if self.is_test else "output/02__preprocessed/train.csv"
            train_path = "output/05__01__SentenceTransformer_Raw/train_test.csv" if self.is_test else "output/05__01__SentenceTransformer_Raw/train.csv"
            # test_path = "output/02__preprocessed/test_test.csv" if self.is_test else "output/02__preprocessed/test.csv"
            test_path = "output/05__01__SentenceTransformer_Raw/test_test.csv" if self.is_test else "output/05__01__SentenceTransformer_Raw/test.csv"
        else:
            train_path, test_path = pathes
        try:
            train_df = pd.read_csv(train_path)
            print(f"Loaded train data from {train_path}")
            self.dataset["X_train"] = train_df.drop(columns=['final_status'])
            self.dataset["y_train"] = train_df['final_status']

            # Get feature columns from training data to ensure consistency
            feature_cols = self.dataset["X_train"].columns.tolist()

            if not self.is_validation:
                test_df = pd.read_csv(test_path)
                print(f"Loaded test data from {test_path}")
                # Reindex test data to match training data feature columns and order
                self.dataset["X_test"] = test_df.reindex(columns=feature_cols, fill_value=0)


            # self.dataset["X_train"], self.dataset["X_test"] = self._merge_text_label(self.dataset["X_train"], self.dataset["X_test"])

        except FileNotFoundError as e:
            print(f"Error loading data: {e}. Please ensure preprocessing has been run.")
            sys.exit(1)
        print("Data loaded successfully.")

    def f02__split_data(self):
        """Splits data for validation or prepares for submission."""
        if self.is_validation:
            print("Splitting training data for validation...")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, test_idx in skf.split(self.dataset["X_train"], self.dataset["y_train"]):
                self.dataset["X_train"], self.dataset["X_test"] = self.dataset["X_train"].iloc[train_idx], self.dataset["X_train"].iloc[test_idx]
                self.dataset["y_train"], self.dataset["y_test"] = self.dataset["y_train"].iloc[train_idx], self.dataset["y_train"].iloc[test_idx]
                break
            print("Data split for validation.")
        else:
            print("Using full training data and separate test data for submission.")

    def f03__fit_predict(self):
        """Fits models and makes predictions."""
        print(f"Training models and making predictions using {self.model_name}...")
        # kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(self.dataset["X_train"]))
        oof_preds_proba = np.zeros((len(self.dataset["X_train"]), 2))
        test_preds_proba = np.zeros((len(self.dataset["X_test"]), 2))

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset["X_train"], self.dataset["y_train"])):
            print(f"--- Fold {fold+1}/5 ---")
            X_train_fold, X_val_fold = self.dataset["X_train"].iloc[train_idx], self.dataset["X_train"].iloc[val_idx]
            y_train_fold, y_val_fold = self.dataset["y_train"].iloc[train_idx], self.dataset["y_train"].iloc[val_idx]

            if self.model_name == "LightGBM":
                model = LGBMClassifier(random_state=42, is_unbalance=True, **self.hyperparameters)
            elif self.model_name == "XGBoost":
                model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **self.hyperparameters)
            elif self.model_name == "MLP":
                model = F01__MLP(hyperparameters=self.hyperparameters, handle_imbalance=self.handle_imbalance)
            elif self.model_name == "RFC":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    **self.hyperparameters,
                    class_weight={0: 1, 1: 2}
                    )
            elif self.model_name == "TabM":
                model = F02__TabM(
                    n_num_features = len(self.dataset["X_train"].columns),
                    hyperparameters=self.hyperparameters,
                    )
            elif self.model_name == "CluCla":
                model = F01__ClusteringClassifier(hyperparameters=self.hyperparameters)

            if self.model_name == "MLP":
                model.fit(X_train_fold, y_train_fold, X_val=X_val_fold, y_val=y_val_fold)
            else:
                model.fit(X_train_fold, y_train_fold)
            self.models.append(model)

            oof_preds[val_idx] = model.predict(X_val_fold)
            oof_preds_proba[val_idx] = model.predict_proba(X_val_fold)
            test_preds_proba += model.predict_proba(self.dataset["X_test"]) / kf.n_splits

        self.oof_preds = oof_preds
        self.oof_preds_proba = oof_preds_proba
        self.test_preds_proba = test_preds_proba
        self.test_predictions = (test_preds_proba[:, 1] >= 0.5).astype(int)
        print("Models trained and predictions made.")

    def f04__save_y_pred(self):
        """Saves predictions and calculates F1 score if in validation mode."""
        output_dir = "output/04__ML"
        os.makedirs(output_dir, exist_ok=True)

        if self.is_validation:
            print("Calculating F1 score for validation...")
            val_preds_proba = np.zeros((len(self.dataset["X_test"]), 2))
            for model in self.models:
                val_preds_proba += model.predict_proba(self.dataset["X_test"]) / len(self.models)
            val_predictions = (val_preds_proba[:, 1] >= 0.5).astype(int)

            f1 = f1_score(self.dataset["y_test"], val_predictions)
            print(f"Validation F1 Score: {f1:.4f}")

            # Save validation predictions
            validation_output_dir = os.path.join(output_dir, "02__validation", self.timestamp, f"exp__{str(self.experiment_id).zfill(3)}")
            os.makedirs(validation_output_dir, exist_ok=True)
            prediction_filepath = os.path.join(validation_output_dir, "01__y_pred.npy")
            np.save(prediction_filepath, val_predictions)
            print(f"Validation predictions saved to {prediction_filepath}")

            # Save configs.json for validation
            configs = {
                "model_name": self.model_name,
                "hyperparameters": self.hyperparameters,
                "is_validation": self.is_validation,
                "is_test": self.is_test,
                "handle_imbalance": self.handle_imbalance,
                "f1_score": f1
            }
            configs_filepath = os.path.join(validation_output_dir, "configs.json")
            with open(configs_filepath, 'w') as f:
                json.dump(configs, f, indent=2)
            print(f"Validation configs saved to {configs_filepath}")

        else:
            # Save submission predictions
            submission_output_dir = os.path.join(output_dir, "03__submission", self.timestamp, f"exp__{str(self.experiment_id).zfill(3)}")
            os.makedirs(submission_output_dir, exist_ok=True)
            prediction_filepath = os.path.join(submission_output_dir, "01__y_pred.npy")
            np.save(prediction_filepath, self.test_predictions)
            print(f"Submission predictions saved to {prediction_filepath}")

            # Save configs.json for submission
            configs = {
                "model_name": self.model_name,
                "hyperparameters": self.hyperparameters,
                "is_validation": self.is_validation,
                "is_test": self.is_test,
                "handle_imbalance": self.handle_imbalance
            }
            configs_filepath = os.path.join(submission_output_dir, "configs.json")
            with open(configs_filepath, 'w') as f:
                json.dump(configs, f, indent=2)
            print(f"Submission configs saved to {configs_filepath}")




if __name__ == '__main__':
    is_validation = '--validation' in sys.argv
    is_test = '--test' in sys.argv

    # Load experiments from config
    config_path = "output/04__ML/01__config_tmp.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        experiments = config.get("experiments", [])
        handle_imbalance = config.get("handle_imbalance", False)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}. No experiments to run.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error decoding config file at {config_path}. No experiments to run.")
        sys.exit(1)

    print(f"Running ML with validation={is_validation}, test_mode={is_test}, handle_imbalance={handle_imbalance}")

    if not experiments:
        print("No experiments defined in config file. Exiting.")
        sys.exit(0)

    def _generate_timestamp():
        """Generates a timestamp in mmdd_hhMM format."""
        return datetime.datetime.now().strftime("%m%d_%H%M")
    common_timestamp = _generate_timestamp()

    for i, experiment in enumerate(experiments):
        print(f"\n--- Running Experiment {i+1}/{len(experiments)} ---")
        model_name = experiment.get("model_name", "LightGBM")
        hyperparameters = experiment.get("hyperparameters", {})

        print(f"Model: {model_name}, Hyperparameters: {hyperparameters}")

        processor = F01__MLModel(
            is_validation=is_validation,
            is_test=is_test,
            handle_imbalance=handle_imbalance,
            model_name=model_name,
            hyperparameters=hyperparameters,
            experiment_id=i + 1, # Pass the experiment ID,
            timestamp=common_timestamp
        )
        processor.f01__get_data()
        processor.f02__split_data()
        processor.f03__fit_predict()
        processor.f04__save_y_pred()

