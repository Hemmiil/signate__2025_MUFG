import pandas as pd
import numpy as np
import os
import sys
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

class F01__LGBMClassifier:
    """
    A class for a RandomForestClassifier prototype based on scripts/README.md.
    """
    def __init__(self, is_validation: bool = False, is_test: bool = False, handle_imbalance: bool = False):
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

    def f01__get_data(self):
        """Loads the preprocessed data."""
        train_path = "output/02__preprocessed/train_test.csv" if self.is_test else "output/02__preprocessed/train.csv"
        test_path = "output/02__preprocessed/test_test.csv" if self.is_test else "output/02__preprocessed/test.csv"

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

        except FileNotFoundError as e:
            print(f"Error loading data: {e}. Please ensure preprocessing has been run.")
            sys.exit(1)
        print("Data loaded successfully.")

    def f02__split_data(self):
        """Splits data for validation or prepares for submission."""
        if self.is_validation:
            print("Splitting training data for validation...")
            # For validation, split the loaded train_df into train and test sets
            # Using StratifiedKFold to ensure class distribution is maintained
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            # Take the first fold for simplicity in this prototype
            for train_idx, test_idx in skf.split(self.dataset["X_train"], self.dataset["y_train"]):
                self.dataset["X_train"], self.dataset["X_test"] = self.dataset["X_train"].iloc[train_idx], self.dataset["X_train"].iloc[test_idx]
                self.dataset["y_train"], self.dataset["y_test"] = self.dataset["y_train"].iloc[train_idx], self.dataset["y_train"].iloc[test_idx]
                break # Only take the first split for this prototype
            print("Data split for validation.")
        else:
            print("Using full training data and separate test data for submission.")
            # X_train, y_train are already set in f01__get_data
            # X_test is already set in f01__get_data

    def f03__fit_predict(self):
        """Fits models and makes predictions."""
        print("Training models and making predictions...")
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(self.dataset["X_train"]))
        test_preds_proba = np.zeros((len(self.dataset["X_test"]), 2))

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset["X_train"], self.dataset["y_train"])):
            print(f"--- Fold {fold+1}/5 ---")
            X_train_fold, X_val_fold = self.dataset["X_train"].iloc[train_idx], self.dataset["X_train"].iloc[val_idx]
            y_train_fold, y_val_fold = self.dataset["y_train"].iloc[train_idx], self.dataset["y_train"].iloc[val_idx]

            model = LGBMClassifier(random_state=42, is_unbalance=self.handle_imbalance)
            model.fit(X_train_fold, y_train_fold)
            self.models.append(model)

            oof_preds[val_idx] = model.predict(X_val_fold)
            test_preds_proba += model.predict_proba(self.dataset["X_test"]) / kf.n_splits

        self.test_predictions = (test_preds_proba[:, 1] >= 0.5).astype(int)
        print("Models trained and predictions made.")

    def f04__save_y_pred(self):
        """Saves predictions and calculates F1 score if in validation mode."""
        output_dir = "output/04__ML_prototype"
        os.makedirs(output_dir, exist_ok=True)

        if self.is_validation:
            print("Calculating F1 score for validation...")
            # Need to re-run prediction on the validation set to get the final score
            # This is a simplified approach for the prototype. In a real scenario,
            # you'd collect OOF predictions during training.
            val_preds_proba = np.zeros((len(self.dataset["X_test"]), 2))
            for model in self.models:
                val_preds_proba += model.predict_proba(self.dataset["X_test"]) / len(self.models)
            val_predictions = (val_preds_proba[:, 1] >= 0.5).astype(int)

            f1 = f1_score(self.dataset["y_test"], val_predictions)
            print(f"Validation F1 Score: {f1:.4f}")
        else:
            prediction_filepath = os.path.join(output_dir, "y_pred.npy")
            np.save(prediction_filepath, self.test_predictions)
            print(f"Predictions saved to {prediction_filepath}")


if __name__ == '__main__':
    is_validation = '--validation' in sys.argv
    is_test = '--test' in sys.argv

    handle_imbalance = '--handle_imbalance' in sys.argv

    print(f"Running ML prototype with validation={is_validation}, test_mode={is_test}, handle_imbalance={handle_imbalance}")

    processor = F01__LGBMClassifier(is_validation=is_validation, is_test=is_test, handle_imbalance=handle_imbalance)
    processor.f01__get_data()
    processor.f02__split_data()
    processor.f03__fit_predict()
    processor.f04__save_y_pred()