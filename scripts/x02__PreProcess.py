import pandas as pd
import numpy as np
import json
import os
import sys
from itertools import combinations

class F02__PreProcess:
    def __init__(self, test_mode=False):
        self.X = None
        self.CategoricalLabels = {}
        self.year_dummies_cols = [] # Added to store year dummy column names
        self.rates = None
        self.nrows = 100 if test_mode else None
        self.test_mode = test_mode
        self.train_processed_columns = [] # Added to store the final column order of processed training data

    def f01__get_data(self, is_train=True):
        """Loads the train or test dataset."""
        path = "data/train.csv" if is_train else "data/test.csv"
        self.X = pd.read_csv(path, nrows=self.nrows)
        with open("output/01__CurrencyRate.json", 'r') as f:
            self.rates = json.load(f)

        rows = len(self.X)
        print(f"Data loaded from {path} successfully (read {rows} rows).")

    def f02__01_ConvCurrency(self):
        """Converts the 'goal' column to USD."""
        if self.X is None or self.rates is None:
            print("Data not loaded. Please run f01__get_data() first.")
            return

        def convert(row):
            return row['goal'] * self.rates.get(row['currency'], 1.0)

        self.X['f01__goal'] = self.X.apply(convert, axis=1)
        self.X['f01__goal'] = self.X['f01__goal'].apply(np.log1p)
        self.X = self.X.drop(columns=['goal'])
        print("Currency conversion completed.")

    def f02__02_TimeDiff(self):
        """Calculates the difference between time-related features."""
        if self.X is None:
            print("Data not loaded. Please run f01__get_data() first.")
            return

        time_cols = ["deadline", "state_changed_at", "created_at", "launched_at"]
        for col in time_cols:
            self.X[col] = self.X[col].astype(int)

        # f04__AD_ratio
        # 特徴量案: launched_at - created_at / deadline - created_at
        self.X["f01__ADratio_a"] = self.X["launched_at"] - self.X["created_at"]
        self.X["f01__ADratio_b"] = self.X["deadline"] - self.X["created_at"]
        self.X["f01__ADratio"] = self.X["f01__ADratio_a"] / self.X["f01__ADratio_b"]
        self.X = self.X.drop(
            ["f01__ADratio_a", "f01__ADratio_b"], axis="columns"
        )

        diff_col_names = []
        for col1, col2 in combinations(time_cols, 2):
            diff_col_name = f"f02__diff__{col1.replace('_at','').replace('line','')}__and__{col2.replace('_at','').replace('line','')}"
            self.X[diff_col_name] = (self.X[col1] - self.X[col2])
            diff_col_names.append(diff_col_name)

        # 積特徴量の作成
        diff_col_names.append("f04__datetime")
        for i in range(len(diff_col_names)-1):
            for j in range(i+1, len(diff_col_names)):
                col1, col2 = diff_col_names[i], diff_col_names[j]
                product_col_name = f"f02__product__{col1.replace('f02__diff__', '')}_X_{col2.replace('f02__diff__', '')}"
                self.X[product_col_name] = (self.X[col1] * self.X[col2])

        # goalとの積特徴量の作成
        for col in diff_col_names:
            product_col_name = f"f02__product__goal_X_{col.replace('f02__', '')}"
            self.X[product_col_name] = (self.X["f01__goal"] * self.X[col])

        self.X = self.X.drop(columns=time_cols)
        print("Time difference features created.")

    def f02__04_Categoricals(self, is_train):
        """Handles categorical variables."""
        if self.X is None:
            print("Data not loaded. Please run f01__get_data() first.")
            return

        cat_cols = ['country', 'currency']
        for col in cat_cols:
            if is_train:
                self.CategoricalLabels[col] = self.X[col].unique().tolist()
                dummies = pd.get_dummies(self.X[col], prefix=f'f03__{col}_')
                self.X = pd.concat([self.X, dummies], axis=1)
            else:
                # Apply categories learned from training data
                self.X[col] = self.X[col].apply(
                    lambda x: x if x in self.CategoricalLabels[col] else 'other'
                    )
                dummies = pd.get_dummies(self.X[col], prefix=f'f03__{col}_')
                # Reindex to ensure all columns from training are present, fill missing with 0
                train_cols = [f'f03__{col}__{label}' for label in self.CategoricalLabels[col]]
                dummies = dummies.reindex(columns=train_cols, fill_value=0)

                # If 'other' category was created and needs to be dropped
                if f'f03__{col}__other' in dummies.columns:
                    dummies = dummies.drop(columns=[f'f03__{col}__other'])
                self.X = pd.concat([self.X, dummies], axis=1)

        self.X = self.X.drop(columns=cat_cols)
        print("Categorical features created.")

    def f02__03_Datetime(self, is_train):
        """Processes datetime features from launched_at."""
        if self.X is None:
            print("Data not loaded. Please run f01__get_data() first.")
            return

        # Convert to datetime objects
        self.X['launched_at'] = pd.to_datetime(self.X['launched_at'], unit='s')

        # f04__after_Jul2014
        self.X['f04__after_Jul2014'] = (self.X['launched_at'] >= pd.to_datetime('2014-07-01')).astype(int)

        # f04__Year_{年数}
        launched_year = self.X['launched_at'].dt.year
        dummies_year = pd.get_dummies(launched_year, prefix='f04__Year')

        if is_train:
            self.year_dummies_cols = dummies_year.columns.tolist()
        else:
            dummies_year = dummies_year.reindex(columns=self.year_dummies_cols, fill_value=0)

        self.X = pd.concat([self.X, dummies_year], axis=1)

        # f04__datetime (Unix timestamp)
        self.X['f04__datetime'] = self.X['launched_at'].astype(int) // 10**9 # Convert to Unix timestamp

        self.X = self.X.drop(columns=['launched_at'])
        print("Datetime features created.")

    def f02__03_Datetime_ver2(self, is_train):
        """Processes datetime features from launched_at."""
        if self.X is None:
            print("Data not loaded. Please run f01__get_data() first.")
            return

        # Convert to datetime objects
        self.X['launched_at'] = pd.to_datetime(self.X['launched_at'], unit='s')

        # f04__after_Jul2014
        self.X['f04__after_Jul2014'] = (self.X['launched_at'] >= pd.to_datetime('2014-07-01')).astype(int)

        # f04__YearQuarter_{YYYYQ#}
        launched_year = self.X['launched_at'].dt.year
        launched_quarter = self.X['launched_at'].dt.quarter
        launched_year_quarter = launched_year.astype(str) + "Q" + launched_quarter.astype(str)

        dummies_year_quarter = pd.get_dummies(launched_year_quarter, prefix='f04__YearQuarter')

        if is_train:
            self.year_quarter_dummies_cols = dummies_year_quarter.columns.tolist()
        else:
            dummies_year_quarter = dummies_year_quarter.reindex(columns=self.year_quarter_dummies_cols, fill_value=0)

        self.X = pd.concat([self.X, dummies_year_quarter], axis=1)

        # f04__datetime (Unix timestamp)
        self.X['f04__datetime'] = self.X['launched_at'].astype(int) // 10**9  # Convert to Unix timestamp

        # Drop original datetime column
        # self.X = self.X.drop(columns=['launched_at'])
        print("Datetime features created (YearQuarter only).")


    def f02__05_DisableCommunication(self):
        """Processes disable_communication feature."""
        if self.X is None:
            print("Data not loaded. Please run f01__get_data() first.")
            return

        self.X['f05__disable_communication'] = self.X['disable_communication']
        self.X = self.X.drop(columns=['disable_communication'])
        print("Disable communication feature processed.")

    def f02__07_ConcurrentProjects(self, is_train):
        from x02__01__SweepLine import main_train, main_test
        if is_train:
            self.X = main_train(self.X)
            self.X_07 = self.X[["deadline", "created_at"]].copy()

        else:
            return main_test(
                self.X_07, self.X
            )


    def f03__save_data(self, is_train):
        """Saves the preprocessed data."""
        if self.X is None:
            print("No data to save. Please run preprocessing steps first.")
            return

        output_dir = "output/02__preprocessed_tmp"
        os.makedirs(output_dir, exist_ok=True)

        suffix = "_test" if self.test_mode else ""
        filename = f"train{suffix}.csv" if is_train else f"test{suffix}.csv"

        if is_train:
            # For training data, determine the final column order and save it
            self.train_processed_columns = [col for col in self.X.columns if col.startswith('f')]
            if 'state' in self.X.columns:
                self.train_processed_columns.append('state')
            self.X[self.train_processed_columns].to_csv(os.path.join(output_dir, filename), index=False)
            print(f"Saved training data with columns: {self.train_processed_columns}")
        else:
            # For test data, reindex to match training data feature columns and order
            # Get feature columns from training data (excluding 'state')
            train_feature_cols = [col for col in self.train_processed_columns if col != 'state']

            # Select and reindex test data to match these feature columns
            # Ensure all columns from train_feature_cols are present, fill missing with 0
            final_df = self.X.reindex(columns=train_feature_cols, fill_value=0)
            final_df.to_csv(os.path.join(output_dir, filename), index=False)
            print(f"Saved test data with columns: {final_df.columns.tolist()}")

        if self.test_mode:
            print(f"Test data saved to {os.path.join(output_dir, filename)}")
        else:
            print(f"Preprocessed data saved to {os.path.join(output_dir, filename)}")


if __name__ == '__main__':
    test_mode = '--test' in sys.argv
    if test_mode:
        print("--- Running in test mode ---")

    preprocessor = F02__PreProcess(test_mode=test_mode)

    # Process training data
    preprocessor.f01__get_data(is_train=True)
    preprocessor.f02__07_ConcurrentProjects(is_train=True)
    preprocessor.f02__01_ConvCurrency()
    preprocessor.f02__03_Datetime_ver2(is_train=True)
    preprocessor.f02__02_TimeDiff()
    preprocessor.f02__04_Categoricals(is_train=True)
    preprocessor.f02__05_DisableCommunication()
    preprocessor.f03__save_data(is_train=True)

    # Process test data
    preprocessor.f01__get_data(is_train=False)
    preprocessor.f02__07_ConcurrentProjects(is_train=True)
    preprocessor.f02__01_ConvCurrency()
    preprocessor.f02__03_Datetime_ver2(is_train=False)
    preprocessor.f02__02_TimeDiff()
    preprocessor.f02__04_Categoricals(is_train=False)
    preprocessor.f02__05_DisableCommunication()
    preprocessor.f03__save_data(is_train=False)