import os
# 環境変数で並列を制御（ハング回避）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from catboost import CatBoostClassifier
import re
import numpy as np
import pandas as pd

class F01__CatBoost:
    def __init__(self, hyperparameters: dict = None, handle_imbalance: bool = False):
        self.hyperparameters = hyperparameters if hyperparameters is not None else {}
        self.model = CatBoostClassifier(
            **self.hyperparameters, verbose=0
            )
        self.cat_feats = []

    def inv_get_dummy(self, X: pd.DataFrame, col_flag: str = "f04__YearQuarter", drop_dummy: bool = True) -> pd.DataFrame:
        """
        ダミー列 `{col_flag}_...` から元のカテゴリ値を復元して、
        X[col_flag] に書き戻す。必要ならダミー列を削除。
        """
        X = X.copy()  # SettingWithCopyWarning回避
        prefix = f"{col_flag}_"
        cols = [c for c in X.columns if c.startswith(prefix)]
        if not cols:
            # 復元対象のダミー列が無ければそのまま返す
            return X

        # 数値化（bool/str混在でもOKにする）
        X_tmp = X.loc[:, cols].apply(pd.to_numeric, errors="coerce")
        # NaNは最小扱いにしてargmaxを安定化
        X_filled = X_tmp.fillna(-np.inf).to_numpy()

        # 行毎に最大の列インデックス取得
        max_idx = np.argmax(X_filled, axis=1)
        # そのときの最大値を取り出し、全ゼロ（または負無限）の行はNaNに
        max_val = X_filled[np.arange(X_filled.shape[0]), max_idx]

        # 最大列名 → 元カテゴリ名へ（prefixを除去）
        chosen_cols = np.array(cols, dtype=object)[max_idx]
        categories = pd.Series(chosen_cols, index=X.index, dtype="object")
        categories[max_val <= 0] = np.nan  # 全ゼロの行をNaNに
        categories = categories.where(categories.isna(),
                                    categories.str.replace(rf"^{re.escape(prefix)}", "", regex=True))

        # 復元結果を書き戻し
        X.loc[:, col_flag] = categories.values

        # ダミー列を消したい場合
        if drop_dummy:
            X = X.drop(columns=cols, axis="columns")

        return X

    def fit(self, X, y):
        X_ = self.inv_get_dummy(X=X, col_flag="f04__YearQuarter")
        self.cat_features = X_.select_dtypes(include=["object", "category"]).columns.tolist()
        self.model.fit(X_, y, cat_features=self.cat_features)

    def predict(self, X):
        X_ = self.inv_get_dummy(X=X, col_flag="f04__YearQuarter")
        return self.model.predict(X_)

    def predict_proba(self, X):
        X_ = self.inv_get_dummy(X=X, col_flag="f04__YearQuarter")
        return self.model.predict_proba(X_)