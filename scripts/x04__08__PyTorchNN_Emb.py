import os
import numpy as np
import pandas as pd
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

from umap import UMAP

# 並列やTokenizerの警告を抑制（必要に応じて調整）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")


# =========================
#   Model Architectures
# =========================
class _MLPWithEmbedding(nn.Module):
    """
    数値特徴量 + カテゴリ特徴量（Embedding変換）の両方を扱うMLP。
    cat_dims: 各カテゴリ変数のユニーク値数（例: [10, 5, 20]）
    emb_dims: 各Embedding次元数（例: [4, 3, 6]）
              通常は min(50, ceil(n_cat**0.5)) くらいでOK。
    """
    def __init__(self, num_cont: int, cat_dims: List[int], emb_dims: List[int],
                 hidden_dim: List[int], dropout: float = 0.1):
        super().__init__()
        if len(cat_dims) != len(emb_dims):
            raise ValueError("cat_dims と emb_dims の長さは一致させてください。")

        # カテゴリEmbedding層
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, emb_dim)
            for cat_dim, emb_dim in zip(cat_dims, emb_dims)
        ])
        emb_total_dim = sum(emb_dims)

        # MLP本体（数値特徴量＋Embeddingを結合した次元を入力とする）
        input_dim = num_cont + emb_total_dim
        layers = []
        for h in hidden_dim:
            layers += [
                nn.Linear(input_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_cont, x_cat):
        """
        x_cont: 数値特徴量 (batch_size, num_cont)
        x_cat:  カテゴリ特徴量 (batch_size, num_cat) -> int64 のカテゴリID
        """
        # 各カテゴリをEmbeddingに通してconcat
        emb = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        x_cat_emb = torch.cat(emb, dim=1)

        # 数値特徴量と結合
        x = torch.cat([x_cont, x_cat_emb], dim=1)
        return self.net(x).squeeze(-1)

class F01__NNClassifier_Emb:
    """
    k近傍法クラスと同一のI/F（fit, predict, predict_proba）を持つPyTorch分類器（バイナリ）。
    - Xの数値カラム抽出は行わない（Xは数値前処理済みを想定）
    - 不均衡対策のランダムアンダーサンプリングは常に実施
    - アーキテクチャは hyperparameters["ModelArch"] で選択: "simple" | "res" | "bn"
    - 隠れ層は hyperparameters["hidden_dim"] に list[int] を必須指定
    """
    def __init__(self, hyperparameters: dict = None):
        hp = hyperparameters.copy() if hyperparameters is not None else {}

        # --- 必須 & 基本 ---
        self.epochs: int = int(hp.pop("epochs", 30))
        self.batch_size: int = int(hp.pop("batch_size", 256))
        self.lr: float = float(hp.pop("lr", 1e-3))
        self.weight_decay: float = float(hp.pop("weight_decay", 0.0))

        # hidden_dim は list[int] 必須
        hidden_dim = hp.pop("hidden_dim", None)
        if not isinstance(hidden_dim, list) or len(hidden_dim) == 0 or not all(isinstance(h, int) for h in hidden_dim):
            raise ValueError("hyperparameters['hidden_dim'] は 1つ以上のintからなる list[int] を指定してください。")
        self.hidden_dim: List[int] = hidden_dim

        self.dropout: float = float(hp.pop("dropout", 0.1))
        self.patience: int = int(hp.pop("patience", 0))  # 0でEarlyStopping無効
        self.num_workers: int = int(hp.pop("num_workers", 0))
        self.pin_memory: bool = bool(hp.pop("pin_memory", False))
        self.device: str = str(hp.pop("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state: int = int(hp.pop("random_state", 42))
        self.verbose: bool = bool(hp.pop("verbose", False))

        # 乱数シード
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # 前処理・Loss
        self.scaler = StandardScaler()
        self.criterion = nn.BCEWithLogitsLoss()

        # 不均衡データ対策（常時実施）
        self.sampler = RandomUnderSampler(random_state=self.random_state)

        # モデル関連
        self.model: nn.Module = None
        self.optimizer = None
        self._fitted = False

        # 未使用パラメータ保持（将来拡張用）
        self._unused_hparams = hp

        self.is_conv_log = bool(hp.pop("is_conv_log", False))
        self.is_conv_emb = bool(hp.pop("is_conv_emb", False))

    # ---------- ヘルパ ----------
    @staticmethod
    def _to_numpy(x):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            return x.values
        return np.asarray(x)

    def _make_loader(self, X, y, shuffle: bool = False):
        """
        X:
        - Embedding非対応モデル → ndarray (float32 数値特徴のみ)
        - Embedding対応モデル → tuple (X_cont, X_cat)
        y: ndarray
        """
        if isinstance(X, tuple):
            # 数値とカテゴリのタプル
            X_cont, X_cat = X
            Xc_tensor = torch.tensor(X_cont, dtype=torch.float32)
            Xcat_tensor = torch.tensor(X_cat, dtype=torch.int64)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            dataset = TensorDataset(Xc_tensor, Xcat_tensor, y_tensor)
        else:
            # 数値だけ
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, y_tensor)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _build_model(self, num_cont, cat_dims, emb_dims):

        self.model = _MLPWithEmbedding(
            num_cont = num_cont,
            cat_dims = cat_dims,
            emb_dims = emb_dims,
            hidden_dim = self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _epoch(self, loader, train: bool = True):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        count = 0

        for batch in loader:
            # --- Embeddingあり (x_cont, x_cat, y) ---
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                xb_cont, xb_cat, yb = batch
                xb_cont = xb_cont.to(self.device)
                xb_cat = xb_cat.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb_cont, xb_cat)

            # --- Embeddingなし (x, y) ---
            else:
                xb, yb = batch
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb)

            loss = self.criterion(logits, yb)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * len(yb)
            count += len(yb)

        return total_loss / count

    def _split__cont_cat(
            self, X: pd.DataFrame, col_flag: str = "f04__YearQuarter", drop_dummy: bool = True, is_train: bool = True,
        ):
        """
        ダミー列 `{col_flag}_...` から元のカテゴリ値を復元して、
        X[col_flag] に書き戻す。必要ならダミー列を削除。
        戻り値: (X_cont, X_cat)  いずれも2次元numpy配列
        """
        import re
        X = X.copy()  # SettingWithCopyWarning回避
        prefix = f"{col_flag}_"
        cols = [c for c in X.columns if c.startswith(prefix)]
        if not cols:
            # 復元対象のダミー列が無ければ → 数値部分だけ返してカテゴリは空
            X_cont = self._to_numpy(X).astype(np.float32)
            X_cat = np.zeros((len(X_cont), 0), dtype=np.int64)
            return X_cont, X_cat

        # --- ダミー列からカテゴリ復元 ---
        X_tmp = X.loc[:, cols].apply(pd.to_numeric, errors="coerce")
        X_filled = X_tmp.fillna(-np.inf).to_numpy()
        max_idx = np.argmax(X_filled, axis=1)
        max_val = X_filled[np.arange(X_filled.shape[0]), max_idx]

        chosen_cols = np.array(cols, dtype=object)[max_idx]
        categories = pd.Series(chosen_cols, index=X.index, dtype="object")
        categories[max_val <= 0] = np.nan
        categories = categories.where(categories.isna(),
                                    categories.str.replace(rf"^{re.escape(prefix)}", "", regex=True))

        if is_train:
            self.le = LabelEncoder()
            categories = pd.Series(self.le.fit_transform(categories))
        else:
            categories = pd.Series(self.le.transform(categories))

        # --- DataFrame更新 ---
        if drop_dummy:
            X = X.drop(columns=cols, axis="columns")

        # --- 戻り値を2次元に整形 ---
        X_cont = self._to_numpy(X).astype(np.float32)
        X_cat = categories.to_numpy().astype(np.int64)

        if X_cat.ndim == 1:
            X_cat = X_cat.reshape(-1, 1)  # ← ここで必ず2次元にする

        return X_cont, X_cat

    def _data_convs(
            self, X, is_train = False
    ):
        is_log_conv = self.is_conv_log
        is_emb = self.is_conv_emb
        # 対数変換
        if is_log_conv:
            cols_pos = X.columns[X.min() > 0].tolist()
            non_target_log = ["f01__goal", "f02__product__goal_X_f04__datetime"]
            for col in non_target_log:
                cols_pos.remove(col)
            X.loc[:, cols_pos] = X.loc[:, cols_pos].astype("float")
            X.loc[:, cols_pos] = X.loc[:, cols_pos].apply(np.log)

        # テキストベクトルの圧縮
        if is_emb:
            if is_train:
                self.EmbeddingModel = UMAP(n_components=3)
                cols_sentence = [col for col in X.columns if "SentenceVec" in col]
                X_sentence_emb = pd.DataFrame(
                    self.EmbeddingModel.fit_transform(X[cols_sentence]),
                    columns=[f"f06__SentenceVec_Emb_{i+1}" for i in range(3)],
                    index=X.index,
                )
                X = pd.concat(
                    [X.drop(cols_sentence, axis="columns"), X_sentence_emb], axis="columns"
                )
            else:

                cols_sentence = [col for col in X.columns if "SentenceVec" in col]
                X_sentence_emb = pd.DataFrame(
                    self.EmbeddingModel.fit_transform(X[cols_sentence]),
                    columns=[f"f06__SentenceVec_Emb_{i+1}" for i in range(3)],
                    index=X.index,
                )
                X = pd.concat(
                    [X.drop(cols_sentence, axis="columns"), X_sentence_emb], axis="columns"
                )

        return X

    # ---------- Public API ----------
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X: DataFrame（数値 + カテゴリ）
        y: ndarray | Series（0/1）
        """
        # --- データ分割 ---
        X = self._data_convs(X, is_train=True)
        X_cont, X_cat = self._split__cont_cat(X, is_train=True)
        X_cont = self._to_numpy(X_cont).astype(np.float32)
        X_cat = self._to_numpy(X_cat).astype(np.int64)  # Embeddingに渡すのでint64
        y = self._to_numpy(y).astype(np.float32).reshape(-1)

        # --- アンダーサンプリング ---
        # samplerでインデックスを生成
        idx_resampled = self.sampler.fit_resample(
            np.arange(len(y)).reshape(-1, 1), y
        )[0].ravel()  # 返るのは (indices, y_resampled)

        # そのインデックスだけ抽出
        X_cont = X_cont[idx_resampled]
        X_cat = X_cat[idx_resampled]
        y = y[idx_resampled]

        # 標準化（数値のみ）
        X_cont_scaled = self.scaler.fit_transform(X_cont)

        # モデル構築
        num_cont = X_cont_scaled.shape[1]
        cat_dims = [int(X_cat[:, i].max()) + 1 for i in range(X_cat.shape[1])] if X_cat.shape[1] > 0 else []
        emb_dims = [min(50, int(np.ceil(np.sqrt(c)))) for c in cat_dims]  # Embedding次元の設計例
        self._build_model(num_cont, cat_dims, emb_dims)

        # DataLoader作成
        train_loader = self._make_loader((X_cont_scaled, X_cat), y, shuffle=True)

        # 検証セット
        val_loader = None
        if X_val is not None and y_val is not None:
            Xc_val, Xcat_val = self._split__cont_cat(X_val)
            Xc_val = self._to_numpy(Xc_val).astype(np.float32)
            Xcat_val = self._to_numpy(Xcat_val).astype(np.int64)
            y_val = self._to_numpy(y_val).astype(np.float32).reshape(-1)
            Xc_val_scaled = self.scaler.transform(Xc_val)
            val_loader = self._make_loader((Xc_val_scaled, Xcat_val), y_val, shuffle=False)

        best_val = float("inf")
        patience_cnt = 0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            tr_loss = self._epoch(train_loader, train=True)
            if val_loader is not None:
                va_loss = self._epoch(val_loader, train=False)
                if self.verbose:
                    print(f"[Epoch {epoch:03d}] train_loss={tr_loss:.6f} val_loss={va_loss:.6f}")
                if self.patience > 0:
                    if va_loss < best_val - 1e-8:
                        best_val = va_loss
                        patience_cnt = 0
                        best_state = {
                            "model": {k: v.cpu() for k, v in self.model.state_dict().items()},
                            "optimizer": self.optimizer.state_dict(),
                        }
                    else:
                        patience_cnt += 1
                        if patience_cnt >= self.patience:
                            if self.verbose:
                                print("Early stopping triggered.")
                            if best_state is not None:
                                self.model.load_state_dict(best_state["model"])
                            break
            else:
                if self.verbose:
                    print(f"[Epoch {epoch:03d}] train_loss={tr_loss:.6f}")

        self._fitted = True


    @torch.no_grad()
    def predict_proba(self, X):
        """
        戻り値: shape (n_samples, 2)
        列0: P(class=0), 列1: P(class=1)
        """
        if not self._fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # --- 分割 & 前処理 ---
        X = self._data_convs(X, is_train=True)
        X_cont, X_cat = self._split__cont_cat(X, is_train=False)
        X_cont = self._to_numpy(X_cont).astype(np.float32)
        X_cat = self._to_numpy(X_cat).astype(np.int64)
        X_cont_scaled = self.scaler.transform(X_cont)

        # 推論
        self.model.eval()
        Xc_tensor = torch.tensor(X_cont_scaled, dtype=torch.float32).to(self.device)
        Xcat_tensor = torch.tensor(X_cat, dtype=torch.int64).to(self.device)
        logits = self.model(Xc_tensor, Xcat_tensor)
        probs1 = torch.sigmoid(logits).cpu().numpy()
        probs0 = 1.0 - probs1
        return np.stack([probs0, probs1], axis=1)


    @torch.no_grad()
    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
