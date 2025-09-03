import os
import numpy as np
import pandas as pd
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

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


class _SimpleMLP(nn.Module):
    """
    多層MLP。hidden_dimは必ずlist[int]。
    例: hidden_dim=[128, 64, 32] -> in -> 128 -> 64 -> 32 -> 1
    """
    def __init__(self, in_features: int, hidden_dim: List[int], dropout: float = 0.1):
        super().__init__()
        if not isinstance(hidden_dim, list) or len(hidden_dim) == 0:
            print(hidden_dim)
            raise ValueError("hidden_dim は1つ以上のintを含む list[int] を指定してください。")
        layers = []
        input_dim = in_features
        for h in hidden_dim:
            if not isinstance(h, int):
                raise ValueError("hidden_dim の各要素は int である必要があります。")
            layers += [nn.Linear(input_dim, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _ResMLP(nn.Module):
    """
    Residual付きMLP。全隠れ層スタックの出力に対して、入力から最終隠れ次元へ射影したショートカットを加算。
    hidden_dim は list[int]（最後の要素が残差加算の基準次元になる）。
    """
    def __init__(self, in_features: int, hidden_dim: List[int], dropout: float = 0.1):
        super().__init__()
        if not isinstance(hidden_dim, list) or len(hidden_dim) == 0:
            raise ValueError("hidden_dim は1つ以上のintを含む list[int] を指定してください。")
        self.hidden_dim = hidden_dim

        blocks = []
        input_dim = in_features
        for h in hidden_dim:
            if not isinstance(h, int):
                raise ValueError("hidden_dim の各要素は int である必要があります。")
            blocks += [nn.Linear(input_dim, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                blocks.append(nn.Dropout(p=dropout))
            input_dim = h
        self.blocks = nn.Sequential(*blocks)

        last_h = hidden_dim[-1]
        self.shortcut = nn.Identity() if in_features == last_h else nn.Linear(in_features, last_h)
        self.post_act = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(last_h, 1)

    def forward(self, x):
        residual = self.shortcut(x)
        h = self.blocks(x)
        h = self.post_act(h + residual)
        out = self.fc_out(h)
        return out.squeeze(-1)


class _MLPWithBN(nn.Module):
    """
    BatchNorm入りMLP。各隠れ層 Linear -> BN -> ReLU -> Dropout。
    hidden_dim は list[int]。
    """
    def __init__(self, in_features: int, hidden_dim: List[int], dropout: float = 0.1):
        super().__init__()
        if not isinstance(hidden_dim, list) or len(hidden_dim) == 0:
            raise ValueError("hidden_dim は1つ以上のintを含む list[int] を指定してください。")

        layers = []
        input_dim = in_features
        for h in hidden_dim:
            if not isinstance(h, int):
                raise ValueError("hidden_dim の各要素は int である必要があります。")
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

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =========================
#   High-level Classifier
# =========================
class F01__NNClassifier:
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
        self.ModelArch: str = str(hp.pop("ModelArch", "simple")).lower()
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

        self.is_conv_log: bool = bool(hp.pop("is_conv_log", False))
        self.is_conv_emb: bool = bool(hp.pop("is_conv_emb", False))


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

    # ---------- ヘルパ ----------
    @staticmethod
    def _to_numpy(x):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            return x.values
        return np.asarray(x)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        ds = TensorDataset(X_tensor, y_tensor)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def _build_model(self, in_features: int):
        arch_map = {
            "simple": _SimpleMLP,
            "res": _ResMLP,
            "bn": _MLPWithBN,
            "emb": _MLPWithEmbedding,
        }
        if self.ModelArch not in arch_map:
            raise ValueError(f"ModelArch は 'simple' | 'res' | 'bn' | 'emb' |のいずれかを指定してください。受け取った値: {self.ModelArch}")

        ModelCls = arch_map[self.ModelArch]
        self.model = ModelCls(
            in_features=in_features,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _epoch(self, loader: DataLoader, train: bool = True) -> float:
        self.model.train(mode=train)
        total_loss, n = 0.0, 0
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=self.pin_memory)
            yb = yb.to(self.device, non_blocking=self.pin_memory)

            with torch.set_grad_enabled(train):
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

            bs = yb.size(0)
            total_loss += loss.item() * bs
            n += bs
        return total_loss / max(n, 1)

    def _split__cont_cat(self, X: pd.DataFrame, col_flag: str = "f04__YearQuarter", drop_dummy: bool = True) -> pd.DataFrame:
        """
        ダミー列 `{col_flag}_...` から元のカテゴリ値を復元して、
        X[col_flag] に書き戻す。必要ならダミー列を削除。
        """
        import re
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

        # ダミー列を消したい場合
        if drop_dummy:
            X = X.drop(columns=cols, axis="columns")

        ## チェック:
        print("Continuous Columns")
        print(X.head())
        print("Categorical Columns")
        print(categories.head())

        return X, categories
    def _data_convs(
            self, X, is_log_conv = True, is_emb = True, is_train = False
    ):
        # 対数変換
        if is_log_conv:
            cols_pos = X.columns[X.min() > 0].tolist()
            non_target_log = ["f01__goal", "f02__product__goal_X_f04__datetime"]
            for col in non_target_log:
                cols_pos.remove(col)
            X.loc[:, cols_pos] = X.loc[:, cols_pos].astype(float)
            X[cols_pos] = X[cols_pos].apply(np.log)

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
        X: ndarray | DataFrame（数値想定）
        y: ndarray | Series（0/1）
        """

        X = self._data_convs(X, is_emb=self.is_conv_emb, is_log_conv=self.is_conv_log, is_train=True)
        X = self._to_numpy(X).astype(np.float32)
        y = self._to_numpy(y).astype(np.float32).reshape(-1)

        # 常にアンダーサンプリングを実施
        X, y = self.sampler.fit_resample(X, y)

        # 標準化
        X_scaled = self.scaler.fit_transform(X)

        # モデル構築
        in_features = X_scaled.shape[1]
        self._build_model(in_features)

        # DataLoader
        train_loader = self._make_loader(X_scaled, y, shuffle=True)

        # 検証セット（任意）
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val = self._to_numpy(X_val).astype(np.float32)
            y_val = self._to_numpy(y_val).astype(np.float32).reshape(-1)
            X_val_scaled = self.scaler.transform(X_val)
            val_loader = self._make_loader(X_val_scaled, y_val, shuffle=False)

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
                                # optimizerの復元は任意
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
        X = self._data_convs(X, is_emb=self.is_conv_emb, is_log_conv=self.is_conv_log, is_train=False)
        X = self._to_numpy(X).astype(np.float32)
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        logits = self.model(X_tensor)
        probs1 = torch.sigmoid(logits).cpu().numpy()
        probs0 = 1.0 - probs1
        return np.stack([probs0, probs1], axis=1)

    @torch.no_grad()
    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
