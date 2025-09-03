import numpy as np
import pandas as pd

# =========================
# 同時プロジェクト数（スイープライン/二分探索）
# =========================

def _validate_and_cast(df: pd.DataFrame, start_col: str, end_col: str):
    if start_col not in df.columns or end_col not in df.columns:
        raise KeyError(f"{start_col=} または {end_col=} が df に存在しません。")
    starts = np.asarray(df[start_col].values, dtype=np.int64)
    ends   = np.asarray(df[end_col].values, dtype=np.int64)
    if len(starts) != len(ends):
        raise ValueError("start_col と end_col の長さが一致していません。")
    # もし end < start が混入していたら交換して整える（必要ならコメントアウト）
    swap_idx = ends < starts
    if np.any(swap_idx):
        tmp = starts[swap_idx].copy()
        starts[swap_idx] = ends[swap_idx]
        ends[swap_idx] = tmp
    return starts, ends

def _counts_against_reference(
    targets_start: np.ndarray,
    targets_end:   np.ndarray,
    ref_starts_sorted: np.ndarray,
    ref_ends_sorted:   np.ndarray,
    inclusive: bool = True
) -> np.ndarray:
    """
    target 区間 [start, end] それぞれに対し、reference 区間との重なり本数を返す。
    """
    if inclusive:
        # 閉区間: (ref_start <= target_end) and (ref_end >= target_start)
        num_start_le_end = np.searchsorted(ref_starts_sorted, targets_end, side="right")
        num_end_lt_start = np.searchsorted(ref_ends_sorted,   targets_start, side="left")
    else:
        # 右端開区間: (ref_start < target_end) and (ref_end > target_start)
        num_start_lt_end = np.searchsorted(ref_starts_sorted, targets_end, side="left")
        num_end_le_start = np.searchsorted(ref_ends_sorted,   targets_start, side="right")
        num_start_le_end, num_end_lt_start = num_start_lt_end, num_end_le_start

    return num_start_le_end - num_end_lt_start

def _prepare_reference(ref_df: pd.DataFrame, start_col: str, end_col: str):
    ref_starts, ref_ends = _validate_and_cast(ref_df, start_col, end_col)
    return np.sort(ref_starts), np.sort(ref_ends)

def add_concurrency_to_train(
    data_train: pd.DataFrame,
    start_col: str = "created_at",
    end_col: str   = "deadline",
    out_col: str   = "concurrent_projects",
    inclusive: bool = True,
) -> pd.DataFrame:
    """
    学習データに、自身を除いた「学習データ内での同時プロジェクト数」を付与する。
    data_test は一切参照しない（データリーク防止）。
    """
    starts, ends = _validate_and_cast(data_train, start_col, end_col)
    ref_starts_sorted, ref_ends_sorted = _prepare_reference(data_train, start_col, end_col)

    counts_vs_train = _counts_against_reference(
        starts, ends, ref_starts_sorted, ref_ends_sorted, inclusive=inclusive
    )

    # 自己自身の区間が 1 本含まれるため除外
    data_train_out = data_train.copy()
    data_train_out[out_col] = (counts_vs_train - 1).astype(np.int64)
    return data_train_out

def add_concurrency_to_test_against_train(
    data_test: pd.DataFrame,
    data_train: pd.DataFrame,
    start_col: str = "created_at",
    end_col: str   = "deadline",
    out_col: str   = "concurrent_projects_against_train",
    inclusive: bool = True,
) -> pd.DataFrame:
    """
    テストデータに「学習データと重なる本数」を付与する。
    学習データのみを参照し、テスト同士の情報は使わない（データリーク防止）。
    """
    test_starts, test_ends = _validate_and_cast(data_test, start_col, end_col)
    ref_starts_sorted, ref_ends_sorted = _prepare_reference(data_train, start_col, end_col)

    counts_vs_train = _counts_against_reference(
        test_starts, test_ends, ref_starts_sorted, ref_ends_sorted, inclusive=inclusive
    )

    data_test_out = data_test.copy()
    data_test_out[out_col] = counts_vs_train.astype(np.int64)
    return data_test_out

# =========================
# 使い方例
# =========================
def main_train(data_train):
    # 学習内での同時プロジェクト数（自身は除く）
    train_with_conc = add_concurrency_to_train(
        data_train, start_col="created_at", end_col="deadline",
        out_col="f07__concurrent_projects", inclusive=True
    )
    return train_with_conc

def main_test(data_train, data_test):
    # テスト ↔ 学習 の重なり数（テスト同士は参照しない）
    test_with_conc = add_concurrency_to_test_against_train(
        data_test, data_train,
        start_col="created_at", end_col="deadline",
        out_col="f07__concurrent_projects", inclusive=True
    )
    return test_with_conc