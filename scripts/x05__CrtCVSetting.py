#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ランダムサーチ用のハイパーパラメータ辞書リストを生成してJSONに保存するスクリプト。
サーチ範囲は JSON ファイル（既定: output/04__ML/02__CV_range.json）で管理します。

想定レンジJSON例:
{
  "LightGBM": {
    "n_estimators": [200, 3000],
    "learning_rate": {"min": 0.001, "max": 0.3, "log": true, "type": "float", "decimals": 5},
    "num_leaves": [15, 255]
  },
  "XGBoost": {
    "n_estimators": [200, 3000],
    "learning_rate": {"min": 0.001, "max": 0.3, "log": true},
    "max_depth": [3, 12]
  }
}

使い方例:
python scripts/x03__CrtRandSearchSettings.py \
  --output output/04__ML/03__randsearch_settings.json \
  --range-file output/04__ML/02__CV_range.json \
  --n-samples 10 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


# ======================
# 1) 引数
# ======================
def f01__parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create random search settings from range JSON.")
    p.add_argument(
        "--range-file", type=str, default="output/04__ML/02__CV_range.json",
        help="サーチ範囲を定義したJSONファイルのパス"
    )
    p.add_argument(
        "--output", type=str, default="output/04__ML/03__randsearch_settings.json",
        help="生成するランダムサーチ設定の出力先JSONパス"
    )
    p.add_argument("--n-samples", type=int, default=10, help="各試行で生成する設定の個数（モデルごとに1つずつ）")
    p.add_argument("--seed", type=int, default=42, help="乱数シード")
    return p.parse_args()


# ======================
# 2) 入出力
# ======================
def f02__load_range_json(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"range-file が見つかりません: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def f03__ensure_parent_dir(path_str: str) -> None:
    p = Path(path_str)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


def f04__save_json(obj: Any, path_str: str) -> None:
    f03__ensure_parent_dir(path_str)
    with open(path_str, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ======================
# 3) レンジ正規化・検証
# ======================
RangeSpec = Dict[str, Dict[str, Any]]  # 例: {"LightGBM": {"n_estimators": {...}, ...}, ...}

def f05__normalize_param_spec(v: Union[List[float], Tuple[float, float], Dict[str, Any]]) -> Dict[str, Any]:
    """
    ユーザ指定のレンジ（[min,max] または dict）を正規化して dict を返す。
    返すdictは {"min":..., "max":..., "type": "int"/"float", "log": bool, "decimals": int}
    """
    if isinstance(v, (list, tuple)) and len(v) == 2:
        vmin, vmax = v
        ptype = "int" if (isinstance(vmin, int) and isinstance(vmax, int)) else "float"
        return {"min": vmin, "max": vmax, "type": ptype, "log": False, "decimals": 5}

    if isinstance(v, dict):
        if "min" not in v or "max" not in v:
            raise ValueError(f"param range dict には 'min' と 'max' が必要です: {v}")
        ptype = v.get("type")
        if ptype is None:
            # ヒューリスティックで型推定
            ptype = "int" if (float(v["min"]).is_integer() and float(v["max"]).is_integer()) else "float"
        return {
            "min": v["min"],
            "max": v["max"],
            "type": ptype,
            "log": bool(v.get("log", False)),
            "decimals": int(v.get("decimals", 5)),
        }

    raise TypeError(f"不正なレンジ指定です。list/tuple([min,max]) または dict を期待しました: {v}")


def f06__validate_and_normalize_ranges(raw_ranges: Dict[str, Any]) -> RangeSpec:
    """
    モデル名ごとのレンジ定義を正規化する。
    未知のトップレベルキーは無視（LightGBM / XGBoost のみ使用）。
    """
    normalized: RangeSpec = {}
    for model_name in ["LightGBM", "XGBoost"]:
        if model_name not in raw_ranges:
            # レンジが未定義ならスキップ（そのモデルを生成しない）
            continue
        params = {}
        for k, v in raw_ranges[model_name].items():
            spec = f05__normalize_param_spec(v)
            if spec["min"] > spec["max"]:
                raise ValueError(f"{model_name}.{k}: min > max は不正です: {spec}")
            params[k] = spec
        if params:
            normalized[model_name] = params
    if not normalized:
        raise ValueError("有効なモデルのレンジが見つかりません（LightGBM / XGBoost）")
    return normalized


# ======================
# 4) 乱数サンプリング
# ======================
def f07__sample_from_spec(spec: Dict[str, Any], rng: random.Random) -> Any:
    """
    spec から1サンプルを生成。
    int: 一様整数
    float(log=False): 一様連続
    float(log=True): 対数一様
    """
    vmin, vmax = float(spec["min"]), float(spec["max"])
    ptype = spec.get("type", "float")
    is_log = bool(spec.get("log", False))
    decimals = int(spec.get("decimals", 5))

    if ptype == "int":
        return rng.randint(int(math.floor(vmin)), int(math.ceil(vmax)))
    else:
        if is_log:
            if vmin <= 0.0:
                raise ValueError(f"log スケールには min>0 が必要です: {spec}")
            x = rng.uniform(math.log(vmin), math.log(vmax))
            return float(round(math.exp(x), decimals))
        else:
            return float(round(rng.uniform(vmin, vmax), decimals))


def f08__sample_model_params(model_ranges: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """
    1モデル分のパラメータ辞書を生成。
    model_ranges: {"param": spec, ...}
    """
    out = {}
    for param_name, spec in model_ranges.items():
        out[param_name] = f07__sample_from_spec(spec, rng)
    return out


# ======================
# 5) 実験リスト構築
# ======================
def f09__build_experiments(n_samples: int, ranges: RangeSpec, rng: random.Random) -> Dict[str, List[Dict[str, Any]]]:
    """
    experiments: List[{"model_name": ..., "hyperparameters": {...}}] を生成。
    各サンプルごとに、定義されている各モデルのパラメータを1つずつ追加。
    """
    experiments: List[Dict[str, Any]] = []
    for _ in range(n_samples):
        if "LightGBM" in ranges:
            experiments.append({
                "model_name": "LightGBM",
                "hyperparameters": f08__sample_model_params(ranges["LightGBM"], rng)
            })
        if "XGBoost" in ranges:
            experiments.append({
                "model_name": "XGBoost",
                "hyperparameters": f08__sample_model_params(ranges["XGBoost"], rng)
            })
    return {"experiments": experiments}


# ======================
# 6) メイン
# ======================
def f10__main() -> None:
    args = f01__parse_args()
    rng = random.Random(args.seed)

    raw_ranges = f02__load_range_json(args.range_file)
    ranges = f06__validate_and_normalize_ranges(raw_ranges)
    config = f09__build_experiments(args.n_samples, ranges, rng)
    f04__save_json(config, args.output)


if __name__ == "__main__":
    f10__main()
