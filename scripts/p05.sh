#!/usr/bin/env bash
# scripts/x03__CrtCVSetting.sh
# ランダムサーチ設定生成パイプライン（整合性チェック付き）

set -euo pipefail

# ===== デフォルト設定 =====
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_SCRIPT="${PROJECT_ROOT}/scripts/x05__CrtCVSetting.py"
RANGE_FILE_DEFAULT="${PROJECT_ROOT}/output/04__ML/02__CV_range.json"
OUTPUT_FILE_DEFAULT="${PROJECT_ROOT}/output/04__ML/01__config_CV.json"
N_SAMPLES_DEFAULT=10
SEED_DEFAULT=42

RANGE_FILE="${RANGE_FILE_DEFAULT}"
OUTPUT_FILE="${OUTPUT_FILE_DEFAULT}"
N_SAMPLES="${N_SAMPLES_DEFAULT}"
SEED="${SEED_DEFAULT}"

# ===== ヘルプ =====
f01__usage() {
  cat <<'USAGE' 1>&2
Usage: scripts/x03__CrtCVSetting.sh [options]

Options:
  -r, --range-file PATH     検索レンジJSONのパス (default: output/04__ML/02__CV_range.json)
  -o, --output PATH         生成するランダムサーチ設定JSON (default: output/04__ML/03__randsearch_settings.json)
  -n, --n-samples INT       サンプル数（各反復でモデルごとに1件ずつ） (default: 10)
  -s, --seed INT            乱数シード (default: 42)
  -h, --help                このヘルプを表示

例:
  scripts/x03__CrtCVSetting.sh -n 20 -s 123
USAGE
}

# ===== 引数処理 =====
f02__parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -r|--range-file) RANGE_FILE="$2"; shift 2 ;;
      -o|--output) OUTPUT_FILE="$2"; shift 2 ;;
      -n|--n-samples) N_SAMPLES="$2"; shift 2 ;;
      -s|--seed) SEED="$2"; shift 2 ;;
      -h|--help) f01__usage; exit 0 ;;
      *) echo "Unknown option: $1" 1>&2; f01__usage; exit 1 ;;
    esac
  done
}

# ===== 依存チェック =====
f03__check_deps() {
  command -v python3 >/dev/null 2>&1 || { echo "python3 が見つかりません" 1>&2; exit 1; }
  [[ -f "${PY_SCRIPT}" ]] || { echo "必要なスクリプトが見つかりません: ${PY_SCRIPT}" 1>&2; exit 1; }
}

# ===== JSONファイルの存在/テンプレ生成 =====
f04__ensure_range_file() {
  local path="${RANGE_FILE}"
  if [[ ! -f "${path}" ]]; then
    echo "レンジファイルが見つからないためテンプレートを作成します: ${path}"
    mkdir -p "$(dirname "${path}")"
    cat > "${path}" <<'JSON'
{
  "LightGBM": {
    "n_estimators": [200, 3000],
    "learning_rate": {"min": 0.001, "max": 0.3, "log": true, "type": "float", "decimals": 5},
    "num_leaves": [15, 255]
  },
  "XGBoost": {
    "n_estimators": [200, 3000],
    "learning_rate": {"min": 0.001, "max": 0.3, "log": true, "type": "float", "decimals": 5},
    "max_depth": [3, 12]
  }
}
JSON
  fi
}

# ===== レンジJSONの整合性検証 =====
f05__validate_range_json() {
  python3 - "$RANGE_FILE" <<'PY'
import json, sys
p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, dict):
    raise SystemExit("range json は dict である必要があります")

allowed_models = {"LightGBM", "XGBoost"}
models_present = set(data.keys()) & allowed_models
if not models_present:
    raise SystemExit("LightGBM / XGBoost のいずれかのキーが必要です")

def check_param_block(block):
    if not isinstance(block, dict):
        raise SystemExit("パラメータブロックは dict である必要があります")
    for k, v in block.items():
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise SystemExit(f"{k}: [min,max] 形式が必要です")
        elif isinstance(v, dict):
            if "min" not in v or "max" not in v:
                raise SystemExit(f"{k}: dict には 'min' と 'max' が必要です")
        else:
            raise SystemExit(f"{k}: 値は [min,max] か dict である必要があります")

for m in models_present:
    check_param_block(data[m])

print("OK: range json validation passed")
PY
}

# ===== 生成実行 =====
f06__run_generator() {
  mkdir -p "$(dirname "${OUTPUT_FILE}")"
  python3 "${PY_SCRIPT}" \
    --range-file "${RANGE_FILE}" \
    --output "${OUTPUT_FILE}" \
    --n-samples "${N_SAMPLES}" \
    --seed "${SEED}"
  echo "生成完了: ${OUTPUT_FILE}"
}

# ===== メイン =====
f07__main() {
  f02__parse_args "$@"
  f03__check_deps
  f04__ensure_range_file
  f05__validate_range_json
  f06__run_generator
}

f07__main "$@"
