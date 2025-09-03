import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from datetime import datetime
import sys

def f01__visualize_goal_from_file(file_path):
    """
    Visualizes the distribution of the 'f01__goal' column from a given CSV file.
    """
    # --- 定義 ---
    SCRIPT_PATH = "scripts/x01_visualize.py"
    OUTPUT_DIR_NAME = "03__PreprocessedGoalDistribution"
    OUTPUT_PATH = os.path.join("output", OUTPUT_DIR_NAME)

    if not os.path.exists(file_path):
        print(f"Error: Input file not found at {file_path}")
        sys.exit(1)

    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    # --- 出力ディレクトリを作成 ---
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # --- データの読み込み ---
    df = pd.read_csv(file_path, usecols=['f01__goal'])

    # --- グラフの描画と保存 (通常スケール) ---
    plt.figure(figsize=(12, 6))
    sns.histplot(df['f01__goal'], bins=50, kde=True)
    plt.title(f'Distribution of Preprocessed Goal ({base_filename})')
    plt.xlabel('Goal (USD)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(OUTPUT_PATH, f'{base_filename}_dist.png'))
    plt.close()

    # --- グラフの描画と保存 (対数スケール) ---
    plt.figure(figsize=(12, 6))
    sns.histplot(np.log1p(df['f01__goal']), bins=50, kde=True)
    plt.title(f'Distribution of Log-Transformed Goal ({base_filename})')
    plt.xlabel('log(f01__goal + 1)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(OUTPUT_PATH, f'{base_filename}_log_dist.png'))
    plt.close()

    # --- description.json の作成 ---
    description = {
        "date": datetime.now().strftime("%Y/%m/%d-%H:%M"),
        "script": SCRIPT_PATH,
        "input_file": file_path,
        "memo": f"Visualization for {base_filename}."
    }
    desc_filename = f'description_{base_filename}.json'
    with open(os.path.join(OUTPUT_PATH, desc_filename), 'w') as f:
        json.dump(description, f, indent=4)

    print(f"Visualization for {file_path} complete. Graphs are saved in '{OUTPUT_PATH}'.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python x01_visualize.py <path_to_input_csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    f01__visualize_goal_from_file(input_file)