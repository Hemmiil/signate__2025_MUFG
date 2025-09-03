# x02__PreProcess.py
- F01__PreProcess
  - atribute
    - `X`: pandas.DataFrame
      - `f01__get_data`で読み込まれるデータ。パスは`data/train.csv`と`data/test.csv`
    - `CategoricalLabels`: dict(str: list[str])
      - カテゴリカル変数のエンコーディングの際に、テストデータのリーキングを防ぐための変数。
      - キーは`country`と`currency`で、それぞれにカテゴリカル変数のバリエーションをリストで保持する
      - 先に訓練データのカテゴリカル変数のバリエーションのすべてを`CategoricalLabels`に保存する
      - テストデータについて、`CategoricalLabels`に登録されていないラベルはエンコードされない
  - ### `f02__01_ConvCurrency`
    - input-> X: pandas.DataFrame
    - output-> X: pandas.DataFrame
    - `output/01__CurrencyRate.json` に基づいて、`X`に含まれる`goal`カラムを変換する。変換後のカラム名称は`f01__goal`
    - `goal`を削除し、`f01__goal`を追加した`X`を返す
  - ### `f02__02_TimeDiff`
    - input-> X: pandas.DataFrame
    - output-> X: pandas.DataFrame
    - 以下４つの特徴量について、それぞれ差分を求めて保存する。4種類なので、6通りの差分カラムが保存される
      ```
      ["deadline", "state_changed_at", "created_at", "launched_at"]
      ```
    - 新しいカラム名称は`f02__diff__deadline__state_changed`のように、`f02__diff__{カラム1}__{カラム2}`のようにする
  - ### `f02__03_Categoricals`
    - input-> `X`: pandas.DataFrame, `is_train`: bool
    - output-> `X`: pandas.DataFrame
    - カテゴリカル変数(`country`, `currency`)を処理する
    - ワンホットエンコーディングで処理する
    - `is_train` が`True`である場合、以下の手順でエンコーディングする
      - カラム`column`について、ワンホットエンコーディングを実行
        - 接頭辞は`f03__column__`とする
      - ラベルのすべてのバリエーションを`CategoricalLabels[column]`にリストで保存
      - 以上を`column`=`country`, `currency` の場合で実行する
    - `is_train`が`False`である場合、以下の手順でエンコーディングする
      - カラム`column`について、`CategoricalLabels[column]`に含まれないラベルはすべて`others`
      - `CategoricalLabels`に記録されたラベルでエンコーディングする。`CategoricalLabels`に記録されていないラベルはすべて`other`にする
      - ワンホットエンコーディングを実行
        - 接頭辞は`f03__column__`とする
      - ワンホットエンコーディングの結果、ラベル`others`に対して生成されたカラム（例：`f03__column__other`）は削除する
      - 以上を`column`=`country`, `currency` の場合で実行する
  - ### `f02__04_Datetime`
    - input-> `X`: pandas.DataFrame
    - output-> `X`: pandas.DataFrame
    - `launched_at`を年月に変換して、**2014年7月**以前以後のラベルを作成。新しいカラム名は`f04__after_Jul2014`
    - `launched_at`を年に変換して、ワンホットエンコード。新しいカラム名は`f04__Year_{年数}`
    - `launched_at`のユニックスタイムをそのまま。新しいカラム名は`f04__datetime`
  - ### `f02__05_DisableCommunication`
    - input-> `X`: pandas.DataFrame
    - output-> `X`: pandas.DataFrame
    - カラム`disable_communication`の値を、新規カラム名`f05__disable_communication`として追加、元のカラムは削除
  - ### `f02__06_SentenceTransformer`
    - input-> `X`: pandas.DataFrame
    - output-> `X`: pandas.DataFrame
    - `output/05__SentenceTransformer`に保存されたファイル`train.npy`と`test.npy`を確認して、`X`と統合する
      - 上位10列のみを追加する
      - `is_train`が`True`の場合、`train.npy`を統合する。`is_train`が`False`の場合、`test.npy`を統合する
      - カラム名は、`i`列目を`f06__sentence_{i}`とする
        - ただし`i`は3桁の0埋めを実行する。例：1->001


  - ### `f01__get_data`
    - データを取得する
  - ### `f03__save_data`
    - 前処理を実行したファイルを`output/02__preprocessed/train.csv`, `output/02__preprocessed/test.csv`として保存する
    - 変換が実行されたカラム（`f{通し番号}`という接頭辞のあるカラム）のみを保存する

# x03__CrtSubmission.py
- ## Purpose
  - 提出用データの作成
- F01__CrtSubmission
  - attribute
    - `path`(str): np.array(32439行1列)のデータが格納された`.npy`ファイルのパス
    - `rawdata`(np.array): `path`のファイルをダウンロードしたデータ。`f01__get_data`で取得
    - `submitdata`(pandas.DataFrame): `f02__crt_submission`で整形された提出用データ
    - `memo`(str): ファイル作成時の備考欄。`output/99_Submision/000__memo.json`に保存される
  - f01__get_data
    - `path`のファイルをロードする
    - `rawdata`を更新
  - f02__timestamp
    - return-> timestamp (str)
    - 実行した時刻の`mmdd_hhMM`のフォーマットでタイムスタンプを返す
  - f03__crt_submission
    - `submitdata`を更新する
      - インデックスは`data/sample_submit.csv`
      - インデックスと`rawdata`の長さが一致しなかったらエラーメッセを出力して停止
  - f04__save_submission
    - `submitdata`を`output/99_Submision/{timestamp}__submit.csv`で保存
    - `timestamp`は`f02__timestamp`で作成
  - 実行ファイル：`scripts/p03.sh`
    - input
      - `path`: np.array(32439行1列)のデータが格納された`.npy`ファイルのパス
      - `memo`: ファイル作成時の備考欄。`output/99_Submision/000__memo.json`に保存される
        - `output/99_Submision/000__memo.json` は以下の構成
        ```json
        {
          "0805_1200": "This is a first memo.",
          "0806_1300": "This is a second memo."
        }
        ```
# x04__ML.py
- ## Purpose
  - 簡易的に、現在のデータでどの程度の精度で分類ができるかを検証したい
  - LightGBM, XGBoost の2種類のモデルを選択的に利用できるようにする
- ## Concept
  - データ全体を5分割して、80%を訓練データとして利用するモデルを5通り作成する
  - テストデータ予測は5つのモデルのアンサンブルを計算する。それぞれのモデルの`predict_proba`の値の平均値を計算し、0.5未満ならラベル`0`、0.5以上ならラベル`1`を与える
  - モデルの選択は`output/04__ML/01__config.json`で管理する
- F01__MLModel
  - attribute
    - `is_validation`
      - `data/train.csv`由来のデータのみを利用した検証実験の時は`True`, `data/test.csv`由来のデータも活用し、提出用データを作成する時は`False`
    - `is_test`
      - `output/02__preprocessed/train_test.csv`を利用した、プログラムテストの時は`True`, `output/02__preprocessed/train.csv`を利用した本番環境の時は`False`
    - `handle_imbalance`
      - 目的変数の不均衡性に対応するかどうか。`True`の場合、モデルの`is_unbalance`などのパラメータを調整する
    - `dataset`
      - キーが`X_train`, `y_train`, `X_test`, `y/test`の辞書型
          - `model_name`
        - `output/04__ML/01__config.json`から読み込まれるモデル名（例: `LightGBM`, `XGBoost`）
  - `_load_model_name_from_config`
    - `output/04__ML/01__config.json`からモデル名を読み込む
  - f01__get_data
    - `is_test`が`True`なら`output/02__preprocessed/train_test.csv`を、`False`なら`output/02__preprocessed/train.csv`をロードする。
    - `is_validation`が`True`なら対応するテストデータもダウンロードする
  - f02__split_data
    - `dataset`の作成
    - `is_validation`がTrueなら、訓練データを分割して`dataset`を作成する。`False`ならテストデータを`dataset["X_test"]`, `dataset["y_test"]`に利用する
  - f03__fit_predict
    - モデルの学習と予測を実行する。
    - `model_name`に基づいて適切なモデルを初期化する。
    - `is_validation`が`True`の場合、検証データに対する予測結果を生成する。
    - `is_validation`が`False`の場合、実際のテストデータに対する予測結果を生成する。
  - f04__save_y_pred
    - 予測結果を保存する。
    - `output/04__ML/y_pred.npy`としてNumPy配列形式で保存する。
    - `is_validation`が`True`の場合、以下の形式でデータを保存する
      - `output/04__ML/02__validation/{タイムスタンプ}`に、予測値`01__y_pred.npy`と設定ファイル`configs.json`を保存
    - `is_validation`が`False`の場合、以下の形式でデータを保存する
      - `output/04__ML/03__submission/{タイムスタンプ}`に、予測値`01__y_pred.npy`と設定ファイル`configs.json`を保存

- ## Execution
  - `scripts/p04.sh` を使用して実行する。
  - `scripts/p04.sh` は `is_validation`, `is_test`, `handle_imbalance` のフラグを制御する。
  - **Validation Mode**: `bash scripts/p04.sh --validation`
    - `data/train.csv` を内部で分割し、モデルの性能を評価する。
    - 結果としてF1スコアがコンソールに出力される。
  - **Validation Mode (Test)**: `bash scripts/p04.sh --validation --test`
    - `output/02__preprocessed/train_test.csv` を内部で分割し、モデルの性能を評価する。
    - 結果としてF1スコアがコンソールに出力される。
  - **Submission Mode**: `bash scripts/p04.sh`
    - `data/train.csv` でモデルを学習し、`data/test.csv` に対する予測を生成する。
    - 結果として `output/04__ML/y_pred.npy` が保存される。
  - **Submission Mode (Test)**: `bash scripts/p04.sh --test`
    - `output/02__preprocessed/train_test.csv` でモデルを学習し、`output/02__preprocessed/test_test.csv` に対する予測を生成する。
    - 結果として `output/04__ML/y_pred.npy` が保存される。
# x06__TextVector.py
- ## Purpose
  - テキストデータ（カラム`desc`をLLMを利用してベクトル化して、ランダムフォレストなどの教師あり学習器で扱えるようにする。また特徴量として追加した効果を検証する
  - 主な処理は以下の通り
    - テキストベクトルのクラスタリング
      - 、、KMeans でクラスタリングする場合のクラスタ数の最適化(f02)
  - テキストのベクトル化は、google clobで計算した結果が`output/05__SentenceTransformer`に`trian.npy`と`test.npy`がダウンロードされている
- `F01__TextVector`
  - f01__get_data
  - f02__optimize_num_clusters
    - KMeans でクラスタリングした場合のクラスタ数の最適値を探索する
    - その結果を可視化する
    - エルボー法とシルエット法を実行。グラフを`output/06__TextVector/01__optimize_num_clusters`に保存
  - f03__clustering
    - クラスタリングの実行
    - ハイパーパラメータ`v02__subset_size`
    - 以下を保存
      - クラスタ別テキストテーブル
      - ラベル列
  - f04__add_label
    - 教師あり学習用のテーブルデータの作成
  - f05__supervisor
    - f03, f04 でラベルづけしたデータを、教師ありモデルで学習させて、ラベルづけされていないデータに対してラベルを予測させる
    - ラベルの出力
# x06__TextVector_tmp.py
- ## Purpose
  - テキストデータ（カラム`desc`）をLLMを利用してベクトル化し、ランダムフォレストなどの教師あり学習器で扱えるようにする。また、特徴量として追加した際の効果を検証する。

- ## `F01__TextVector` クラス
  - **属性**
    - `embeddings_train`: `output/05__SentenceTransformer/train.npy` から読み込まれる訓練データ用のテキスト埋め込みベクトル。
    - `embeddings_test`: `output/05__SentenceTransformer/test.npy` から読み込まれるテストデータ用のテキスト埋め込みベクトル。
    - `original_train_df`: `data/train.csv` から読み込まれる元の訓練データDataFrame。
    - `original_test_df`: `data/test.csv` から読み込まれる元のテストデータDataFrame。
    - `cluster_labels_train`: 訓練データに対するクラスタリング結果のラベル。
    - `cluster_labels_test`: テストデータに対するクラスタリング結果のラベル。

  - ### `f01__get_data(self, is_test_mode: bool = False)`
    - **機能**: 必要な生データとテキスト埋め込みデータをロードする。
    - **入力**:
      - `is_test_mode` (bool): `True`の場合、`_test.csv`および`_test.npy`ファイルをロードする。
    - **処理**:
      1. `data/train.csv` および `data/test.csv` を `original_train_df`, `original_test_df` としてそれぞれロードする。
      2. `output/05__SentenceTransformer/train.npy` および `output/05__SentenceTransformer/test.npy` を `embeddings_train`, `embeddings_test` としてそれぞれロードする。
         - `is_test_mode` が `True` の場合、`output/05__SentenceTransformer/train_test.npy` と `output/05__SentenceTransformer/test_test.npy` をロードする。
    - **出力**: なし (クラスの属性にデータを格納)。

  - ### `f02__optimize_num_clusters(self, embeddings: np.ndarray, max_clusters: int = 10)`
    - **機能**: KMeansクラスタリングにおける最適なクラスタ数を探索し、結果を可視化する。
    - **入力**:
      - `embeddings` (np.ndarray): テキスト埋め込みベクトル（例: `embeddings_train`）。
      - `max_clusters` (int): 探索するクラスタ数の最大値。
    - **処理**:
      1. エルボー法 (`KMeans.inertia_`) を用いて、各クラスタ数でのWSS (Within-cluster Sum of Squares) を計算する。
      2. シルエット法 (`silhouette_score`) を用いて、各クラスタ数でのシルエットスコアを計算する。
      3. それぞれの結果をプロットし、`output/06__TextVector/01__optimize_num_clusters/elbow_method.png` および `output/06__TextVector/01__optimize_num_clusters/silhouette_method.png` として保存する。
         - ディレクトリが存在しない場合は作成する。
    - **出力**: なし (グラフをファイルとして保存)。

  - ### `f03__clustering(self, embeddings: np.ndarray, n_clusters: int, subset_size: float = 1.0)`
    - **機能**: 指定されたクラスタ数でKMeansクラスタリングを実行し、クラスタラベルとクラスタ別テキストテーブルを生成する。
    - **入力**:
      - `embeddings` (np.ndarray): テキスト埋め込みベクトル。
      - `n_clusters` (int): クラスタリングに用いるクラスタ数。
      - `subset_size` (float): クラスタリングに使用するデータの割合（0.0から1.0）。`v02__subset_size`に対応。
    - **処理**:
      1. `subset_size`に基づいて、`embeddings`からサブセットをサンプリングする。
      2. サンプリングされたデータに対してKMeansクラスタリングを実行する。
      3. 全てのデータポイントに対してクラスタラベルを予測する。
      4. 各クラスタに属するテキスト（`desc`カラム）を抽出し、クラスタ別テキストテーブルを作成する。
      5. クラスタ別テキストテーブルを `output/06__TextVector/clustered_texts_k{n_clusters}.csv` として保存する。
    - **出力**:
      - `cluster_labels` (np.ndarray): 各データポイントのクラスタラベル。

  - ### `f04__add_label(self, original_df: pd.DataFrame, cluster_labels: np.ndarray, label_col_name: str = 'text_cluster_label')`
    - **機能**: 元のDataFrameにクラスタリング結果のラベルカラムを追加する。
    - **入力**:
      - `original_df` (pd.DataFrame): ラベルを追加する元のDataFrame（例: `original_train_df`）。
      - `cluster_labels` (np.ndarray): `f03__clustering`から得られたクラスタラベル。
      - `label_col_name` (str): 追加するラベルカラムの名称。
    - **処理**:
      1. `original_df`に`label_col_name`として`cluster_labels`を追加する。
    - **出力**:
      - `labeled_df` (pd.DataFrame): ラベルカラムが追加されたDataFrame。

  - ### `f05__supervisor(self, labeled_train_df: pd.DataFrame, unlabeled_test_df: pd.DataFrame, feature_cols: list, target_col: str)`
    - **機能**: クラスタリングで得られたラベルを目的変数として、教師あり学習モデルを訓練し、ラベル付けされていないデータに対して予測を行う。
    - **入力**:
      - `labeled_train_df` (pd.DataFrame): クラスタラベルが追加された訓練データDataFrame。
      - `unlabeled_test_df` (pd.DataFrame): 予測を行う対象のテストデータDataFrame。
      - `feature_cols` (list): モデルの訓練に使用する特徴量カラムのリスト。
      - `target_col` (str): 目的変数となるクラスタラベルのカラム名。
    - **処理**:
      1. `labeled_train_df` を訓練データ (`X_train`, `y_train`) とテストデータ (`X_test`) に分割する。
      2. ランダムフォレストなどの分類モデルを初期化し、訓練データで学習させる。
      3. `unlabeled_test_df` の特徴量を用いてラベルを予測する。
      4. 必要に応じて、モデルの評価指標（例: 精度、F1スコア）を計算し、表示する。
    - **出力**:
      - `predicted_labels` (np.ndarray): `unlabeled_test_df`に対する予測されたクラスタラベル。
# x07__Stack.py
- ## Purpose
  - 複数のベースモデルの予測を組み合わせて、最終的な予測精度を向上させるスタッキングモデルを実装する。

- ## Concept
  - **データ分割（Validation Mode）**:
    - `data/train.csv` を層化K分割交差検証（例: 5分割）し、各フォールドでベースモデルを学習。学習に使用しなかったフォールド（Out-Of-Fold, OOF）に対して予測を生成。これにより、`data/train.csv`全体に対するOOF予測（ベースモデルの訓練データと同じサイズ）が得られる。
    - このOOF予測データと元の`data/train.csv`の目的変数を用いて、メタモデルの訓練と評価を行う。
    - メタモデルの訓練と評価のために、OOF予測データをさらに層化K分割交差検証（例: 5分割）する。この5分割されたデータのうち、
      - 最初の3ブロックを**`train_meta`**（メタモデルの学習範囲）としてメタモデルを訓練する。
      - 残りの2ブロックを**`validation`**（検証範囲）としてメタモデルの予測を行い、この範囲でF1スコアを算出する。
  - **データ分割（Submission Mode）**:
    - **ベースモデル**: `data/train.csv`全体を**`train_base`**として学習し、`data/test.csv`全体に対して予測を生成する。
    - **メタモデル**:
      - **訓練データ**: ベースモデルが`data/train.csv`に対して生成したOOF予測（`data/train.csv`と同じサイズ）を訓練データとする。
      - **予測データ**: ベースモデルが`data/test.csv`に対して生成した予測（`data/test.csv`と同じサイズ）を予測データとする。
  - これらの予測を特徴量として、メタモデル（例: ロジスティック回帰、LightGBM）を訓練し、最終的な予測を生成する。

- ## `F01__StackModel` クラス
  - **属性**
    - `base_model_oof_preds`: ベースモデルの訓練データに対するOut-Of-Fold (OOF) 予測を格納するDataFrame。
    - `base_model_test_preds`: ベースモデルのテストデータに対する予測を格納するDataFrame。
    - `meta_model`: スタッキングの最終段階で使用されるメタモデル。
    - `is_validation`: 検証モードか否かを示すブール値。`x04__ML.py`と同様。
    - `is_test`: テストモードか否かを示すブール値。`x04__ML.py`と同様。

  - ### `f01__get_base_predictions(self)`
    - **機能**: `output/04__ML/02__validation/` および `output/04__ML/03__submission/` ディレクトリから、各ベースモデルの予測結果（`.npy`ファイル）と設定ファイル（`configs.json`）を読み込む。
    - **処理**:
      1. `output/04__ML/02__validation/` 内の各タイムスタンプ付きディレクトリを走査する。
      2. 各タイムスタンプディレクトリ内の `exp__*` サブディレクトリから `01__y_pred.npy` と `configs.json` を読み込む。
      3. 読み込んだ予測を `base_model_oof_preds` に、対応するテスト予測を `base_model_test_preds` に格納する。
         - 各予測には、モデル名とハイパーパラメータを識別子としてカラム名に含める（例: `lgbm_exp1_pred`）。
    - **出力**: なし (クラスの属性にデータを格納)。

  - ### `f02__prepare_meta_features(self)`
    - **機能**: メタモデルの訓練と予測に使用する特徴量（ベースモデルの予測）を準備する。
    - **処理**:
      1. `is_validation` が `True` の場合、`base_model_oof_preds` をさらに3ブロックと2ブロックに分割する。
      2. `base_model_oof_preds` と `base_model_test_preds` をそれぞれメタモデルの訓練データとテストデータとして整形する。
      3. 必要に応じて、元の特徴量の一部をメタモデルの特徴量として追加することも検討する（オプション）。
    - **出力**:
      - `meta_X_train` (pd.DataFrame): メタモデルの訓練特徴量。
      - `meta_y_train` (pd.Series): メタモデルの訓練ターゲット（元の訓練データの`final_status`）。
      - `meta_X_test` (pd.DataFrame): メタモデルのテスト特徴量。

  - ### `f03__fit_predict_meta_model(self, meta_model_name: str = "LogisticRegression", meta_hyperparameters: dict = None)`
    - **機能**: メタモデルを訓練し、最終的な予測を生成する。
    - **入力**:
      - `meta_model_name` (str): 使用するメタモデルの名前（例: "LogisticRegression", "LightGBM"）。
      - `meta_hyperparameters` (dict): メタモデルのハイパーパラメータ。
    - **処理**:
      1. `meta_model_name` に基づいてメタモデルを初期化する。
      2. `is_validation` が `True` の場合、`f02__prepare_meta_features` で分割された3ブロックのデータでメタモデルを訓練し、残りの2ブロックで予測を行いF1スコアを算出する。
      3. `is_validation` が `False` の場合、`meta_X_train` と `meta_y_train` を使用してメタモデルを訓練し、`meta_X_test` に対して予測を行う。
    - **出力**:
      - `final_predictions` (np.ndarray): 最終的な予測結果。

  - ### `f04__save_final_predictions(self, final_predictions: np.ndarray)`
    - **機能**: 最終予測結果とメタモデルの設定を保存する。
    - **入力**:
      - `final_predictions` (np.ndarray): `f03__fit_predict_meta_model`から得られた最終予測結果。
    - **処理**:
      1. 予測結果を `output/07__Stacking/y_pred.npy` として保存する。
      2. メタモデルの設定（モデル名、ハイパーパラメータ、使用したベースモデルの予測ファイルパスなど）を `output/07__Stacking/configs.json` として保存する。
         - `is_validation` が `True` の場合、`output/07__Stacking/02__validation/{timestamp}/` に保存。
         - `is_validation` が `False` の場合、`output/07__Stacking/03__submission/{timestamp}/` に保存。
    - **出力**: なし。
# x09__StackNoLeak.py
## Concept
  - リークに注意してスタッキングモデルを構築する
  - スタッキングについて再度リサーチしたところ、OOF(Out of Fold)という手法が有効そう、かつリークがなさそうだった
  - 以下、validationモードのデータ活用フロー
    1. 事前に、利用可能なデータ全体をCV, Testに分割する
    2. CVをk-foldに分割して、Train, Valに分割する
    3. k回だけ、Trainデータでベースモデルを訓練して、Val領域とTest領域を予測することで、CV領域、Test領域全てでメタ特徴量を得られる
      - Test領域の予測値は、k回予測するためk通りの予測値が得られる。今回作成したスクリプトではkモデルの加重平均（重みはfoldごとのf1スコア）
    4. メタ特徴量も含めてTrainデータ領域全てでメタモデルを学習する
    5. Testデータでメタモデルの予測・評価
    - 一応、リーキングは発生していない・・・はず。
  - 提出用データ作成時には、train.csv全体をCVデータ、test.csvをTestデータとしてフローに従う
  - ベースモデルのバリエーションは以下の通り
    - LightGBM
    - XGBoost
    - MLP
    - CluCla(後述)
    - K近傍法モデル
  - CluCla
    - `desc`（テキストデータ列）に基づくジャンルごとに独立して分類モデルを構築し、それぞれ予測する
    - ジャンル・クラスタリング
      - `desc`をベクトル化（外部のスクリプトで実行）
      - KMeansで12ラベル（ジャンル）にクラスタリング
    - ジャンル別分類
      - モデル：LightGBM(可変)
  - ## Tips
    - 不均衡データへの対応
      - ツリー系モデル
        - 重みづけ
      - その他（MLP, CluCla, K近傍法）
        - ダウンサンプリングで目的変数ごとのサイズを同じにする
        - 今の所ランダムサンプリング。改善の余地あり
    - ベースモデル選択
      - ツリー系、ニューラルネット、k近傍法と性質の異なるモデルを混ぜた

- 家族全員分の所得及び課税に関する証明書(市区町村発行、所得及び住民税（所得割、均等割）の内訳が記載されているもの、コピー不可、本人、就学者分は除く)
- 奨学金の受給状況申告書
- 源泉徴収票
- 令和６年分所得税の確定申告書
- 最新の年金払込通知書または年金額改定通知書のコピー
- 