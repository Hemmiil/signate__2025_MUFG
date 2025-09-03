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
# x04__ML_prototype.py
- ## Purpose
  - 簡易的に、現在のデータでどの程度の精度で分類ができるかを検証したい
- ## Concept
  - データ全体を5分割して、80%を訓練データとして利用するモデルを5通り作成する
  - テストデータ予測は5つのモデルのアンサンブルを計算する。それぞれのモデルの`sklearn.ensemble.RandomForestClassifier.predict_proba`の値の平均値を計算し、0.5未満ならラベル`0`、0.5以上ならラベル`1`を与える
- F01__RandomForestClassifier.py
  - attribute
    - `is_validation`
      - `data/train.csv`由来のデータのみを利用した検証実験の時は`True`, `data/test.csv`由来のデータも活用し、提出用データを作成する時は`False`
    - `is_test`
      - `output/02__preprocessed/train_test.csv`を利用した、プログラムテストの時は`True`, `output/02__preprocessed/train.csv`を利用した本番環境の時は`False`
    - `dataset`
      - キーが`X_train`, `y_train`, `X_test`, `y/test`の辞書型
  - f01__get_data
    - `is_test`が`True`なら`output/02__preprocessed/train_test.csv`を、`False`なら`output/02__preprocessed/train.csv`をロードする。
    - `is_validation`が`True`なら対応するテストデータもダウンロードする
  - f02__split_data
    - `dataset`の作成
    - `is_validation`がTrueなら、訓練データを分割して`dataset`を作成する。`False`ならテストデータを`dataset["X_test"]`, `dataset["y_test"]`にりゆおする
  - f03__fit_predict
    - モデルの学習
  - f03__fit_predict
    - モデルの学習と予測を実行する。
    - `is_validation`が`True`の場合、検証データに対する予測結果を生成する。
    - `is_validation`が`False`の場合、実際のテストデータに対する予測結果を生成する。
  - f04__save_y_pred
    - 予測結果を保存する。
    - `output/04__ML_prototype/y_pred.npy`としてNumPy配列形式で保存する。
    - `is_validation`が`True`の場合、F1スコアを計算し、コンソールに出力する。
- ## Execution
  - `scripts/p04.sh` を使用して実行する。
  - `scripts/p04.sh` は `is_validation` と `is_test` のフラグを制御する。
  - **Validation Mode**: `python x04__ML_prototype.py --validation [--test]`
    - `data/train.csv` を内部で分割し、モデルの性能を評価する。
    - 結果としてF1スコアがコンソールに出力される。
  - **Submission Mode**: `python x04__ML_prototype.py [--test]`
    - `data/train.csv` でモデルを学習し、`data/test.csv` に対する予測を生成する。
    - 結果として `output/04__ML_prototype/y_pred.npy` が保存される。