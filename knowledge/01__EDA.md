# 1. 説明変数について
- `goal`
  - 目標調達金額
  - 対数変換することでほぼ左右対称の分布になる、が、目標金額の通貨単位がバラバラなので、一定のレートで変換する必要がありそう
  - 対数変換すると、`goal`と`final_status`は（だいたい）線形な関係にありそう
- `currency`
  - 通貨単位。以下の9種類が存在する
    - USD: 64485
    - GBP: 6017
    - CAD: 2637
    - AUD: 1335
    - EUR: 563
    - NZD: 255
    - SEK: 185
    - DKK: 142
    - NOK: 71
  - `country`で完全に代替できる
    - `US`と他のラベルとで、目的変数の分布が明らかに異なるため、質的変数での活用が望ましい
  - `disable_communication`
    - `True`の場合、`final_status`はすべて0になってる
    - これあり・なしで学習・提出したときを比較して、特徴量として利用するか否か決める。すべて0になるとか不自然すぎる

### 2015年5月時点の対USD為替レート
| 通貨 | 1単位あたりのUSD |
| :--- | :--- |
| **USD** | 1.0 |
| **GBP** | ~1.545 USD |
| **CAD** | ~0.824 USD |
| **AUD** | ~0.764 USD |
| **EUR** | ~1.116 USD |
| **NZD** | ~0.700 USD (2015年平均) |
| **SEK** | ~0.120 USD |
| **DKK** | ~0.152 USD |
| **NOK** | ~0.132 USD |

   * USD: United States Dollar
   * GBP: British Pound Sterling
   * CAD: Canadian Dollar
   * AUD: Australian Dollar
   * EUR: Euro
   * NZD: New Zealand Dollar
   * SEK: Swedish Krona
   * DKK: Danish Krone
   * NOK: Norwegian Krone

- 時刻系特徴量
  - 基本的に、どの特徴量もtrain, testで分布はあんまり変わらなさそう
  - `f02__diff__dead__and__state_changed`
    - ほとんどが0付近だが外れ値が存在する
    ```
    count    7.569000e+04
    mean     9.522085e+04
    std      4.984549e+05
    min     -1.089743e+06
    25%     -3.000000e+00
    50%     -1.000000e+00
    75%      0.000000e+00
    max      7.777728e+06
    Name: f02__diff__dead__and__state_changed, dtype: float64
    ```
  - 外れ値が多そう（ほとんど0に張り付いてる）なのは以下
    - `f02__diff__dead__and__state_changed`
    - `f02__diff__dead__and__created`
    - `f02__diff__changed__and__created`
    - `f02__diff__created__and__launched`
  - 上記を利用して外れ値特定に利用できるか?でもtestデータとの分布の乖離はない
  - `launched_at`と`final_status`を比較すると、2017年7月より前の以降で成功率が変動している（以降の方が、成功率が低い）
# 2. 評価指標
- 以下、コンペの記述を引用
  ```
    本コンペティションでは、クラウドファンディングプロジェクトの基本情報(目標金額、締め切り日など)をもとに、そのプロジェクトが資金調達に成功したかを予測する2値分類モデルを構築していただきます。
    参加者の皆様には、テーブルデータとテキスト情報を活用した多角的な特徴量設計・モデル構築を期待しております。

    評価関数
    精度評価は、評価関数「F1Score(binary)」を使用します。
    参考リンク: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    評価値は0～1の値をとり、精度が高いほど大きな値となります。
  ```
- F1スコア
- 調べたい事項
  - numeric データについて、train, test で分布が異なる可能性がある（validationではスコアが0.6くらいだったが、PublicScore は0.3だったため）
  - 時間系の特徴量の分布が異なるかも
  - レートを一定価格で変動させているが、実際は年によりレートは変化してる可能性もあるし、geminiのサーチの裏どりをしていないためレート計算が間違っている可能性がある
  - 目的変数が不均衡なデータなため、今後調整が必要か