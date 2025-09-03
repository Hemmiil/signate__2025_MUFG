# パラメータチューニングの結果
- LightGBM
  - 交差検証に使用するデータを増やしすぎると実行時間がめちゃ長くなってパソコンが火を吹くので、40000件程度にしておいた
  - しかしある程度データサイズを増やさないと、評価指標が小さくなる。10000件でテストしたところ、validationデータの評価指標は0.4程度だった。提出結果はおよそ0.58くらい出てるので、データサイズは40000が実現性と正確さのちょうど良い落とし所と思われる
  - num_leaves が最も変化が大きかった。値を大きくしすぎると、過学習が発生する。

    <img src="/Users/henmi_note/Desktop/signate2/output/98__HPT/0820/LightGBM/0075/num_leaves.png">

  - その他、正則化項(lambda, alpha)は10くらい大きくするとvalidation score が上昇した
  - 全体的に過学習の傾向にあることがわかった
- XGBoost(40000行、100 estimators)
  - 変化が大きかったグラフのみ記載

    <img src="/Users/henmi_note/Desktop/signate2/output/98__HPT/0820/XGBoost/0100/learning_rate.png">
    <img src="/Users/henmi_note/Desktop/signate2/output/98__HPT/0820/XGBoost/0100/max_depth.png">
    <img src="/Users/henmi_note/Desktop/signate2/output/98__HPT/0820/XGBoost/0100/subsample.png">
