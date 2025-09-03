import pandas as pd
import numpy as np
import json
## テスト予測値のうち、確信度が大きなデータをtrainデータに加えて、新規データを作成する

class F01__ShiftTest():
    def __init__(self):
        # LBの結果がローカルの結果より低かったため、テストデータの予測結果に誤りが多かったと考えられる
        # 次の取り組み: threshold_pos の値を0.75->0.8に変更
        # それに伴い、pos/neg の比率を同じくらいにするため、threshold_neg の値を0.1->0.08に変更
        self.threshold_pos = 0.80
        self.threshold_neg = 0.08

    def f01__get_data(self, data_train=None, data_test=None):
        self.data_submit = pd.read_csv("output/07__Stack/03__Ensemble/0901_1240/model_level_probas_is_submit.csv", index_col=0)

        if isinstance(data_train, pd.DataFrame):
            self.data_train = data_train
        else:
            self.data_train = pd.read_csv("output/05__01__SentenceTransformer_Raw/train.csv")
        if isinstance(data_test, pd.DataFrame):
            self.data_test = data_test
        else:
            self.data_test = pd.read_csv("output/05__01__SentenceTransformer_Raw/test.csv")

    def f02__get_shift(self, data_submit_mean=None):
        if isinstance(data_submit_mean, pd.Series):
            pass
        else:
            data_submit_mean = self.data_submit.mean(axis=1)

        data_shift = data_submit_mean[np.logical_or(
            data_submit_mean < self.threshold_neg,
            data_submit_mean > self.threshold_pos,
        )]

        self.logs = {
            "num_shift_positive": int((data_submit_mean > self.threshold_pos).sum()),
            "num_shift_negative": int((data_submit_mean < self.threshold_neg).sum()),
            "thresholds": [self.threshold_pos, self.threshold_neg]
        }

        self.shift_index = data_shift.index
        self.shift_label = (data_shift>0.5).astype(int)

        self.data_test_shift = self.data_test.iloc[self.shift_index]
        self.data_test_shift.loc[:, "final_status"] = self.shift_label.values

    def f03__save(self):
        self.data_train_shift = pd.concat(
            [self.data_train, self.data_test_shift], axis="index"
        ).reset_index(drop=True)
        self.data_train_shift.to_csv("output/10__PseudoLabeling/train.csv")

        with open("output/10__PseudoLabeling/log.json", "w") as f:
            json.dump(
                self.logs, f, indent=2
            )

if __name__ == "__main__":
    print("------ Start Cherry Picking ------")
    instance = F01__ShiftTest()
    instance.f01__get_data()
    instance.f02__get_shift()
    instance.f03__save()
    print("------ Done ------")