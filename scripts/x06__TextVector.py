import pandas as pd
import numpy as np

class F01__TextVector():
    def __init__(self):

        self.v02__subset_size = 20
        self.v02__max_cluster_size = 5
        self.v03__num_PC = 20


    def f01__get_data(self):
        self.v01__dataset_preprocessed = {
            "train": pd.read_csv("output/02__preprocessed/train.csv"),
            "test": pd.read_csv("output/02__preprocessed/test.csv")
        }

        self.v01__dataset_original = {
            "train": pd.read_csv("data/train.csv"),
            "test": pd.read_csv("data/test.csv")
        }

        self.v01__dataset_sentence = {
            "train": pd.DataFrame(
                np.load("output/05__SentenceTransformer/train.npy"), columns=[
                    f"f06__PC{str(i).zfill(3)}__sentencePC" for i in range(1, 385)
                ]
            ),
            "test": pd.DataFrame(
                np.load("output/05__SentenceTransformer/test.npy"), columns=[
                    f"f06__PC{str(i).zfill(3)}__sentencePC" for i in range(1, 385)
                ]
            ),
        }

    def f02__optimize_num_clusters(self):
        # ハイパーパラメータ：クラスタ数の最適化

        from sklearn.cluster import KMeans
        import matplotlib.pyplot as plt

        dataset_sentence_subset = self.v01__dataset_sentence["train"].iloc[:self.v02__subset_size]
        inertias = []
        for k in range(2, self.v02__max_cluster_size+1, 2):
            model = KMeans(n_clusters=k, random_state=42)
            model.fit(dataset_sentence_subset.iloc[:, :self.v02__subset_size])
            inertias.append(model.inertia_)

        plt.plot(range(2, 31, 2), inertias, marker='o')
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia (SSE)")
        plt.savefig("output/06__TextVector/01__optimize_num_clusters/01__elbow.png")
        plt.close()

        from sklearn.metrics import silhouette_score

        scores = []
        for k in range(2, 31, 2):
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(dataset_sentence_subset.iloc[:, :self.v02__subset_size])
            score = silhouette_score(dataset_sentence_subset.iloc[:, :self.v02__subset_size], labels)
            scores.append(score)

        plt.plot(range(2, 31, 2), scores, marker='o')
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette score")
        plt.savefig("output/06__TextVector/01__optimize_num_clusters/02__silhouette.png")
        plt.close()

    def f03__clustering(self):
        # 全体データではサイズが大きすぎたのでサブサンプリング。ここはブラッシュアップの余地ありそう
        from sklearn.cluster import KMeans

        self.v03__num_clusters = 12

        dataset_sentence_subset = self.v01__dataset_sentence["train"].iloc[:self.v02__subset_size, :self.v03__num_PC]

        model = KMeans(n_clusters=self.v03__num_clusters, random_state=42)
        y_pred = model.fit_predict(dataset_sentence_subset.iloc[:, :self.v])

        df__desc__label = self.v01__dataset_original[["desc"]].iloc[:self.v02__subset_size].copy()
        df__desc__label["label"] = y_pred

        ## ラベル別に集計
        dict__label__desc = {}
        for label in range(self.v03__num_clusters):
            dict__label__desc[label] = df__desc__label[df__desc__label["label"]==label]["desc"]

        # 一旦結果の保存
        save_dir = "../output/06__TextVector/02__cluster__text/"
        import os
        # os.makedirs(save_dir)
        for i in range(self.v03__num_clusters):
            dict__label__desc[i].to_csv(f"{save_dir}/cls_{str(i+1).zfill(3)}.csv")
            pass