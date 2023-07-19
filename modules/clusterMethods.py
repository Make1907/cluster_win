import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
# from tslearn.clustering import KShape
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d


class Cluster:
    def __init__(self, data):
        self.data = data
        self.step = 10
        self.score = 0

    def normalization(self):
        self.data = self.data.rolling(window=self.step, win_type='gaussian', min_periods=1, axis='columns').mean(std=10)

        # plt.figure(figsize=(20, 5))
        # plt.plot(range(0, 1000), self.data.loc[0])
        # plt.xlabel("BP")
        # plt.ylabel("moving average")
        # plt.show()
        # plt.close()

        self.data = self.data.T

        # self.data = self.data.rolling(window=self.step, min_periods=1).mean()
        my_scaler = preprocessing.StandardScaler().fit(self.data)
        self.data = my_scaler.transform(self.data)
        self.data = self.data.T
        self.data = pd.DataFrame(self.data)

    def bench_k_means(self, kmeans):
        """Benchmark to evaluate the KMeans initialization methods.

        Parameters
        ----------
        kmeans : KMeans instance
            A :class:`~sklearn.cluster.KMeans` instance with the initialization
            already set.
        """
        estimator2 = kmeans.fit_predict(self.data)
        score = metrics.silhouette_score(self.data, estimator2)
        self.score = score

        # plt.figure(figsize=(20, 5))
        # plt.plot(range(0, 1000), self.data.loc[0])
        # plt.xlabel("BP")
        # plt.ylabel("Normalization and moving average")
        # plt.show()
        # plt.close()

        return estimator2

    @staticmethod
    def plot_result(data, n_digits):

        # reduced_data = PCA(n_components=100).fit_transform(data)
        # print(reduced_data.shape, type(reduced_data))
        reduced_data = data
        kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
        kmeans.fit(reduced_data)

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect="auto",
            origin="lower",
        )

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="x",
            s=169,
            linewidths=3,
            color="w",
            zorder=10,
        )
        plt.title(
            "K-means clustering on the digits dataset (PCA-reduced data)\n"
            "Centroids are marked with white cross"
        )
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()


class ResultAnalysis:
    def __init__(self, result_data, save_path, method):
        self.result_data = result_data
        self.save_path = save_path
        self.method = method
        grouped_genes = self.result_data.groupby("predict labels")
        self.mean_of_grouped_genes = grouped_genes.agg("mean")

        # print(type(grouped_genes.groups))

        print(80 * "*")
        self.groups = {}
        for key, item in grouped_genes.groups.items():
            print("{0} : {1}".format(key, len(item)))
            self.groups[key] = len(item)
        print(80 * "*")
        # print(grouped_genes.groups)

    def plot_means(self):
        plt.figure(figsize=(8, 5))
        for idx, row in self.mean_of_grouped_genes.iterrows():
            # row = gaussian_filter1d(row, sigma=5)
            plt.plot(range(0, 1000), row, label=str(idx) + "-->" + str(self.groups[idx]))
            plt.legend()

        # plt.title(str(idx))
        plt.xlabel("BP")
        plt.ylabel("Means Normalised Reads")

        plt.savefig(self.save_path + os.path.sep + self.method + "_0707.png",
                    dpi=180,
                    bbox_inches='tight')
        plt.close()


def run_benchmark(cluster_method="k-means++"):
    n_digits = 4
    file_name = r'..\data\genes_arr_pd.csv'
    save_path = r"..\result"
    data_ori = pd.read_csv(file_name)
    data = data_ori.iloc[:, 8:]

    # plt.figure(figsize=(20, 5))
    # plt.plot(range(0, 1000), data.loc[0])
    # plt.xlabel("BP")
    # plt.ylabel("original signal")
    # plt.show()
    # plt.close()

    cluster = Cluster(data)
    cluster.normalization()

    (n_samples, n_features) = data.shape

    print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")
    print(82 * "_")
    # scores = []
    # for i in range(4, 5):
    #     cluster_methods = {
    #                       "k-means++": KMeans(init="k-means++", n_clusters=n_digits, random_state=0),
    #                       "random": KMeans(init="random", n_clusters=n_digits, random_state=0),
    #                       "DBSCAN": DBSCAN(eps=30, min_samples=10)
    #                       }
    #     predict_labels = cluster.bench_k_means(kmeans=cluster_methods[cluster_method])
    #     scores.append(cluster.score)
    # print("scores: ", scores)

    cluster_methods = {
        "k-means++": KMeans(init="k-means++", n_clusters=n_digits, random_state=0),
        "random": KMeans(init="random", n_clusters=n_digits, random_state=0),
        "DBSCAN": DBSCAN(eps=30, min_samples=10)
    }
    predict_labels = cluster.bench_k_means(kmeans=cluster_methods[cluster_method])

    # plt.figure(figsize=(8, 5))
    # plt.plot(range(2, 12), scores)
    # plt.scatar
    # plt.xlabel("n_digits")
    # plt.ylabel("silhouette_score")
    # plt.show()
    # plt.close()

    method = cluster_method + str(n_digits)

    data["predict labels"] = predict_labels
    predict_labels_data = data_ori.iloc[:, :9]
    predict_labels_data["predict labels"] = predict_labels

    # predict_labels_data.to_csv(save_path + os.path.sep + cluster_method + "_predict_result.csv", index=None)

    print(data.head())

    result_analysis = ResultAnalysis(data, save_path, method)
    result_analysis.plot_means()

    print(82 * "_")


