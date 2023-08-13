import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


class Cluster:
    def __init__(self, data, save_path=r"..\result"):
        self.data = data
        self.length_of_gene = 1000
        self.step = 10
        self.score = 0
        self.save_path = save_path

    def plot(self):
        plt.figure(figsize=(10, 3))
        plt.plot(range(0, self.length_of_gene), self.data.loc[0])
        plt.xlabel("BP")
        plt.ylabel("Frequency")
        plt.show()
        plt.close()

    def normalization(self):
        self.data = self.data.rolling(window=self.step, win_type='gaussian', min_periods=1, axis='columns').mean(std=10)
        self.plot()
        self.data = self.data.T
        my_scaler = preprocessing.StandardScaler().fit(self.data)
        self.data = my_scaler.transform(self.data)
        self.data = self.data.T
        self.data = pd.DataFrame(self.data)

        self.plot()

        return self.data

    def cluster_fit(self, method):
        """Benchmark to evaluate the KMeans initialization methods.

        Parameters
        ----------
        method : KMeans instance
            A :class:`~sklearn.cluster.KMeans` instance with the initialization
            already set.
        """
        estimator = method.fit_predict(self.data)
        score = metrics.silhouette_score(self.data, estimator)
        self.score = score
        # self.plot()
        return estimator

    def heat_map(self):
        print(self.data.head())
        bounds = np.linspace(-1.5, 1.5, 15)
        # plt.imshow(self.data, alpha=1, origin="lower", vmin=0, vmax=3, cmap='viridis', interpolation='bicubic')
        plt.imshow(self.data, norm=colors.BoundaryNorm(boundaries=bounds, ncolors=256), origin="lower", aspect=0.5)
        plt.colorbar()
        plt.show()

    def get_dbscan_eps(self):
        data_np = np.array(self.data)
        print(data_np.shape)
        # data_np = data_np[0: 100, :20]

        def select_MinPts(data, k):
            k_dist = []
            for i in range(data.shape[0]):
                dist = (((data[i] - data) ** 2).sum(axis=1) ** 0.5)
                dist.sort()
                k_dist.append(dist[k])
                print("i: ", i, dist[k])
            return np.array(k_dist)

        k = data_np.shape[1] * 2 - 1
        k_dist = select_MinPts(data_np, k)
        k_dist.sort()
        k_dist_pd = pd.DataFrame({"k:dist": k_dist})
        k_dist_pd.to_csv(self.save_path + os.path.sep + "DBSCAN_k_dist.csv", index=None)

        plt.figure()
        plt.plot(np.arange(k_dist.shape[0]), k_dist[::-1])
        plt.show()
        plt.figure()
        plt.scatter(np.arange(k_dist.shape[0]), k_dist[::-1])
        plt.show()


class ResultAnalysis:
    def __init__(self, result_data, save_path, method):
        self.result_data = result_data
        self.save_path = save_path
        self.method = method
        grouped_genes = self.result_data.groupby("predict labels")
        self.mean_of_grouped_genes = grouped_genes.agg("mean")
        self.length_gene = 1000

        # print(type(grouped_genes.groups))

        print(80 * "*")
        self.groups = {}
        for key, item in grouped_genes.groups.items():
            print("{0} : {1}".format(key, len(item)))
            self.groups[key] = len(item)
        print(80 * "*")
        # print(grouped_genes.groups)

    def plot_means(self):
        font_size = 16
        plt.figure()
        for idx, row in self.mean_of_grouped_genes.iterrows():
            # row = gaussian_filter1d(row, sigma=5)
            plt.plot(range(0, self.length_gene), row, label=str(idx) + "-->" + str(self.groups[idx]))
            plt.legend()

        # plt.title(str(idx))
        plt.xlabel("BP", fontsize=font_size)
        plt.ylabel("Means Normalised Reads", fontsize=font_size)

        plt.savefig(self.save_path + os.path.sep + self.method + "_0729.png",
                    dpi=180,
                    bbox_inches='tight')
        plt.close()


def get_kmeans_k(cluster_method="DBSCAN", feature_start=8, file_name=r'..\data\genes_arr_pd.csv'):

    data_ori = pd.read_csv(file_name)
    data = data_ori.iloc[:, feature_start:]

    cluster = Cluster(data)
    cluster.normalization()
    scores = []
    for n_clusters in range(2, 12):
        print("i: ", n_clusters)
        cluster_methods = KMeans(init="k-means++", n_clusters=n_clusters, random_state=0)
        cluster.cluster_fit(method=cluster_methods[cluster_method])
        scores.append(cluster.score)

    fontsize = 15
    plt.figure()
    plt.plot(range(2, 12), scores)
    plt.scatter(range(2, 12), scores, c="r")
    plt.xlabel("K", fontsize=fontsize)
    plt.ylabel("Silhouette score", fontsize=fontsize)
    plt.show()
    plt.close()


def dbscan_eps_and_min_samples(feature_start=8,
                               file_name=r'..\data\genes_arr_pd.csv'):
    save_path = r"..\result"
    data_ori = pd.read_csv(file_name)
    data = data_ori.iloc[:, feature_start:]

    cluster = Cluster(data=data, save_path=save_path)
    cluster.normalization()
    # cluster.get_dbscan_eps()

    eps_begin = 28
    eps_end = 32
    mim_samples_begin = 98
    min_samples_end = 120
    eps = np.arange(eps_begin, eps_end, 1)
    min_samples = np.arange(mim_samples_begin, min_samples_end, 2)
    results = [[0 for _ in range(5)]]

    for i in eps:
        for j in min_samples:
            print(i, j)
            try:
                cluster_method = DBSCAN(eps=i, min_samples=j)
                predict_labels = cluster.cluster_fit(method=cluster_method)
                score = metrics.silhouette_score(data, predict_labels)  # 轮廓系数评价聚类的好坏，值越大越好
                raito = len(predict_labels[predict_labels[:] == -1]) / len(predict_labels)  # 计算噪声点个数占总数的比例
                n_clusters = len(set(predict_labels)) - (1 if -1 in predict_labels else 0)  # 获取分簇的数目
                results.append([i, j, score, raito, n_clusters])
                print("gird: ", [i, j, score, raito, n_clusters])
            except:
                continue
    results = pd.DataFrame(results)
    results.columns = ['eps', 'min_samples', 'score', 'raito', 'n_clusters']

    sns.relplot(x="eps", y="min_samples", size='score', data=results, hue="n_clusters")
    plt.xlim([23, 35])
    plt.ylim([93, 106])
    plt.show()
    sns.relplot(x="eps", y="min_samples", size='raito', data=results, hue="n_clusters")
    plt.xlim([23, 35])
    plt.ylim([93, 106])
    plt.show()


def run_cluster(cluster_method="k-mean++", feature_start=8, file_name=r'..\data\genes_arr_pd.csv'):
    n_digits = 6
    save_path = r"..\result"
    data_ori = pd.read_csv(file_name)
    data = data_ori.iloc[:, feature_start:]

    cluster = Cluster(data)
    cluster.normalization()
    (n_samples, n_features) = data.shape

    print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")
    print(82 * "_")
    cluster_methods = {
        "k-means++": KMeans(init="k-means++", n_clusters=n_digits, random_state=0),
        "DBSCAN": DBSCAN(eps=30, min_samples=100)
    }
    predict_labels = cluster.cluster_fit(method=cluster_methods[cluster_method])
    print("score: ", cluster.score)

    method = cluster_method + str(n_digits)

    data["predict labels"] = predict_labels
    predict_labels_data = data_ori.iloc[:, :feature_start + 1]
    predict_labels_data["predict labels"] = predict_labels

    # predict_labels_data.to_csv(save_path + os.path.sep + cluster_method + "_predict_result.csv", index=None)

    print(data.head())

    result_analysis = ResultAnalysis(data, save_path, method)
    result_analysis.plot_means()

    print(82 * "_")


def get_heat_map(feature_start=8, file_name=r'..\data\genes_arr_pd.csv'):
    save_path = r"..\result"
    data_ori = pd.read_csv(file_name)
    data = data_ori.iloc[:, feature_start:]
    cluster = Cluster(data=data, save_path=save_path)
    cluster.normalization()
    cluster.heat_map()


get_heat_map()
