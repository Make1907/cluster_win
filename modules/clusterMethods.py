import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
# from tslearn.clustering import KShape
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d


class Cluster:
    def __init__(self, data, save_path=r"..\result"):
        self.data = data
        self.step = 10
        self.score = 0
        self.save_path = save_path

    def normalization(self):

        # # self.data = self.data.rolling(window=self.step, win_type='gaussian', min_periods=1, axis='columns').mean(std=10)
        # self.data = self.data.rolling(window=self.step, min_periods=1, axis='columns').mean()
        #
        # # plt.figure(figsize=(10, 3))
        # # plt.plot(range(0, 1000), self.data.loc[0])
        # # plt.xlabel("BP")
        # # plt.ylabel("Frequency")
        # # plt.show()
        # # plt.close()
        #
        # self.data = self.data.T

        # self.data = self.data.rolling(window=self.step, min_periods=1).mean()
        my_scaler = preprocessing.StandardScaler().fit(self.data)
        self.data = my_scaler.transform(self.data)
        # self.data = self.data.T
        self.data = pd.DataFrame(self.data)

        # plt.figure(figsize=(10, 3))
        # plt.plot(range(0, 1000), self.data.loc[0])
        # plt.xlabel("BP")
        # plt.ylabel("Frequency")
        # plt.show()
        # plt.close()

        return self.data

    def bench_k_means(self, method):
        """Benchmark to evaluate the KMeans initialization methods.

        Parameters
        ----------
        method : KMeans instance
            A :class:`~sklearn.cluster.KMeans` instance with the initialization
            already set.
        """
        estimator2 = method.fit_predict(self.data)
        score = metrics.silhouette_score(self.data, estimator2)
        self.score = score

        # plt.figure(figsize=(10, 3))
        # plt.plot(range(0, 1000), self.data.loc[0])
        # plt.xlabel("BP")
        # plt.ylabel("Normalization and moving average")
        # plt.show()
        # plt.close()

        return estimator2

    def heat_map(self):
        # plt.figure(figsize=[5, 10])
        # plt.contourf(self.data, 20, cmap='RdGy')
        # plt.colorbar()
        # plt.show()
        # plt.close()

        height = self.data.shape[0]

        import matplotlib.colors as colors
        # plt.figure(figsize=[3, 10])
        bounds = np.linspace(-1.5, 1.5, 15)
        # plt.imshow(self.data, alpha=1, origin="lower", vmin=0, vmax=3, cmap='viridis', interpolation='bicubic')
        plt.imshow(self.data, norm=colors.BoundaryNorm(boundaries=bounds, ncolors=256), origin="lower", aspect=0.5)
        # plt.ylim([0, height])
        plt.colorbar()
        plt.show()

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

        # plt.figure()
        # plt.plot(np.arange(k_dist.shape[0]), k_dist[::-1])
        # plt.show()
        # plt.figure()
        # plt.scatter(np.arange(k_dist.shape[0]), k_dist[::-1])
        # plt.show()


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
        font_size = 16
        plt.figure()
        for idx, row in self.mean_of_grouped_genes.iterrows():
            # row = gaussian_filter1d(row, sigma=5)
            plt.plot(range(0, 1000), row, label=str(idx) + "-->" + str(self.groups[idx]))
            plt.legend()

        # plt.title(str(idx))
        plt.xlabel("BP", fontsize=font_size)
        plt.ylabel("Means Normalised Reads", fontsize=font_size)

        plt.savefig(self.save_path + os.path.sep + self.method + "_0729.png",
                    dpi=180,
                    bbox_inches='tight')
        plt.close()


def get_kmeans_k(cluster_method="DBSCAN", feature_start=8, file_name=r'..\data\genes_arr_pd.csv'):
    n_digits = 3
    save_path = r"..\result"
    data_ori = pd.read_csv(file_name)
    data = data_ori.iloc[:, feature_start:]

    # plt.figure(figsize=(10, 3))
    # plt.plot(range(0, 1000), data.loc[0])
    # plt.xlabel("BP")
    # plt.ylabel("Frequency")
    # plt.show()
    # plt.close()

    cluster = Cluster(data)
    cluster.normalization()

    ## heat_map
    # cluster.heat_map()
    (n_samples, n_features) = data.shape

    print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")
    print(82 * "_")
    scores = []
    for i in range(2, 12):
        print("i: ", i)
        cluster_methods = {
                          "k-means++": KMeans(init="k-means++", n_clusters=i, random_state=0),
                          "random": KMeans(init="random", n_clusters=n_digits, random_state=0),
                          "DBSCAN": DBSCAN(eps=30, min_samples=10)
                          }
        predict_labels = cluster.bench_k_means(method=cluster_methods[cluster_method])
        scores.append(cluster.score)
    print("scores: ", scores)

    # scores = [0.14792224221494626, 0.12112233918451712,
    #           0.09935008009206037, 0.07663103888390906,
    #           0.061511976996644006, 0.0602122169322626,
    #           0.06384733474805666, 0.050873534719762914,
    #           0.05501661246380542, 0.052448744814671155]
    #
    fontsize = 15
    plt.figure()
    plt.plot(range(2, 12), scores)
    plt.scatter(range(2, 12), scores, c="r")
    plt.xlabel("K", fontsize=fontsize)
    plt.ylabel("Silhouette score", fontsize=fontsize)
    plt.show()
    plt.close()


def run_benchmark(cluster_method="k-mean++", feature_start=8, file_name=r'..\data\genes_arr_pd.csv'):
    n_digits = 6
    save_path = r"..\result"
    data_ori = pd.read_csv(file_name)
    data = data_ori.iloc[:, feature_start:]

    # plt.figure(figsize=(10, 3))
    # plt.plot(range(0, 1000), data.loc[0])
    # plt.xlabel("BP")
    # plt.ylabel("Frequency")
    # plt.show()
    # plt.close()

    cluster = Cluster(data)
    cluster.normalization()

    ## heat_map
    # cluster.heat_map()
    (n_samples, n_features) = data.shape

    print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")
    print(82 * "_")
    cluster_methods = {
        "k-means++": KMeans(init="k-means++", n_clusters=n_digits, random_state=0),
        "random": KMeans(init="random", n_clusters=n_digits, random_state=0),
        "DBSCAN": DBSCAN(eps=30, min_samples=100)
    }
    predict_labels = cluster.bench_k_means(method=cluster_methods[cluster_method])
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


def run_dbscan(cluster_method="DBSCAN",
               feature_start=8,
               file_name=r'..\data\genes_arr_pd.csv'):
    save_path = r"..\result"
    data_ori = pd.read_csv(file_name)
    data = data_ori.iloc[:, feature_start:]

    cluster = Cluster(data=data, save_path=save_path)
    cluster.normalization()
    cluster.get_dbscan_eps()


    # print(82 * "_")
    # scores = []
    # begin = 50
    # end = 51
    # eps = np.arange(begin, end, 1)
    # min_samples = np.arange(2000, 2010, 2)
    # results = [[0 for _ in range(5)]]
    #
    # for i in eps:
    #     for j in min_samples:
    #         print(i, j)
    #         try:
    #             cluster_method = DBSCAN(eps=i, min_samples=j)
    #             predict_labels = cluster.bench_k_means(method=cluster_method)
    #             score = metrics.silhouette_score(data, predict_labels)  # 轮廓系数评价聚类的好坏，值越大越好
    #             raito = len(predict_labels[predict_labels[:] == -1]) / len(predict_labels)  # 计算噪声点个数占总数的比例
    #             n_clusters = len(set(predict_labels)) - (1 if -1 in predict_labels else 0)  # 获取分簇的数目
    #             results.append([i, j, score, raito, n_clusters])
    #             print("gird: ", [i, j, score, raito, n_clusters])
    #         except:
    #             continue
    # results = pd.DataFrame(results)
    # results.columns = ['eps', 'min_samples', 'score', 'raito', 'n_clusters']
    #
    # # results = pd.read_csv(r"D:\code\Final_project_data_science\data\DBSCAN_eps_para_back_up.csv")
    # # results.to_csv(save_path + os.path.sep + "DBSCAN_eps_para.csv", index=None)
    # sns.relplot(x="eps", y="min_samples", size='score', data=results, hue="n_clusters")
    # plt.xlim([23, 35])
    # plt.ylim([93, 106])
    # plt.show()
    # sns.relplot(x="eps", y="min_samples", size='raito', data=results, hue="n_clusters")
    # plt.xlim([23, 35])
    # plt.ylim([93, 106])
    # plt.show()
    #
    # # plt.scatter(results["eps"], results["min_samples"], s=results["score"] * (-100))
    # # plt.colorbar()
    # # plt.show()
    # # plt.scatter(results["eps"], results["min_samples"], results['raito'] * 30, cmap='viridis')
    # # plt.colorbar()
    # # plt.show()

