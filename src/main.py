from functools import wraps
import time

from modules import preprocessing
from modules import clusterMethods


def dev(fun):
    @wraps(fun)
    def wrap_the_function():
        start = time.time()
        fun()
        end = time.time()
        time_run = (end - start)/60
        print("-----------------------------------")
        print("time: ", time_run, "min")
        print("-----------------------------------")
    return wrap_the_function


@dev
def main():
    # preprocessing.run_preprocess()
    # preprocessing.run_chrom_all()
    # preprocessing.run_process()
    # preprocessing.run_plot_mean_of_genes()
    # clusterMethods.run_cluster(cluster_method="k-means++", feature_start=13, file_name=r'..\data\genes_arr_pd_plus_1.csv')
    # clusterMethods.run_cluster(cluster_method="k-means++", feature_start=8, file_name=r'..\data\genes_arr_pd.csv')
    # clusterMethods.run_cluster(cluster_method="DBSCAN", feature_start=8, file_name=r'..\data\genes_arr_pd.csv')
    clusterMethods.dbscan_eps_and_min_samples(feature_start=8, file_name=r'..\data\genes_arr_pd.csv')


if __name__ == "__main__":
    main()


