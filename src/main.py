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
    clusterMethods.run_benchmark()


if __name__ == "__main__":
    main()


