import os
import math

import matplotlib.pyplot as plt
import pandas as pd

CHR = {
    "chrI":    1,
    "chrII":   2,
    "chrIII":  3,
    "chrIV":   4,
    "chrV":    5,
    "chrVI":   6,
    "chrVII":  7,
    "chrVIII": 8,
    "chrIX":   9,
    "chrX":    10,
    "chrXI":   11,
    "chrXII":  12,
    "chrXIII": 13,
    "chrXIV":  14,
    "chrXV":   15,
    "chrXVI":  16,
    "chrM":    1000
}


class Preprocessing:
    def __init__(self, file_name, save_path):
        self.file_name = file_name
        self.save_path = save_path
        self.columns_name = ['chr', 'strand', 't5', 't3', 'ypd', 'gal', 'type', 'name']

    def read_bin_data(self):
        f_ori = open(self.file_name, 'r')
        count = 0

        data_list = []
        for line in f_ori.readlines():
            if count % 1000 == 0:
                print(count)
            # if count == 100:
            #     break
            line_list = line.split()
            res = line_list[:6]
            temp = ""
            for idx, ele in enumerate(line_list[6:]):
                if idx == len(line_list) - 7:
                    res.append(temp.rstrip())
                    res.append(ele)
                else:
                    temp = temp + ele + " "
            data_list.append(res)
            count += 1
        f_ori.close()
        data = pd.DataFrame(data_list, index=None, columns=self.columns_name)
        data.columns = data.loc[0]
        data = data[1:]
        data.to_csv(self.save_path + os.path.sep + "data.csv", index=None)

    def get_max_ypd(self, file_name):
        data_all = pd.read_csv(file_name)
        data_all.dropna(axis=0, how='any', inplace=True)

        genes = list(set(data_all["name"]))
        print(len(genes))
        result = []
        count = 0

        for gene in genes:
            if count % 500 == 0:
                print("count: ", count)
            max_ypd = data_all[(data_all["name"] == gene) & (data_all["type"] == "Covering one intact ORF")]
            if max_ypd.empty is False:
                max_ypd = max_ypd[max_ypd["ypd"] == max_ypd["ypd"].max()]
                result.append(max_ypd.values.tolist()[0])
            count += 1

        result_unsort = pd.DataFrame(result, columns=self.columns_name)
        result_unsort = result_unsort.groupby("chr")

        result_sort = pd.DataFrame(index=None, columns=self.columns_name)
        for name, group in result_unsort:
            group_sort = group.sort_values(by="t5")
            result_sort = result_sort.append(group_sort, ignore_index=True)
        print("result shape:", result_sort.shape)
        result_sort.to_csv(self.save_path + os.path.sep + "genes002.csv", index=None)

    #  combine all of the genes' value data
    def chrom_all(self):
        path_dir = os.listdir(self.file_name)
        path_dir.sort(key=lambda x: int(x[:-4]))
        genes_all = pd.read_csv(self.file_name + os.path.sep + path_dir[0])
        genes_all.columns = ["start", "1"]
        genes_all = genes_all.reindex(index=range(1531900), fill_value=0)

        for file_name_chrom in path_dir:
            if file_name_chrom == "1.csv":
                continue
            print("f: ", file_name_chrom)
            gene = pd.read_csv(self.file_name + os.path.sep + file_name_chrom)
            genes_all[file_name_chrom[:-4]] = gene["size"]
            print("genes_shape: ", genes_all.shape)

        print(genes_all.sample(10))
        genes_all.to_csv(self.save_path, index=None)


class Process:
    def __init__(self, file_path_gene, file_name_genes, save_path):
        self.file_path_gene = file_path_gene
        self.file_name_genes = file_name_genes
        self.save_path = save_path
        self.length_gene = 1000

    def plot_gene(self):
        f1 = pd.read_csv(self.file_path_gene)
        f2 = pd.read_csv(self.file_name_genes)
        f2_part = f2[f2["Chr"] == 1]
        plt.figure(figsize=(20, 5))
        plt.plot(f1["start"], f1["val"])
        # plt.scatter(f1["start"], f1["size"])
        # plt.scatter(data["start"], data["size"], s=0.2)
        # for start in f2_part["t5"]:
        #     plt.axvline(x=start, c="r", linestyle='dotted')
        # for end in f2_part["t3"]:
        #     plt.axvline(x=end, c="b", linestyle='dashed')
        plt.xlim()
        plt.ylim()
        plt.savefig(self.save_path + os.path.sep + self.file_path_gene[-8:-4] + ".png", dpi=90, bbox_inches='tight')
        plt.close()

    def get_array_1000_row(self):
        genes = pd.read_csv(self.file_name_genes)
        count = 0
        file_name_gene = ""
        genes_arr = []
        for index, row in genes.iterrows():
            if count % 100 == 0:
                print("count:", count)
            if file_name_gene[:-4] != str(row["chr"]):
                file_name_gene = str(row["chr"]) + ".csv"
                gene = pd.read_csv(self.file_path_gene + os.sep + file_name_gene)

            start = min(row["t3"], row["t5"])
            end   = max(row["t3"], row["t5"]) + 1
            if abs(start - end) < self.length_gene or row["ypd"] == 0:
                continue
            gene_arr = gene.iloc[start: end, [1]]
            gene_arr = gene_arr["size"].tolist()
            if row["t3"] < row["t5"]:
                gene_arr = gene_arr[::-1]
            gene_arr = gene_arr[:self.length_gene]
            genes_arr.append(row.tolist() + gene_arr)
            # print(len(genes_arr), len(genes_arr[0]))
            count += 1
        genes_arr_pd = pd.DataFrame(genes_arr)
        genes_arr_pd.to_csv(self.save_path + os.path.sep + "genes_arr_pd.csv", index=None)

        print(genes_arr_pd.shape)
        # print(genes_arr_pd)

    def get_array_1000_row_plus_1(self, t3, t5):
        genes = pd.read_csv(self.file_name_genes)
        count = 0
        file_name_gene = ""
        genes_arr = []
        genes[t5] = genes["-1 nucleosome"]
        genes[t3] = genes["ORF End"] + abs(genes["-1 nucleosome"] - genes["ORF Start"])
        print(genes.head())

        for index, row in genes.iterrows():
            if count % 100 == 0:
                print("count:", count)
            chr_name_mapping = str(CHR[row["Chr"]])
            if file_name_gene[:-4] != chr_name_mapping:
                file_name_gene = chr_name_mapping + ".csv"
                gene = pd.read_csv(self.file_path_gene + os.sep + file_name_gene)

            start = min(row[t3], row[t5])
            end   = max(row[t3], row[t5]) + 1
            if abs(start - end) < self.length_gene:
                continue
            gene_arr = gene.iloc[start: end, [1]]
            gene_arr = gene_arr["size"].tolist()
            if row[t3] < row[t5]:
                gene_arr = gene_arr[::-1]
            gene_arr = gene_arr[:self.length_gene]
            genes_arr.append(row.tolist() + gene_arr)
            # print(len(genes_arr), len(genes_arr[0]))
            count += 1
        genes_arr_pd = pd.DataFrame(genes_arr)
        genes_arr_pd.to_csv(self.save_path + os.path.sep + "genes_arr_pd_plus_1.csv", index=None)

        print(genes_arr_pd.shape)
        # print(genes_arr_pd)


def run_preprocess():
    file_name = r'..\data\S2_tcd_mTIFAnno.txt'
    save_path = r"..\result"
    prepro = Preprocessing(file_name, save_path)

    # txt to csv
    # prepro.read_bin_data()

    data_csv_file_name = r'..\data\data.csv'
    prepro.get_max_ypd(data_csv_file_name)


def run_chrom_all():
    file_name = r'..\data\genes_all'
    save_path = r"..\result\GenesAll.csv"
    prepro = Preprocessing(file_name, save_path)
    prepro.chrom_all()


def run_process():
    file_path_gene = r'..\data\genes_all'
    file_name_genes = r'..\data\13059_2018_1398_MOESM2_ESM_using.csv'
    save_path = r"..\result"
    process = Process(file_path_gene, file_name_genes, save_path)
    # process.plot_gene()
    process.get_array_1000_row_plus_1(t3="t3", t5="t5")


def run_plot_mean_of_genes():
    file_path_gene = r'..\data\genes_all'
    file_name_genes = r'..\data\genes_arr_pd.csv'
    save_path = r"..\result"
    data = pd.read_csv(file_name_genes)
    print(data.iloc[:, 7: 15].head())
    data = data.iloc[:, 8:]
    print(data.head())

    data_mean = data.mean(axis=0)
    print(data_mean)
    plt.figure(figsize=(10, 3))
    plt.plot(range(1000), data.loc[100])
    plt.show()



