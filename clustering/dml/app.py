import glob
import os
import time
import csv
import numpy as np
import umap
import hdbscan
from decimal import Decimal


def run(**kwargs):

    fn_in_train = kwargs.get("../cluster")
    fn_in_train_zscore = kwargs.get("z_score")
    folder_in_test_orig = "testing"
    folder_in_test_zscore = "testing"
    fn_out_train_csv = "training_dml.csv"
    folder_out_test = "testing_dml"

    # 30
    n_neighbors = int(kwargs.get("knn"))
    # 0.3
    min_dist = float(kwargs.get("min_dist"))

    # 5,7,8,9,11,12,15,16,17,18,19,21,22,24,26,27
    # marker_list = sys.argv[9].split(',')
    # markers = [int(x) for x in marker_list]
    markers = kwargs.get("markers")
    if not markers:
        markers = [1, 2]

    print(fn_in_train)
    print(fn_in_train_zscore)
    print(folder_in_test_orig)
    print(folder_in_test_zscore)
    print(fn_out_train_csv)
    print(folder_out_test)
    print(n_neighbors)
    print(min_dist)
    print(markers)

    fn_in_test_list_orig = []
    fn_in_test_list_zscore = []
    fn_out_test_csv_list = []

    if not os.path.exists(folder_out_test):
        os.makedirs(folder_out_test)

    for file in glob.glob(folder_in_test_orig + "/*.csv"):
        fn_in_test_list_orig.append(file)
        fn_in_test_list_zscore.append(
            folder_in_test_zscore + "/" + os.path.basename(file)
        )
        fn_out_test_csv_list.append(folder_out_test + "/" + os.path.basename(file))

    length = len(fn_in_test_list_orig)

    train_data = fn_in_train
    train_data_zscore = fn_in_train_zscore

    train_labels = train_data[:, -1]
    train_data = train_data[:, :-1]
    train_data_for_calc = train_data_zscore[:, markers]
    t1 = time.time()
    mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42).fit(
        train_data_for_calc, train_labels
    )
    t2 = time.time()
    print("training umap done ", t2 - t1)

    # headers = headers[:-1]
    # headers.append("umap_x")
    # headers.append("umap_y")
    # headers.append("cluster_id")
    result1 = np.column_stack(
        (train_data, mapper.embedding_, train_labels.astype(float))
    )
    (H, W) = train_data.shape
    # np.savetxt(
    #     fn_out_train_csv,
    #     result1,
    #     fmt="%.5f",
    #     delimiter=",",
    #     header=",".join(headers),
    #     comments="",
    # )
    res = {'dml': result1}

    print("training done")

    for i in range(length):
        fn_in_test_orig = fn_in_test_list_orig[i]
        fn_in_test_zscore = fn_in_test_list_zscore[i]
        fn_out_test_csv = fn_out_test_csv_list[i]

        with open(fn_in_test_orig, "r") as f:
            reader = csv.reader(f, delimiter=",")
            headers2 = next(reader)
            test_data_orig = np.array(list(reader)).astype(float)
        with open(fn_in_test_zscore, "r") as f:
            reader = csv.reader(f, delimiter=",")
            headers2 = next(reader)
            test_data_zscore = np.array(list(reader)).astype(float)

        test_data_for_calc = test_data_zscore[:, markers]
        t0 = time.time()
        test_embedding = mapper.transform(test_data_for_calc)
        t1 = time.time()
        test_labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=50).fit_predict(
            test_embedding
        )
        t2 = time.time()

        # test_data[:, markers] = test_data_for_calc
        result2 = np.column_stack((test_data_orig, test_embedding, test_labels))
        # np.savetxt(
        #     fn_out_test_csv,
        #     result2,
        #     fmt="%.5f",
        #     delimiter=",",
        #     header=",".join(headers),
        #     comments="",
        # )
        res.update({f'dml_{i}': result2})
        print("{0} done: {1}, {2}".format(fn_out_test_csv, t1 - t0, t2 - t1))

    return res
