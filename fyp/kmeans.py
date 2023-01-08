import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import sys


def kmeans(dataset, mode=0):
    """
    labels -> predicted result
    best_n -> most suitable num of cluster
    :param mode:
    :param dataset:
    :return:
    """
    # read in dataset and strip last column
    output_real = 'k-means_output_real_' + dataset[0:3] + '.txt'
    output_pred = 'k-means_output_pred_' + dataset[0:3] + '.txt'
    df = pd.read_excel(dataset, engine="openpyxl")
    np.savetxt(output_real, df.iloc[:, -1], fmt='%s')
    df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)

    # select best cluster numbers using sihouette analysis
    range_n_clusters, results, best_n, max_avg = [2, 3, 4, 5, 6], [], 0, 0
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(df)
        silhouette_avg = silhouette_score(df, cluster_labels)
        # print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
        results.append(silhouette_avg)
        if max_avg < silhouette_avg:
            best_n = n_clusters
            max_avg = silhouette_avg
    # # plot line graph to show result, save figure into file
    # plt.plot(range_n_clusters, results)
    # plt.show()
    #     plt.savefig('silhouette_'+ dataset +'.png', dpi=300, bbox_inches='tight')

    # run kmeans using best number of clusters
    clustering = KMeans(n_clusters=best_n, init='random',
                        n_init=10, max_iter=300,
                        tol=1e-04, random_state=0)
    labels = clustering.fit_predict(df)
    np.set_printoptions(threshold=sys.maxsize)

    max_occur = 1
    max_count = 0
    for i in range(best_n):
        n = np.count_nonzero(labels == i)
        if n > max_count:
            max_count = n
            max_occur = i

    YN_labels = []
    # output labels to file
    with open(output_pred, "w") as txt_file:
        for line in labels:
            if line == max_occur:
                txt_file.write("N" + "\n")
                YN_labels.append("N")
            else:
                txt_file.write("Y" + "\n")
                YN_labels.append("Y")

    # # plot scatter graph and save
    # plt.scatter(
    #     df.iloc[labels == 0, 0].values, df.iloc[labels == 0, 1].values,
    #     s=50, c='lightgreen',
    #     marker='s', edgecolor='black',
    #     label='cluster 1'
    # )
    # plt.scatter(
    #     df.iloc[labels == 1, 0].values, df.iloc[labels == 1, 1].values,
    #     s=50, c='orange',
    #     marker='o', edgecolor='black',
    #     label='cluster 2'
    # )
    # plt.show()
    # #     plt.savefig('clustering_'+ dataset +'.png', dpi=300, bbox_inches='tight')

    if mode == 0:
        confusion_matrix(output_pred, output_real, 1)

    if mode == 1:
        return YN_labels


def confusion_matrix(predicted, real, mode=0):
    prediction = open(predicted, "r")
    actual = open(real, "r")

    predicted_array = []
    actual_array = []

    for line in prediction:
        line = line.strip()
        predicted_array.append(line)

    for line in actual:
        line = line.strip()
        actual_array.append(line)

    # total positives
    P = 0
    # total negatives
    N = 0
    # true positive
    TP = 0
    # true negative
    TN = 0
    # false positive
    FP = 0
    # false negative
    FN = 0

    for i in range(len(actual_array)):
        if actual_array[i] == "Y":
            P += 1
        else:
            N += 1
        j = predicted_array[i]
        if actual_array[i] == j:
            # if actual is yes and predicted is yes
            if i == "Y":
                TP += 1
            # if actual is nos and predicted is no
            else:
                TN += 1
        else:
            # if actual is yes but predicted is no
            if i == "Y":
                FN += 1
            # if actual is no but predicted is yes
            else:
                FP += 1

    TPR = (TP / P) * 100
    TNR = (TN / N) * 100
    FPR = (FP / N) * 100
    FNR = (FN / P) * 100
    try:
        Precision = (TP / (TP + FP)) * 100
    except ZeroDivisionError:
        Precision = 0
    try:
        Accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100
    except ZeroDivisionError:
        Accuracy = 0
    try:
        Recall = (TP / (TP + FN)) * 100
    except ZeroDivisionError:
        Recall = 0
    try:
        F1 = (2 * ((Precision * Recall) / (Precision + Recall))) * 100
    except ZeroDivisionError:
        F1 = 0

    # write to file
    with open("confusion_matrix_kmeans.txt", "a") as txt_file:
        txt_file.write(predicted + "\n")
        txt_file.write("TPR = " + str(TPR) + "\n")
        txt_file.write("TNR = " + str(TNR) + "\n")
        txt_file.write("FPR = " + str(FPR) + "\n")
        txt_file.write("FNR = " + str(FNR) + "\n")
        txt_file.write("Precision = " + str(Precision) + "\n")
        txt_file.write("Accuracy = " + str(Accuracy) + "\n")
        txt_file.write("Recall = " + str(Recall) + "\n")
        txt_file.write("F1 Score = " + str(F1) + "\n")
        txt_file.write("\n")

    if mode == 1:
        return Accuracy, Precision, Recall


# def main():
#     data = ["CM1.xlsx", "JM1.xlsx", "KC1.xlsx", "KC3.xlsx", "KC4.xlsx", "MC1.xlsx", "MC2.xlsx", "MW1.xlsx", "PC1.xlsx",
#             "PC2.xlsx", "PC3.xlsx", "PC4.xlsx", "PC5.xlsx"]
#
#     for i in data:
#         kmeans(i, 1)
#
#
# main()
