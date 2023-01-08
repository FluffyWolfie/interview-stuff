# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy.cluster.hierarchy as sc
from sklearn.cluster import AgglomerativeClustering


def Hierarchical_clustering(dataset, mode=0):
    # Import data
    output_real = 'hierarchical_output_real_' + dataset[0:3] + '.txt'
    output_pred = 'hierarchical_output_pred_' + dataset[0:3] + '.txt'
    df = pd.read_excel(dataset, engine="openpyxl")
    np.savetxt(output_real, df.iloc[:, -1], fmt='%s')
    df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)

    n_cl = 2
    cluster = AgglomerativeClustering(n_clusters=n_cl, affinity='euclidean', linkage='ward')

    cluster.fit(df)
    labels = cluster.labels_

    max_occur = 1
    max_count = 0
    for i in range(n_cl):
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
    with open("confusion_matrix_hierarchical.txt", "a") as txt_file:
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
#         Hierarchical_clustering(i, 1)
#
#
# main()
