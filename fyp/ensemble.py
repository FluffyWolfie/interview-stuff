from FCM import *
from kmeans import *
from HierarchicalClustering import *
import pandas as pd


def ensemble(filename, mode=0):
    output_real = 'ensemble_output_real_' + filename[0:3] + '.txt'
    output_pred = 'ensemble_output_pred_' + filename[0:3] + '.txt'
    df = pd.read_excel(filename, engine="openpyxl")
    np.savetxt(output_real, df.iloc[:, -1], fmt='%s')

    fcm_out = FCM(filename, 1)
    kmeans_out = kmeans(filename, 1)
    hierarchical_out = Hierarchical_clustering(filename, 1)
    YN_labels = []
    for i in range(len(fcm_out)):
        N_count = 0
        Y_count = 0
        if fcm_out[i] == "N":
            N_count += 1
        else:
            Y_count += 1
        if kmeans_out[i] == "N":
            N_count += 1
        else:
            Y_count += 1
        if hierarchical_out[i] == "N":
            N_count += 1
        else:
            Y_count += 1
        if N_count > Y_count:
            YN_labels.append("N")
        else:
            YN_labels.append("Y")

    # output labels to file
    with open(output_pred, "w") as txt_file:
        for line in YN_labels:
            if line == "N":
                txt_file.write("N" + "\n")
            else:
                txt_file.write("Y" + "\n")

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
    with open("confusion_matrix_ensemble.txt", "a") as txt_file:
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


ensemble("CM1.xlsx")
