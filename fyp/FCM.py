#!/usr/bin/env python
# coding: utf-8

### Importing the required Libraries
import pandas as pd
import numpy as np
import random
import operator
import math
from matplotlib import pyplot as plt


def FCM(filename):
    # ## Reading the Input Data


    ### reading the input csv file 

    df_full = pd.read_csv(filename,header=None)
    columns = list(df_full.columns)
    features = columns[:len(columns)-1]
    class_labels = list(df_full[columns[-1]])
    df = df_full[features]



    print(class_labels)



    # Number of Attributes
    num_attr = len(df.columns) - 1

    # Number of Clusters to make
    k = 2

    # Maximum number of iterations
    MAX_ITER = 300

    # Number of data points
    n = len(df)

    # Fuzzy parameter
    m = 1.7


    # ## Utility Function for Evaluating the Result


    #### accuracy function for measuring the accuracy of our result

    def accuracy(cluster_labels, class_labels):
        county = [0,0]
        countn = [0,0]
        tp = [0, 0]
        tn = [0, 0]
        fp = [0, 0]
        fn = [0, 0]
        
        for i in range(len(df)):
            # Yes = 1, No = 0
            if cluster_labels[i] == 1 and class_labels[i] == 'Y':
                tp[0] = tp[0] + 1
            if cluster_labels[i] == 0 and class_labels[i] == 'N':
                tn[0] = tn[0] + 1
            if cluster_labels[i] == 1 and class_labels[i] == 'N':
                fp[0] = fp[0] + 1
            if cluster_labels[i] == 0 and class_labels[i] == 'Y':
                fn[0] = fn[0] + 1
        
        for i in range(len(df)):
            # Yes = 0, No = 1
            if cluster_labels[i] == 0 and class_labels[i] == 'Y':
                tp[1] = tp[1] + 1
            if cluster_labels[i] == 1 and class_labels[i] == 'N':
                tn[1] = tn[1] + 1
            if cluster_labels[i] == 0 and class_labels[i] == 'N':
                fp[1] = fp[1] + 1
            if cluster_labels[i] == 1 and class_labels[i] == 'Y':
                fn[1] = fn[1] + 1
        
        a0 = float((tp[0] + tn[0]))/(tp[0] + tn[0] + fn[0] + fp[0])
        a1 = float((tp[1] + tn[1]))/(tp[1] + tn[1] + fn[1] + fp[1])
        p0 = float(tp[0])/(tp[0] + fp[0])
        p1 = float(tp[1])/(tp[1] + fp[1])
        r0 = float(tp[0])/(tp[0] + fn[0])
        r1 = float(tp[1])/(tp[1] + fn[1])
        
        accuracy = [a0*100,a1*100]
        precision = [p0*100,p1*100]
        recall = [r0*100,r1*100]
        
        return accuracy, precision, recall



    # ### initializing the membership matrix with random values

    # def initializeMembershipMatrix():
    #     membership_mat = list()
    #     for i in range(n):
    #         random_num_list = [random.random() for i in range(k)]
    #         summation = sum(random_num_list)
    #         temp_list = [x/summation for x in random_num_list]
    #         membership_mat.append(temp_list)
    #     return membership_mat

    def initializeMembershipMatrix(): # initializing the membership matrix
        membership_mat = []
        for i in range(n):
            random_num_list = [random.random() for i in range(k)]
            summation = sum(random_num_list)
            temp_list = [x/summation for x in random_num_list]
            
            flag = temp_list.index(max(temp_list))
            for j in range(0,len(temp_list)):
                if(j == flag):
                    temp_list[j] = 1
                else:
                    temp_list[j] = 0
            
            membership_mat.append(temp_list)
        return membership_mat



    ### calculating the cluster center, is done in every iteration

    # def calculateClusterCenter(membership_mat): # calculating the cluster center
    #     cluster_mem_val = list(zip(*membership_mat))
    #     cluster_centers = []
    #     for j in range(k):
    #         x = list(cluster_mem_val[j])
    #         xraised = [p ** m for p in x]
    #         denominator = sum(xraised)
    #         temp_num = []
    #         for i in range(n):
    #             data_point = list(df.iloc[i])
    #             prod = [xraised[i] * val for val in data_point]
    #             temp_num.append(prod)
    #         numerator = map(sum, list(zip(*temp_num)))
    #         center = [z/denominator for z in numerator]
    #         cluster_centers.append(center)
    #     return cluster_centers

    def calculateClusterCenter(membership_mat): # calculating the cluster center
        cluster_mem_val = list(zip(*membership_mat))
        cluster_centers = []
        for j in range(k):
            x = list(cluster_mem_val[j])
            xraised = [p ** m for p in x]
            denominator = sum(xraised)
            temp_num = []
            for i in range(n):
                data_point = list(df.iloc[i])
                prod = [xraised[i] * val for val in data_point]
                temp_num.append(prod)
            numerator = map(sum, list(zip(*temp_num)))
            center = [z/denominator for z in numerator]
            cluster_centers.append(center)
        return cluster_centers


    # ### Step 3 : Updating Membership Values using Cluster Centers from Step 2

    # In[307]:


    # def updateMembershipValue(membership_mat, cluster_centers): # Updating the membership value
    #     p = float(2/(m-1))
    #     for i in range(n):
    #         x = list(df.iloc[i])
    #         distances = [np.linalg.norm(np.array(list(map(operator.sub, x, cluster_centers[j])))) for j in range(k)]
    #         for j in range(k):
    #             den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
    #             membership_mat[i][j] = float(1/den)       
    #     return membership_mat

    def updateMembershipValue(membership_mat, cluster_centers): # Updating the membership value
        p = float(2/(m-1))
        for i in range(n):
            x = list(df.iloc[i])
            distances = [np.linalg.norm(np.array(list(map(operator.sub, x, cluster_centers[j])))) for j in range(k)]
            for j in range(k):
                den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
                membership_mat[i][j] = float(1/den)       
        return membership_mat


    # ### Function defined which returns the Clusters from the Membership Matrix


    # def getClusters(membership_mat):
    #     cluster_labels = list()
    #     for i in range(n):
    #         max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
    #         cluster_labels.append(idx)
    #     return cluster_labels

    def getClusters(membership_mat): # getting the clusters
        cluster_labels = list()
        for i in range(n):
            max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
            cluster_labels.append(idx)
        return cluster_labels


    # ### Calling fcm function which runs for MAX_ITER number of times and returns the Result


    def fuzzyCMeansClustering(): #Third iteration Random vectors from data
        # Membership Matrix
        membership_mat = initializeMembershipMatrix()
        curr = 0
        acc=[]
        while curr < MAX_ITER:
            cluster_centers = calculateClusterCenter(membership_mat)
            membership_mat = updateMembershipValue(membership_mat, cluster_centers)
            cluster_labels = getClusters(membership_mat)
            
            acc.append(cluster_labels)
            
            if(curr == 0):
                print("Cluster Centers:")
                print(np.array(cluster_centers))
            curr += 1
        print("---------------------------")
        print("Partition matrix:")
        print(np.array(membership_mat))
        #return cluster_labels, cluster_centers
        return cluster_labels, cluster_centers, acc


    # ## Displaying the Results
    # ### Outputting Cluster Labels and Cluster Centers


    ### calling the main function and storing the final results in labels, centers
    labels, centers, acc = fuzzyCMeansClustering()

    print(labels)

    counter = 0

    for label in class_labels:
        if label == 'N':
            counter += 1

    if ( counter/len(class_labels) ) > 0.5 and (sum(labels)/len(labels) > 0.5) or ( (counter/len(class_labels) < 0.5) and ((sum(labels)/len(labels)) < 0.5) ) :
        for ind in range(len(labels)):
            if labels[ind] == 0:
                labels[ind] = 1
            else:
                labels[ind] = 0


    print("Labels:")
    print(labels)


    l_out = []
    for label in labels:
        if label == 0:
            l_out.append('N')
        else:
            l_out.append('Y')


    c_labels = []
    for label in class_labels:
        if label == 'N':
            c_labels.append(0)
        else:
            c_labels.append(1)

    print(c_labels)
    print(l_out)
    print(class_labels)


    out = np.array([class_labels,l_out,c_labels,labels])
    out = np.transpose(out, axes=None)
    np.savetxt("compare.csv", out, delimiter=",", header="True Label,Predicted, True Label, Predicted",
               fmt="%1s", comments='')


    df[len(df.columns)] = l_out
    df.to_csv("predicted.csv", header = None, index = False)





    # ### Outputting the accuracy achieved


    ### measuring the accuracy of the result

    a,p,r = accuracy(labels, class_labels)

    ### printing the values

    print("Accuracy = " + str(a))
    print("Precision = " + str(p))
    print("Recall = " + str(r))

    return a,p,r


    # cents=np.array(centers)


    # # plot result
    # rows=5
    # f, axes = plt.subplots(rows, 2, figsize=(11,25))
    # for i in range(rows):
    #     axes[i,0].scatter(list(df.iloc[:,i]), list(df.iloc[:,i+1]), alpha=0.8)
    #     axes[i,1].scatter(list(df.iloc[:,i]), list(df.iloc[:,i+1]), c=labels, alpha=0.8)
    #     axes[i,1].scatter(cents[:,i+1], cents[:,i+1], marker="+", s=500, c='g')
    # plt.show()