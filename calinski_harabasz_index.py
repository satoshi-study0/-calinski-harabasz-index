# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 14:26:58 2022

@author: Takada Satoshi
"""


from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = data
calinski_harabasz_index = []

NUM_CLUSTERS_RANGE = range(2,16)  # クラスター数を2～15個の範囲で比較
for num in NUM_CLUSTERS_RANGE:
    # K-Means
    kmeans = KMeans(n_clusters=num, random_state=1).fit(X)
 
    # we store the cluster labels
    labels = kmeans.labels_
 
    #print(metrics.calinski_harabasz_score(X, labels))
    calinski_harabasz_index.append(calinski_harabasz_score(X, labels))
   

plt.plot(NUM_CLUSTERS_RANGE, calinski_harabasz_index, marker="o")
plt.xlabel("Number of classes")
plt.ylabel("calinski harabasz index")
