#!/bin/python 
import pandas as pd
import numpy as np
import os
from sklearn.cluster.k_means_ import KMeans
import pickle 
import sys
import time


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0]))
        print("mfcc_csv_file -- path to the mfcc csv file")
        print("cluster_num -- number of cluster")
        print("output_file -- path to save the k-means model")
        exit(1)

    mfcc_csv_file = sys.argv[1]; 
    output_file = sys.argv[3]
    cluster_num = int(sys.argv[2])
    
    # 1. load all mfcc features in one array
    selection = pd.read_csv(mfcc_csv_file, sep=';', dtype='float')
    X = selection.values
    start = time.time()
    kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_jobs=4).fit(X)
    end = time.time()

    # 2. Save trained model
    pickle.dump(kmeans, open(output_file, 'wb'))

    print("K-means trained successfully!")
    print("Time elapsed for training: ", end-start)
