#!/bin/python

import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pickle
import argparse
import sys
import pdb
import time

# Train SVM

list_videos = 'labels/trainval.csv'
mfcc_dirs = ['bof10/','bof25/', 'bof/']
output_file = 'models/best_mfcc.model'

factor = 5
std = .05

np.random.seed(0)

start_time = time.time()

# 1. read all features in one array.
fread = open(list_videos, "r")
feat_list = []
# labels are [0-9]
label_list = []
# load video names and events in dict
df_videos_label = {}
for line in open(list_videos).readlines()[1:]:
  video_id, category = line.strip().split(",")
  df_videos_label[video_id] = category

layer_shapes = {}
mfcc_shape = {}

for line in fread.readlines()[1:]:
  video_id = line.strip().split(",")[0]
  feat = np.array([])

  label_list.append(int(df_videos_label[video_id]))

  # Add mfcc features
  for mfcc_dir in mfcc_dirs:
    feat_filepath = os.path.join(mfcc_dir, video_id + '.csv')
    if os.path.exists(feat_filepath):
      mfcc_feat = np.genfromtxt(feat_filepath, delimiter=";", dtype="float") 
      feat = np.concatenate((feat, mfcc_feat))
      mfcc_shape[mfcc_dir] = mfcc_feat.shape 
    else:
      feat = np.concatenate((feat, np.zeros(mfcc_shape[mfcc_dir])))

  feat_list.append(feat)

n = len(label_list)
inds = np.arange(n)
np.random.shuffle(inds)

label_list = np.array(label_list)
feat_list = np.array(feat_list)
print('feat_list shape ',feat_list.shape, np.mean(feat_list), np.std(feat_list))


start_time = time.time()

y = np.array(label_list)
X = np.array(feat_list)

# Augment data
y = np.tile(y, factor)
X = np.tile(X, (factor,1))

X[len(label_list):,:] += (np.random.randn(len(X) - len(label_list), X.shape[1])*std)

clf = MLPClassifier(hidden_layer_sizes=(1000,), activation="relu", solver="adam", \
    max_iter=1000, early_stopping=True, alpha=5e-4, n_iter_no_change=10, verbose=True)
clf.fit(X, y)


# save trained SVM in output_file
pickle.dump(clf, open(output_file, 'wb'))
print('Elapsed Time ', time.time() - start_time, ' seconds')
print('training accuracy: ', accuracy_score(y[:len(label_list)], clf.predict(X[:len(label_list)])))
