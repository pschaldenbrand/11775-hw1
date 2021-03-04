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
soundnet_dir = 'SoundNet-tensorflow/output'
soundnet_layers = [18, 21]
output_file = 'models/best_augmented.model'

factor = 3
std = 0.1

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

  # Add soundnet features
  for layer in soundnet_layers:
    feat_filepath = os.path.join(soundnet_dir, video_id + 'tf_fea{}.npy'.format(str(layer).zfill(2)))
    if os.path.exists(feat_filepath):
      sn_feat = np.load(feat_filepath)
      layer_feat = np.concatenate((np.min(sn_feat, axis=0), \
        np.max(sn_feat, axis=0), np.mean(sn_feat, axis=0),\
        np.std(sn_feat, axis=0), np.quantile(sn_feat, 0.25, axis=0), np.quantile(sn_feat, 0.75, axis=0)))
      feat = np.concatenate((feat, layer_feat))
      layer_shapes[layer] = layer_feat.shape
    else:
      feat = np.concatenate((feat, np.zeros(layer_shapes[layer])))

  feat_list.append(feat)

n = len(label_list)
inds = np.arange(n)
np.random.shuffle(inds)

label_list = np.array(label_list)
feat_list = np.array(feat_list)
print('feat_list shape ',feat_list.shape)


start_time = time.time()

y = np.array(label_list)
X = np.array(feat_list)

# Augment data
y = np.tile(y, factor)
X = np.tile(X, (factor,1))

X[len(label_list):,:] += (np.random.randn(len(X) - len(label_list), X.shape[1])*.1)

clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", \
    max_iter=1000, early_stopping=True, alpha=5e-4, n_iter_no_change=3)
clf.fit(X, y)


# save trained SVM in output_file
pickle.dump(clf, open(output_file, 'wb'))
print('Elapsed Time ', time.time() - start_time, ' seconds')
print('training accuracy: ', accuracy_score(y, clf.predict(X)))
