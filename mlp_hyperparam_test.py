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
output_file = 'models/best.model'
mfcc_dirs = []#['bof10/','bof25/', 'bof/']
folds = 5


factor = 2

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

  # Add mfcc features
  for mfcc_dir in mfcc_dirs:
    feat_filepath = os.path.join(mfcc_dir, video_id + '.csv')
    if os.path.exists(feat_filepath):
      mfcc_feat = np.genfromtxt(feat_filepath, delimiter=";", dtype="float") / 25.
      feat = np.concatenate((feat, mfcc_feat))
      mfcc_shape[mfcc_dir] = mfcc_feat.shape 
    else:
      feat = np.concatenate((feat, np.zeros(mfcc_shape[mfcc_dir])))

  feat_list.append(feat)

n = len(label_list)
inds = np.arange(n)
np.random.shuffle(inds)

label_list = np.array(label_list)
print(label_list.shape)
feat_list = np.array(feat_list)
print('feat_list shape ',feat_list.shape)
print(np.mean(feat_list), np.std(feat_list))


# for hidden_size in [(100,), (1000,), (100,100), (1000,1000), (1000,1000,1000), (1000,1000,1000,1000)]:
# for hidden_size in [(50,), (10,), (10,10), (50,10), (10,10,10), (50,10,10,10)]:
# for std in [.2, .4, .6, .75, 1., 1.5]:
for factor in [1,2,3,4,6,8,10]:
  print(factor)
  start_time = time.time()

  best_acc = -1
  all_val_acc = []
  all_train_acc = []
  best_model = None


  for fold in range(folds):
    start_val = int(n * (float(fold)/folds))
    end_val = min(int(n * (float(fold+1)/folds)), n)

    train_fold_inds = np.concatenate((inds[:start_val], inds[end_val:]))
    val_fold_inds = inds[start_val:end_val]

    train_label_list = label_list[train_fold_inds]
    train_feat_list = feat_list[train_fold_inds]

    val_label_list = label_list[val_fold_inds]
    val_feat_list = feat_list[val_fold_inds]

    y = np.array(train_label_list)
    X = np.array(train_feat_list)

    y = np.tile(y, factor)
    X = np.tile(X, (factor,1))

    X[len(train_label_list):,:] += (np.random.randn(len(X) - len(train_label_list), X.shape[1])*.1)
    #print(X.shape, y.shape)

    # pass array for svm training
    # one-versus-rest multiclass strategy
    # clf = SVC(cache_size=2000, decision_function_shape='ovr', kernel="rbf", tol=1e-2, C=2.5)
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", \
        max_iter=1000, early_stopping=True, alpha=5e-4, n_iter_no_change=3)
    clf.fit(X, y)

    y_val = np.array(val_label_list)
    X_val = np.array(val_feat_list)
    acc = accuracy_score(y_val, clf.predict(X_val))
    # print("fold ", fold, " train accuracy: ", accuracy_score(y, clf.predict(X)))
    # print("fold ", fold, " validation accuracy: ", acc)
    # print()
    all_train_acc.append(accuracy_score(y, clf.predict(X)))
    all_val_acc.append(acc)
    if acc > best_acc:
      best_acc = acc
      best_model = clf


  # save trained SVM in output_file
  pickle.dump(best_model, open(output_file, 'wb'))
  # print('One-versus-rest multi-class SVM trained successfully')
  print('Elapsed Time ', time.time() - start_time, ' seconds')
  print('Average training accuracy: ', np.array(all_train_acc).mean())
  print('Average validation accuracy: ', np.array(all_val_acc).mean())
  # print('Best validation accuracy: ', best_acc)
