#!/bin/python

import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import pickle
import argparse
import sys
import pdb
import time

# Train SVM

list_videos = 'labels/trainval.csv'
soundnet_dir = 'SoundNet-tensorflow/output'
soundnet_layers = [21]
mfcc_dir = 'bof/'
folds = 5

# for soundnet_layers in [[9],[10],[11],[12],[13],[14],[15],[16],[17], [18],[19],[20],[21],[22],[23],[24],[25]]:
# for soundnet_layers in [[18,21]]:
for soundnet_layers in [[18,21]]:
# for soundnet_layers in [[21]]:
  print(soundnet_layers)
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
  mfcc_shape = None

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

    # # Add mfcc features
    # feat_filepath = os.path.join(mfcc_dir, video_id + '.csv')
    # if os.path.exists(feat_filepath):
    #   mfcc_feat = np.genfromtxt(feat_filepath, delimiter=";", dtype="float")
    #   feat = np.concatenate((feat, mfcc_feat))
    #   mfcc_shape = mfcc_feat.shape 
    # else:
    #   feat = np.concatenate((feat, np.zeros(mfcc_shape)))
    feat_list.append(feat)

  n = len(label_list)
  inds = np.arange(n)
  np.random.shuffle(inds)

  label_list = np.array(label_list)
  # print(label_list.shape)
  feat_list = np.array(feat_list)
  print('feat_list shape ',feat_list.shape)

  all_val_acc = []
  all_train_acc = []

  # Normalization
  # print('mean ', np.mean(feat_list), ' std ', np.std(feat_list))
  # feat_list = (feat_list - np.mean(feat_list)) / np.std(feat_list)
  # print('mean ', np.mean(feat_list), ' std ', np.std(feat_list))

  conf_mat = None

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

    clf = MLPClassifier(hidden_layer_sizes=(100), activation="relu", solver="adam", \
      max_iter=1000, early_stopping=True, n_iter_no_change=10)
    clf.fit(X, y)

    y_val = np.array(val_label_list)
    X_val = np.array(val_feat_list)
    acc = accuracy_score(y_val, clf.predict(X_val))

    cf = confusion_matrix(y_val, clf.predict(X_val))
    conf_mat = cf if conf_mat is None else conf_mat + cf

    all_train_acc.append(accuracy_score(y, clf.predict(X)))
    all_val_acc.append(acc)
  print(conf_mat)

  print('Elapsed Time ', time.time() - start_time, ' seconds')
  print('Average training accuracy: ', np.array(all_train_acc).mean())
  print('Average validation accuracy: ', np.array(all_val_acc).mean())
