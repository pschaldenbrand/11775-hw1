#!/bin/python

import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import argparse
import sys
import pdb
import time

# Train SVM

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--layer", type=int, default=25) # THe soundnet layer


np.random.seed(0)

if __name__ == '__main__':
  args = parser.parse_args()
  start_time = time.time()

  # 1. read all features in one array.
  fread = open(args.list_videos, "r")
  feat_list = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category


  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    if ('SoundNet' in args.feat_dir):
      feat_filepath = os.path.join(args.feat_dir, video_id + 'tf_fea{}.npy'.format(str(args.layer).zfill(2)))
      if os.path.exists(feat_filepath):
        feat_list.append(np.mean(np.load(feat_filepath), axis=0))
        label_list.append(int(df_videos_label[video_id]))
        # print(feat_filepath)
        # print(feat_list[-1].shape)
      else:
        feat_list.append(np.zeros(feat_list[-1].shape))

    else:
      # for videos with no audio, ignore
      if os.path.exists(feat_filepath):
        feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))
        label_list.append(int(df_videos_label[video_id]))

  n = len(label_list)
  inds = np.arange(n)
  np.random.shuffle(inds)

  label_list = np.array(label_list)
  print(label_list.shape)
  feat_list = np.array(feat_list)
  print('feat_list shape ',feat_list.shape)

  best_acc = -1
  all_val_acc = []
  all_train_acc = []
  best_model = None

  for fold in range(args.folds):
    start_val = int(n * (float(fold)/args.folds))
    end_val = min(int(n * (float(fold+1)/args.folds)), n)

    train_fold_inds = np.concatenate((inds[:start_val], inds[end_val:]))
    val_fold_inds = inds[start_val:end_val]

    train_label_list = label_list[train_fold_inds]
    train_feat_list = feat_list[train_fold_inds]

    val_label_list = label_list[val_fold_inds]
    val_feat_list = feat_list[val_fold_inds]

    y = np.array(train_label_list)
    X = np.array(train_feat_list)

    # pass array for svm training
    # one-versus-rest multiclass strategy
    clf = SVC(cache_size=2000, decision_function_shape='ovr', kernel="rbf", tol=1e-2)
    clf.fit(X, y)

    y_val = np.array(val_label_list)
    X_val = np.array(val_feat_list)
    acc = accuracy_score(y_val, clf.predict(X_val))
    all_train_acc.append(accuracy_score(y, clf.predict(X)))
    # print("fold ", fold, " train accuracy: ", accuracy_score(y, clf.predict(X)))
    # print("fold ", fold, " validation accuracy: ", acc)
    # print()
    all_val_acc.append(acc)
    if acc > best_acc:
      best_acc = acc
      best_model = clf

  # save trained SVM in output_file
  pickle.dump(best_model, open(args.output_file, 'wb'))
  print('One-versus-rest multi-class SVM trained successfully')
  print('Elapsed Time ', time.time() - start_time, ' seconds')
  print('Average training accuracy: ', np.array(all_train_acc).mean())
  print('Average validation accuracy: ', np.array(all_val_acc).mean())
  print('Best validation accuracy: ', best_acc)
