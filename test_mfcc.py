#!/bin/python

import argparse
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
import pickle
import sys
import numpy as np

# Apply the MLP model to the testing videos;
# Output prediction class for each video


list_videos = 'labels/test_for_student.label'
output_file = 'mfcc.csv'
model_file = 'models/best_mfcc.model'

mfcc_dirs = ['bof10/','bof25/', 'bof/']

if __name__ == '__main__':


  # 1. load mlp model
  mlp = pickle.load(open(model_file, "rb"))

  # 2. Create array containing features of each sample
  fread = open(list_videos, "r")
  feat_list = []
  video_ids = []
  mfcc_shape = {}

  # 2. Create array containing features of each sample
  fread = open(list_videos, "r")
  feat_list = []
  video_ids = []
  for line in fread.readlines():
    # HW00006228
    video_id = os.path.splitext(line.strip())[0]
    video_ids.append(video_id)

    feat = np.array([])
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

  X = np.array(feat_list)

  # 3. Get predictions
  # (num_samples) with integer
  pred_classes = mlp.predict(X)

  # 4. save for submission
  with open(output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(pred_classes):
      f.writelines("%s,%d\n" % (video_ids[i], pred_class))
