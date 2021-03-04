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


parser = argparse.ArgumentParser()
parser.add_argument("model_file")
parser.add_argument("list_videos")
parser.add_argument("output_file")


soundnet_dir = 'SoundNet-tensorflow/output'
soundnet_layers = [18, 21]

if __name__ == '__main__':

  args = parser.parse_args()

  # 1. load mlp model
  mlp = pickle.load(open(args.model_file, "rb"))

  # 2. Create array containing features of each sample
  fread = open(args.list_videos, "r")
  feat_list = []
  video_ids = []
  layer_shapes = {}

  # 2. Create array containing features of each sample
  fread = open(args.list_videos, "r")
  feat_list = []
  video_ids = []
  for line in fread.readlines():
    # HW00006228
    video_id = os.path.splitext(line.strip())[0]
    video_ids.append(video_id)

    feat = np.array([])
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

  X = np.array(feat_list)

  # 3. Get predictions
  # (num_samples) with integer
  pred_classes = mlp.predict(X)

  # 4. save for submission
  with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(pred_classes):
      f.writelines("%s,%d\n" % (video_ids[i], pred_class))
