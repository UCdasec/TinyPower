# Import necessary modules and packages
from __future__ import division  # Ensure division behaves as expected in Python 2 and 3
import os
import pathlib
import sys
from scipy.spatial import distance  # Import distance calculation from SciPy
import argparse  # Import argument parsing library

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, Model  # Import TensorFlow and Keras
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score as MI  # Import mutual information score from scikit-learn
import utils.loadData as loadData  # Import custom data loading utility
import json
import re

# Get the current working directory
cur_path = pathlib.Path().absolute()
ResDir = os.path.join(cur_path, 'rank_result')
outDir = os.path.join(cur_path, 'pruned_model')

# Create directories if they don't exist
if not os.path.exists(ResDir):
    os.mkdir(ResDir)
if not os.path.exists(outDir):
    os.mkdir(outDir)

# Define a function to calculate mutual information between two arrays
def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = MI(None, None, contingency=c_xy)
    return mi

# Define a function for quantization and mapping of weights
def mapping(W, min_w, max_w):
    scale_w = (max_w - min_w) / 100
    min_arr = np.full(W.shape, min_w)
    q_w = np.round((W - min_arr) / scale_w).astype(np.uint8)
    return q_w

# Define a function to rank feature maps in groups
def grouped_rank(feature_map, num_groups):
    dis = 256 / num_groups
    grouped_feature = np.round(feature_map / dis)
    r = np.linalg.matrix_rank(grouped_feature)
    return r

# Define a function to update distances between layers
def update_dis(Distances, layer_idx, dis):
    if layer_idx in Distances.keys():
        for k, v in dis.items():
            Distances[layer_idx][k] += v
    else:
        Distances[layer_idx] = dis
    return Distances

# Define a function to extract layers from a model
def extract_layers(model):
    layers = model.layers
    o = []
    model.summary()
    for i, l in enumerate(layers):
        if "conv" in l.name or "fc" in l.name:
            o.append(l.output)
    return o

# Define a function to calculate rank of filters in each layer
def cal_rank(features, Results):
    for layer_idx, feature_layer in enumerate(features):
        after = np.squeeze(feature_layer)
        n_filters = after.shape[-1]
        filter_rank = list()
        if len(after.shape) == 2:
            for i in range(n_filters):
                a = after[:, i]
                rtf = np.average(a)
                filter_rank.append(rtf)
            filter_rank = sorted(filter_rank, reverse=True)
        else:
            filter_rank = sorted(after, reverse=True)
        filter_rank = mapping(np.array(filter_rank), np.min(filter_rank), np.max(filter_rank))
        Results[layer_idx] = np.add(Results[layer_idx], np.array(filter_rank))
    return Results

# Define a function to extract feature maps from a model
def extract_feature_maps(opts, model, output_layers):
    dpath = opts.input
    tmp = opts.attack_window.split('_')
    attack_window = [int(tmp[0]), int(tmp[1])]
    method = opts.preprocess
    test_num = opts.max_trace_num
    Results = list()
    num_trace = 50
    extractor = Model(inputs=model.inputs, outputs=output_layers)
    for l in output_layers:
        Results.append(np.zeros(l.shape[-1]))
    whole_pack = np.load(dpath)
    x_data, plaintext, key = loadData.load_data_base(whole_pack, attack_window, method, test_num)
    for f in x_data[:num_trace]:
        x = np.expand_dims(f, axis=0)
        features = extractor(x)
        Results = cal_rank(features, Results)
    R_after = np.array(Results) / num_trace
    R_list = [list(r) for r in R_after]
    df = pd.DataFrame(R_list)
    df.to_csv(cur_path + "/stm_cnn_act.csv", header=False)
    return R_list

# Define a function to extract weights from a model
def extract_weights(model, opts):
    layers = model.layers
    model.summary()
    weights = list()
    Results = list()
    idx_results = list()
    r = list()
    for l in layers:
        if "conv" in l.name or "fc" in l.name:
            print(l.name)
            a = l.get_weights()[0]
            n_filters = a.shape[-1]
            for i in range(n_filters):
                w = a[..., i]
                r.append(np.linalg.norm(w, 2))
            r = mapping(np.array(r), np.min(r), np.max(r))
            Results.append(sorted(r, reverse=True))           
            idx_dis = np.argsort(r, axis=0)
            idx_results.append(idx_dis)
            r = list()
    os.makedirs(opts.output, exist_ok=True)
    df = pd.DataFrame(Results, index=None)
    df.to_csv(os.path.join(opts.output, "l2.csv"), header=False)
    df = pd.DataFrame(idx_results, index=None)
    df.to_csv(os.path.join(opts.output, "l2_idx.csv"), header=False)

# Define a function to apply FPGM (Filter Pruning via Geometric Median) on a model
def fpgm(model, opts, dist_type="l2"):
    layers = model.layers
    results = list()
    idx_results = list()
    r = list()
    for l in layers:
        if "conv" in l.name or "fc" in l.name:
            print(l.name)
            w = l.get_weights()[0]
            weight_vec = np.reshape(w, (-1, w.shape[-1]))

            if dist_type == "l2" or "l1":
                dist_matrix = distance.cdist(np.transpose(weight_vec), np.transpose(weight_vec), 'euclidean')
            elif dist_type == "cos":
                dist_matrix = 1 - distance.cdist(np.transpose(weight_vec), np.transpose(weight_vec), 'cosine')
            squeeze_matrix = np.sum(np.abs(dist_matrix), axis=0)
            distance_sum = sorted(squeeze_matrix, reverse=True)
            idx_dis = np.argsort(squeeze_matrix, axis=0)
            r = mapping(np.array(distance_sum), np.min(distance_sum), np.max(distance_sum))
            results.append(r)
            idx_results.append(idx_dis)
            r = list()
    os.makedirs(opts.output, exist_ok=True)
    df = pd.DataFrame(results, index=None)
    df.to_csv(os.path.join(opts.output, "fpgm.csv"), header=False)
    df = pd.DataFrame(idx_results, index=None)
    df.to_csv(os.path.join(opts.output, "fpgm_idx.csv"), header=False)

# Parse command line arguments
def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-m', '--model_dir', help='')
    parser.add_argument('-aw', '--attack_window', default='', help='overwrite the attack window')
    parser.add_argument('-pp', '--preprocess', default='', choices={'', 'norm', 'scaling'})
    parser.add_argument('-tn', '--max_trace_num', type=int, default=10000, help='')
    parser.add_argument('-type', '--type', choices={'l2', 'fpgm'}, help='')
    opts = parser.parse_args()
    return opts

if __name__ == '__main__':
    opts = parseArgs(sys.argv)
    model = load_model(os.path.join(opts.model_dir,'model','best_model.h5'))
    model.summary()
    
    if opts.type == 'l2':
        extract_weights(model, opts)
    else:
        fpgm(model, opts)
