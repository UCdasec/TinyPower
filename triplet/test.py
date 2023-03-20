import matplotlib
matplotlib.use('Agg')
import os
import sys
sys.path.append("/home/uc_sec/Documents/lhp/PowerPruning/")
import pandas as pd
import pdb
import argparse
import matplotlib.pyplot as plt
import time

import numpy as np
import pandas as pd
from joblib import dump, load

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import utils.loadData as loadData
import utils.key_rank_new as key_rank


def print_run_time(func):
    ''' time decrator '''
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        last_time = end_time - start_time

        m, s = divmod(last_time, 60)
        h, m = divmod(m, 60)
        formatted_time = "{:d}:{:02d}:{:02d}".format(int(h), int(m), int(s))

        print('[LOG] -- function {} run time is: {}'.format(func.__name__, formatted_time))
    return wrapper


def plot_figure(x, y, real_key, xlabel, ylabel, title, fig_save_path):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks([0, real_key, 255])
    plt.plot(x, y)
    plt.savefig(fig_save_path)
    plt.show()
    plt.close()


def get_x_feat(opts):
    ''' load data and compute features '''
    # loading feature extractor & extract features for training
    root_dir = opts.root_dir
    dpath = opts.input
    tmp = opts.attack_window.split('_')
    attack_window = [int(tmp[0]), int(tmp[1])]
    method = opts.preprocess
    test_num = opts.test_num

    feat_ext_path = os.path.join(root_dir, 'feat_model', 'best_model_{}.h5'.format(opts.test_num))
    print('[LOG] -- loading the feature extractor from path: ', feat_ext_path)
    feat_model = load_model(feat_ext_path)
    feat_model.summary()
    whole_pack = np.load(dpath)
    x_data, plaintext, key = loadData.load_data_base(whole_pack, attack_window, method, test_num)

    x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], 1))
    x_data_feat = feat_model.predict(x_data)
    loadData.data_info(x_data_feat.shape, plaintext.shape, key)
    return x_data_feat, plaintext, key


def compute_knn_ranks(opts, x_test, plaintext, key):
    """ generate ranking raw data """
    print('[LOG] -- making predictions of guessed key and its corresopnding k-nn models ...')
    # get all the params
    root_dir = opts.root_dir
    knn_model_path = os.path.join(root_dir, 'knn_model', 'knn_model.m')
    if opts.cross_dev:
        ranking_root = os.path.join(root_dir, 'ranking_curve_cross_dev')
    else:
        ranking_root = os.path.join(root_dir, 'ranking_curve')

    eval_type = opts.eval_type
    target_byte = opts.target_byte
    leakage_model = opts.leakage_model

    y_test = loadData.get_labels(plaintext, key[target_byte], target_byte, leakage_model)
    knn_model = load(knn_model_path)

    # trained knn models path
    print('[LOG] -- k-nn models path: ', knn_model_path)
    start_time = time.time()
    acc = accuracy_score(y_test, knn_model.predict(x_test))
    end = time.time()
    print('[LOG] --  prediction time: {}'.format(end - start_time))
    print('[LOG] -- knn model: {}, real key is: {} test acc is: {}'.format(knn_model_path, key[target_byte], acc))

    preds = knn_model.predict_proba(x_test)
    max_trace_num = opts.test_num
    key_rank.ranking_curve(preds, key, plaintext, target_byte, ranking_root, leakage_model, max_trace_num)

    print('[LOG] -- all done!')


def main(opts):
    print('[LOG] -- load the data')
    x_data_feat, plaintext, key = get_x_feat(opts)

    print('[LOG] -- run compute rank')
    compute_knn_ranks(opts, x_data_feat, plaintext, key)

    print('[LOG] -- all done!')


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-rd', '--root_dir', help='')
    parser.add_argument('-cd', '--cross_dev', action='store_true', help='')
    parser.add_argument('-aw', '--attack_window', help='')
    parser.add_argument('-pp', '--preprocess', default='', choices={'', 'norm', 'scaling'}, help='')
    parser.add_argument('-tn', '--test_num', type=int, default=5000, help='')
    parser.add_argument('-tb', '--target_byte', type=int, default=2, help='')
    parser.add_argument('-lm', '--leakage_model', choices={'HW', 'ID'}, help='')
    parser.add_argument('-et', '--eval_type', help='')
    opts = parser.parse_args()
    return opts


class test_opts():
    def __init__(self):
        self.input = "/home/uc_sec/Documents/lhp/dataset/power_dataset/X1_K0_U_Delay_200k.npz"
        # self.input = "/home/uc_sec/Documents/lhp/dataset/power_dataset/X2_K1_U_20k.npz"
        self.root_dir = "./trained_model/unmasked_xmega_delay_cnn"
        self.test_num = 10000
        self.target_byte = 2
        self.leakage_model = 'HW'
        self.eval_type = ""
        self.preprocess = ""
        self.cross_dev = 0   
        self.attack_window = "0_147"
     


if __name__ == "__main__":
    # opts = parseArgs(sys.argv)
    opts = test_opts()
    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main(opts)

