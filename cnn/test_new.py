#!/usr/bin/python3
import matplotlib
matplotlib.use('Agg')
import os
import sys
sys.path.append("/home/erc528/lhp/PowerPruning")
import argparse
import pdb
import h5py

import tensorflow as tf
import numpy as np
from collections import defaultdict
import ast
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

import checking_tool
from utils import loadData
from utils import key_rank_new as key_rank


def load_data(opts):
    ''' load test data '''
    target_byte = opts.target_byte
    leakage_model = opts.leakage_model
    dpath = opts.input
    print("loading test data: {}".format(dpath))
    tmp = opts.attack_window.split('_')
    attack_window = [int(tmp[0]), int(tmp[1])]
    method = opts.preprocess
    test_num = opts.test_num

    whole_pack = np.load(dpath)
    traces, text_in, key = loadData.load_data_base(whole_pack, attack_window, method, opts.start_idx, train_num=test_num)
    inp_shape = (traces.shape[1], 1)
    labels = loadData.get_labels(text_in, key[target_byte], target_byte, leakage_model)

    clsNum = 9 if 'HW' == leakage_model else 256
    labels = to_categorical(labels, clsNum)
    return traces, labels, text_in, key, inp_shape


def plot_figure(x, y, model_file_name, dataset_name, fig_save_name, testType):
    plt.title('Performance of ' + model_file_name + ' against ' + dataset_name + ' testType ' + testType)
    plt.xlabel('number of traces')
    plt.ylabel('rank')
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig(fig_save_name)
    plt.show(block=False)
    plt.figure()


# Check a saved model against one of the testing databases Attack traces
def main(opts):
    # checking model file existence
    target_byte = opts.target_byte
    out_root = opts.output_root
    leakage_model = opts.leakage_model
    # cnn_model_path = os.path.join(out_root, 'model', 'best_model.h5')
    cnn_model_path = opts.model_file
    if opts.cross_dev:
        ranking_root = os.path.join(out_root, 'ranking_dir_cross_dev')
    else:
        ranking_root = os.path.join(out_root, 'ranking_dir')

    # Load model
    model = checking_tool.load_best_model(cnn_model_path)
    # model = load_model(cnn_model_path)
    model.summary()

    # Load profiling and attack data and metadata from the ASCAD database
    # val_traces, val_label, val_textin, key
    X_attack, Y_attack, plaintext, key, inp_shape = load_data(opts)

    # Get the input layer shape and Sanity check
    input_layer_shape = model.get_layer(index=0).input_shape
    if isinstance(input_layer_shape, list):
        input_layer_shape = input_layer_shape[0]
    Reshaped_X_attack = loadData.sanity_check(input_layer_shape, X_attack)

    # run the accuracy test
    score, acc = model.evaluate(Reshaped_X_attack, Y_attack, verbose=opts.verbose)
    print('[LOG] -- test acc is: {:f}'.format(acc))

    preds = model.predict(X_attack)
    max_trace_num = opts.test_num
    key_rank.ranking_curve(preds, key, plaintext, target_byte, ranking_root, leakage_model, max_trace_num)
    print('[LOG] -- [LOG] -- all done!')


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output_root', help='')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-cd', '--cross_dev', action='store_true', help='')
    parser.add_argument('-tb', '--target_byte', type=int, default=0, help='default value is 0')
    parser.add_argument('-lm', '--leakage_model', choices={'HW', 'ID'}, help='')
    parser.add_argument('-aw', '--attack_window', default='', help='overwrite the attack window')
    parser.add_argument('-pp', '--preprocess', default='', choices={'', 'norm', 'scailing'})
    parser.add_argument('-tn', '--test_num', type=int, default=5000, help='')
    opts = parser.parse_args()
    return opts


class test_opts():
    def __init__(self):
        self.input = "/home/erc528/lhp/dataset/power_dataset/X2_K1_U_Delay_20k.npz"
        # self.input = "/home/uc_sec/Documents/lhp/dataset/power_dataset/X1_K0_U_200k.npz"
        self.output_root = "/home/erc528/lhp/PowerPruning/cnn/test_results/unmasked_xmega_delay_cnn"
        self.model_file = "/home/erc528/lhp/PowerPruning/cnn/trained_model/unmasked_xmega_delay_cnn/hw_model_dir/best_model_1800_2800_trace_10000.hdf5"
        self.verbose = 1
        
        self.target_byte = 2
        self.leakage_model = "HW"
        # self.shifted = 0
        self.cross_dev = 1
        self.attack_window = "1800_2800"
        self.start_idx = 10000
        self.test_num = 10000
        self.preprocess = 0


if __name__=="__main__":
    # opts = parseArgs(sys.argv)
    opts = test_opts()
    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main(opts)