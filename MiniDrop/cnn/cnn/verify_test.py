#!/usr/bin/python3
import os
import sys
import argparse
import pdb
import h5py
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import ast

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

sys.path.append("tools")
import checking_tool
import loadData
import key_rank


def plot_figure(x, y, title_str, fig_save_name):
    plt.title(title_str)
    plt.xlabel('number of traces')
    plt.ylabel('rank')
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig(fig_save_name)
    plt.show(block=False)
    plt.figure()


def load_data(opts):
    ''' load test data '''
    target_byte = opts.target_byte
    leakage_model = opts.leakage_model
    dpath = opts.input
    attack_window = opts.attack_window
    method = opts.preprocess
    trace_num = opts.test_num
    shifted = opts.shifted

    whole_pack = np.load(dpath)
    traces, text_in, key = loadData.load_data_base_test(whole_pack, attack_window, method, trace_num, shifted)
     
    ''' 
    #the following lines are for down sampleling
    n1=[]
    for i in range(0,int((np.shape(traces)[0]))):
        t1=[]
        for j in range(0,4000,4):
            t1.append(traces[i][j])
        n1.append(t1)
    traces=np.array(n1)
    '''


    inp_shape = (traces.shape[1], 1)
    labels = loadData.get_labels(text_in, key[target_byte], target_byte, leakage_model)

    clsNum = 9 if 'HW' == leakage_model else 256
    labels = to_categorical(labels, clsNum)
    return traces, labels, text_in, key, inp_shape


# Check a saved model against one of the testing databases Attack traces
def main(opts):
    # checking model file existence
    target_byte = opts.target_byte
    leakage_model = opts.leakage_model
    cnn_model_path = os.path.join(opts.model_dir, 'model', 'best_model.h5')
    ranking_root = os.path.join(opts.output, 'rank_dir')
    os.makedirs(ranking_root, exist_ok=True)

    # Load model
    # Load the model from the .h5 file
    model = tf.keras.models.load_model(cnn_model_path)

    # Print the model summary
    model.summary()
    # Define the shape of the input layer
    input_shape = (None, 1000, 1)

    # Add the input layer to the model
    input_layer = tf.keras.layers.Input(shape=input_shape[1:])
    model = tf.keras.models.Model(inputs=input_layer, outputs=model.output)

    # Print the model summary
    model.summary()
    # Load profiling and attack data and metadata from the ASCAD database
    X_attack, Y_attack, plaintext, key, inp_shape = load_data(opts)

    # Get the input layer shape and Sanity check
    input_layer_shape = model.get_layer(index=0).input_shape
    print(input_layer_shape)
    if isinstance(input_layer_shape, list):
        input_layer_shape = input_layer_shape[0]
    Reshaped_X_attack = loadData.sanity_check(input_layer_shape, X_attack)

    # run the accuracy test
    score, acc = model.evaluate(Reshaped_X_attack, Y_attack, verbose=opts.verbose)
    print('[LOG] -- test acc is: {:f}'.format(acc))

    preds = model.predict(X_attack)
    max_trace_num = min(5000, preds.shape[0])
    key_rank.ranking_curve(preds, key, plaintext, target_byte, ranking_root, leakage_model, max_trace_num)
    print('[LOG] ---- all done!')


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-m', '--model_dir', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-tb', '--target_byte', type=int, default=0, help='default value is 0')
    parser.add_argument('-lm', '--leakage_model', choices={'HW', 'ID'}, help='')
    parser.add_argument('-aw', '--attack_window', default='', help='overwrite the attack window')
    parser.add_argument('-pp', '--preprocess', default='', choices={'', 'norm', 'scailing'})
    parser.add_argument('-tn', '--test_num', type=int, default=5000, help='')
    parser.add_argument('-sh', '--shifted', type=int, default=0, help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main(opts)

