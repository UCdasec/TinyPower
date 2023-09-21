#!/usr/bin/python3
import os
import sys
import argparse
import pdb
import h5py
import statistics
import math

import tensorflow as tf
import numpy as np
from collections import defaultdict
import ast
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

import tools.checking_tool as checking_tool
import tools.process_data as process_data
import tools.model_zoo as model_zoo
import mytools.draw_base as draw_base


# Compute the rank of the real key for a give set of predictions
def rank(predictions, plaintext_list, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, target_byte):
    # Compute the rank
    if len(last_key_bytes_proba) == 0:
        # If this is the first rank we compute, initialize all the estimates to zero
        key_bytes_proba = np.zeros(256)
    else:
        # This is not the first rank we compute: we optimize things by using the
        # previous computations to save time!
        key_bytes_proba = last_key_bytes_proba

    for p in range(0, max_trace_idx-min_trace_idx):
        # Go back from the class to the key byte. '2' is the index of the byte (third byte) of interest.
        plaintext = plaintext_list[p][target_byte]
        for i in range(0, 256):
            # Our candidate key byte probability is the sum of the predictions logs
            # AES_Sbox[plaintext ^ i]
            tmp_label = process_data.aes_internal(plaintext, i)
            proba = predictions[p][tmp_label]
            if proba != 0:
                key_bytes_proba[i] += np.log(proba)
            else:
                # We do not want an -inf here, put a very small epsilon
                # that corresponds to a power of our min non zero proba
                min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                if len(min_proba_predictions) == 0:
                    print("Error: got a prediction with only zeroes ... this should not happen!")
                    sys.exit(-1)
                min_proba = min(min_proba_predictions)
                key_bytes_proba[i] += np.log(min_proba**2)
                '''
                min_proba = 0.000000000000000000000000000000000001
                key_bytes_proba[i] += np.log(min_proba**2)
                '''

    # Now we find where our real key candidate lies in the estimation.
    # We do this by sorting our estimates and find the rank in the sorted array.
    sorted_proba = np.array(list(map(lambda a : key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    return (real_key_rank, key_bytes_proba)


def full_ranks(model, dataset, key, plaintext_attack, min_trace_idx, max_trace_idx, target_byte, rank_step):
    # Real key byte value that we will use. '2' is the index of the byte (third byte) of interest.
    real_key = key[target_byte]
    # Check for overflow
    if max_trace_idx > dataset.shape[0]:
        raise ValueError("Error: asked trace index %d overflows the total traces number %d" % (max_trace_idx, dataset.shape[0]))

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape
    if isinstance(input_layer_shape, list):
        input_layer_shape = input_layer_shape[0]
    # Sanity check
    if input_layer_shape[1] != dataset.shape[1]:
        raise ValueError("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(dataset[0, :])))

    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        print('# This is a MLP')
        input_data = dataset[min_trace_idx:max_trace_idx, :]
    elif len(input_layer_shape) == 3:
        print('# This is a CNN: reshape the data')
        input_data = dataset[min_trace_idx:max_trace_idx, :]
        input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    else:
        raise ValueError("Error: model input shape length %d is not expected ..." % len(input_layer_shape))

    # Predict our probabilities
    predictions = model.predict(input_data, batch_size=200, verbose=0)

    index = np.arange(min_trace_idx+rank_step, max_trace_idx, rank_step)
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
    key_bytes_proba = []
    for t, i in zip(index, range(0, len(index))):
        real_key_rank, key_bytes_proba = rank(predictions[t-rank_step:t], plaintext_attack[t-rank_step:t], real_key, t-rank_step, t, key_bytes_proba, target_byte)
        f_ranks[i] = [t - min_trace_idx, real_key_rank]
    return f_ranks


def get_the_labels(textins, key, target_byte):
    labels = []
    for i in range(textins.shape[0]):
        text_i = textins[i]
        label = process_data.aes_internal(text_i[target_byte], key[target_byte])
        labels.append(label)

    labels = np.array(labels)
    return labels


def load_data(input, opts):
    # checking_tool.check_file_exists(ascad_database_file)
    # in_file = h5py.File(ascad_database_file, "r")
    target_byte = opts.target_byte
    network_type = opts.network_type
    data_path = input
    shifted = opts.shifted

    if not os.path.isfile(data_path):
        raise ValueError('file did not find: {}'.format(data_path))
    whole_pack = np.load(data_path)

    attack_window = opts.attack_window
    if attack_window:
        tmp = opts.attack_window.split('_')
        start_idx, end_idx = int(tmp[0]), int(tmp[1])
        attack_window = [start_idx, end_idx]

    if shifted:
        print('data will be shifted in range: ', [0, shifted])
        traces, labels, text_in, key, inp_shape = process_data.process_raw_data_shifted(whole_pack, target_byte, network_type, shifted, attack_window)
    else:
        traces, labels, text_in, key, inp_shape = process_data.process_raw_data(whole_pack, target_byte, network_type, attack_window)
    return traces, labels, text_in, key, inp_shape


def plot_figure(x_samekey, y_samekey, x_diffkey, y_diffkey, fig_save_name, scenario, marker_num, marker_size):
    #plt.title('')
    xlabel = 'Number of traces'
    ylabel = 'Mean rank'
    showGrid = False

    all_x = [x_samekey, x_diffkey]
    y1 = [y_samekey, y_diffkey]
    data = draw_base.DataArray(all_x, y1)

    if 'same' == scenario:
        label1 = 'Same device, Same key'
        label2 = 'Same device, Diff key'
    elif 'weak' == scenario:
        label1 = 'Week cross-device, Same key'
        label2 = 'Week cross-device, Diff key'
    elif 'strong' == scenario:
        label1 = 'Strong cross-device, Same key'
        label2 = 'Strong cross-device, Diff key'
    else:
        raise ValueError()

    if scenario == 'strong':
        ### for strong setting
        xTicks, yTicks = 0, [0, 64, 128, 192, 256]
        xLim, yLim = 0, 0
    else:
        ### for same and weak setting
        xTicks, yTicks = [0, 5, 10, 15, 20], [0, 2, 4, 6, 8]
        xLim, yLim = 0, 0

    # default value: LABELSIZE=26, LEGENDSIZE=26, TICKSIZE=26, LWidth=2
    LABELSIZE = 30
    LEGENDSIZE = 23
    TICKSIZE = 26
    LWidth = 2
    context = draw_base.Context(xlabel, ylabel, file2save=fig_save_name, y1Lim=yLim, xLim=xLim,
                                label1=label1, label2=label2, xTicks=xTicks, y1Ticks=yTicks,
                                showGrid=showGrid, MSize=marker_size, marker_num=marker_num,
                                LABELSIZE=LABELSIZE, LEGENDSIZE=LEGENDSIZE, TICKSIZE=TICKSIZE,
                                LWidth=LWidth)

    draw_base.draw_results(data, 1, context)


def shuffleData(X, y):
    # Data is currently unshuffled; we should shuffle 
    # each X[i] with its corresponding y[i]
    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]
    return X, y


def calc_acc(model, X_attack, Y_attack):
    ''' shuffle and run the test data 5 times '''
    print('shuffle the data and then calculate acc')
    X_attack_shuffled, Y_attack_shuffled = shuffleData(X_attack, Y_attack)

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape
    if isinstance(input_layer_shape, list):
        input_layer_shape = input_layer_shape[0]

    # Sanity check
    Reshaped_X_attack = process_data.sanity_check(input_layer_shape, X_attack_shuffled)

    # run the accuracy test
    Y_attack_shuffled = to_categorical(Y_attack_shuffled, num_classes=256)
    score, acc = model.evaluate(Reshaped_X_attack, Y_attack_shuffled, verbose=opts.verbose)

    print('test acc is: ', acc)
    return acc


def computing_ranking_data(model, X_attack, plaintext_attack, key, max_traces):
    # We test the rank over traces of the Attack dataset, with a step of 10 traces
    print('start computing rank value...')
    min_trace_idx = 0
    max_trace_idx = max_traces
    rank_step = 1
    target_byte = opts.target_byte
    ranks = full_ranks(model, X_attack, key, plaintext_attack, min_trace_idx, max_trace_idx, target_byte, rank_step)

    # We plot the results
    # f_ranks[i] = [t, real_key_rank]
    x = [ranks[i][0] for i in range(0, ranks.shape[0])]
    y = [ranks[i][1] for i in range(0, ranks.shape[0])]

    x, y = np.array(x, dtype=int), np.array(y, dtype=int)

    min_rank = process_data.compute_min_rank(y)
    return x, y, min_rank


def computing_avg_ranking(model, X_attack, plaintext_attack, key, n_times, max_traces):
    ''' floor the avg ranking numbers '''
    x, y, min_rank = computing_ranking_data(model, X_attack, plaintext_attack, key, max_traces)
    for i in range(n_times-1):
        X_attack_s, plaintext_attack_s = shuffleData(X_attack, plaintext_attack)
        tmp_x, tmp_y, tmp_min_rank = computing_ranking_data(model, X_attack_s, plaintext_attack_s, key, max_traces)
        y = y + tmp_y
        min_rank += tmp_min_rank

    y = y/n_times
    y = np.trunc(y).astype(int)
    min_rank = math.floor(min_rank/n_times)
    return x, y, min_rank


# Check a saved model against one of the testing databases Attack traces
def main(opts):
    # checking model file existence
    model_file = opts.model_file
    target_byte = opts.target_byte
    max_traces = opts.max_traces

    # Load model
    model = checking_tool.load_best_model(model_file)
    model.summary()

    # Load profiling and attack data and metadata from the ASCAD database
    # val_traces, val_label, val_textin, key
    X_samekey, Y_samekey, plaintext_samekey, samekey, inp_shape = load_data(opts.samekey_input, opts)
    X_diffkey, Y_diffkey, plaintext_diffkey, diffkey, inp_shape = load_data(opts.diffkey_input, opts)
    n_times = 5
    avg_acc_samekey = calc_acc(model, X_samekey, Y_samekey)
    avg_acc_diffkey = calc_acc(model, X_diffkey, Y_diffkey)

    print('samekey accuracy is: ', avg_acc_samekey)
    print('diffkey accuracy is: ', avg_acc_diffkey)

    samekey_x, samekey_y, samekey_min_rank = computing_avg_ranking(model, X_samekey, plaintext_samekey, samekey, n_times, max_traces)
    diffkey_x, diffkey_y, diffkey_min_rank = computing_avg_ranking(model, X_diffkey, plaintext_diffkey, diffkey, n_times, max_traces)
    print('samekey min rank is: ', samekey_min_rank)
    print('diffkey min rank is: ', diffkey_min_rank)

    os.makedirs(opts.output, exist_ok=True)
    testType = os.path.basename(opts.samekey_input).split('.')[0]
    testType = testType.split('_')[1]
    device_name = os.path.basename(os.path.dirname(opts.samekey_input))
    fig_save_dir = os.path.join(opts.output, opts.network_type)
    os.makedirs(fig_save_dir, exist_ok=True)
    fig_save_name = os.path.join(fig_save_dir, str(device_name) + '_rank_performance_byte_{}_{}.png'.format(target_byte, testType))
    print('figure save to file: {}'.format(fig_save_name))

    plot_figure(samekey_x, samekey_y, diffkey_x, diffkey_y, fig_save_name, opts.scenario, opts.marker_num, opts.marker_size)

    test_summary_path = 'test_summary.txt'
    contents = '{} --- {} --- {:f}\n'.format(opts.samekey_input, opts.network_type, avg_acc_samekey)
    contents += '{} --- {} --- {:f}\n\n'.format(opts.diffkey_input, opts.network_type, avg_acc_samekey)
    with open(test_summary_path, 'a') as f:
        f.write(contents)


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-si', '--samekey_input', help='')
    parser.add_argument('-di', '--diffkey_input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-m', '--model_file', help='')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-tb', '--target_byte', type=int, default=0, help='default value is 0')
    parser.add_argument('-nt', '--network_type', default='cnn2', choices={'mlp', 'cnn', 'cnn2', 'wang'}, help='')
    parser.add_argument('-s', '--shifted', type=int, default=10, help='')
    parser.add_argument('-aw', '--attack_window', default='', help='overwrite the attack window')
    parser.add_argument('-mt', '--max_traces', type=int, default=20, help='')
    parser.add_argument('-sn', '--scenario', choices={'same', 'weak', 'strong'}, help='')
    parser.add_argument('-mn', '--marker_num', type=int, default=3, help='')
    parser.add_argument('-ms', '--marker_size', type=int, default=8, help='')
    opts = parser.parse_args()
    return opts


if __name__=="__main__":
    opts = parseArgs(sys.argv)
    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main(opts)
