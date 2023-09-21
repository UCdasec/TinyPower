#!/usr/bin python3.6
import os
import sys
import argparse
import pdb

import h5py
import numpy as np
from datetime import datetime
import configparser
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import tools.checking_tool as checking_tool
import tools.process_data as process_data
import tools.model_zoo as model_zoo
import mytools.tools as mytools


def load_sca_model(model_file):
    checking_tool.check_file_exists(model_file)
    try:
        model = load_model(model_file)
    except Exception as e:
        raise ValueError("Error: can't load Keras model file {} with message {}".format(model_file, e))
    return model


def load_data_general(opts, fpath):
    '''data loading function'''
    data_path = fpath
    target_byte = opts.target_byte
    network_type = opts.network_type
    attack_window = opts.attack_window

    if attack_window:
        tmp = attack_window.split('_')
        start_idx, end_idx = int(tmp[0]), int(tmp[1])
        attack_window = [start_idx, end_idx]

    whole_pack = np.load(data_path)
    traces, labels, text_in, key, inp_shape = process_data.process_raw_data(whole_pack, target_byte, network_type, attack_window)
 
    return traces, labels, text_in, key, inp_shape


def load_tuning_data(opts):
    # load the datas
    data_path = opts.input
    traces, labels, text_in, key, inp_shape = load_data_general(opts, data_path)
    # limit the data
    data_dict = mytools.data2datadict(traces, labels, sampleLimit=opts.trace_num)
    new_traces, new_labels = mytools.datadict2data(data_dict)
    return new_traces, new_labels


# Training high level function
def tuning_model(X_profiling, Y_profiling, best_model, save_file_name, epochs=50, batch_size=100, verbose=False, non_fixed=3):
    # check modelDir existence
    checking_tool.check_file_exists(os.path.dirname(save_file_name))

    # Save model every epoch
    log_dir = "logs/tune_fit/" + datetime.now().isoformat(sep='_', timespec='hours')
    tensorBoard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpointer = ModelCheckpoint(save_file_name, monitor='val_accuracy', verbose=verbose, save_best_only=True, mode='max')
    earlyStopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)
    callbacks = [checkpointer, earlyStopper, tensorBoard]

    # load existing model
    model = best_model
    depth = len(model.layers)
    for i in range(depth - non_fixed):
        model.layers[i].trainable = False

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape
    if isinstance(input_layer_shape, list):
        input_layer_shape = input_layer_shape[0]

    # Sanity check
    Reshaped_X_profiling = process_data.sanity_check(input_layer_shape, X_profiling)

    Y_profiling = to_categorical(Y_profiling, num_classes=256)
    history = model.fit(x=Reshaped_X_profiling, y=Y_profiling,
                        validation_split=0.1, batch_size=batch_size,
                        verbose=verbose, epochs=epochs,
                        shuffle=True, callbacks=callbacks)
    print('model save to path: {}'.format(save_file_name))
    return history


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to input data')
    parser.add_argument('-o', '--output', help='path to output folder')
    parser.add_argument('-m', '--model_file', help='path to model file')
    parser.add_argument('-tb', '--target_byte', type=int, help='')
    parser.add_argument('-nt', '--network_type', choices={'cnn', 'mlp', 'cnn2', 'wang'}, help='')
    parser.add_argument('-tn', '--trace_num', type=int, help='trace num per class for tuning')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-aw', '--attack_window', default='', help='attack window to be used')
    opts = parser.parse_args()
    return opts


def main(opts):
    # make resDir and modelDir
    resDir = opts.output
    modelDir = os.path.join(resDir, 'modelDir')
    os.makedirs(modelDir, exist_ok=True)
    dataset_name = os.path.basename(os.path.dirname(opts.input))
    network_type = opts.network_type
    target_byte = opts.target_byte

    model_save_file = os.path.join(modelDir, 'tune_model_{}_dataset_{}_targetbyte_{}_tuneNum_{}.hdf5'.format(network_type, dataset_name, target_byte, opts.trace_num))
    # set all the params
    epochs = 30
    batch_size = 128

    #load traces
    X_profiling, Y_profiling = load_tuning_data(opts)

    # get the model
    best_model = load_sca_model(opts.model_file)

    # training
    history = tuning_model(X_profiling, Y_profiling, best_model, model_save_file, epochs, batch_size, opts.verbose)

    print('all done!')


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main(opts)
