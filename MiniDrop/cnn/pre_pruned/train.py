#!/usr/bin python3.6
import os
import sys
sys.path.append("/home/mabon/Tiny_power/code/TinyPower")
import argparse
import pdb
import h5py

import tensorflow as tf
import numpy as np
from datetime import datetime

# from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from one_cycle_lr import OneCycleLR

import checking_tool
import process_data
from utils import model_zoo


def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


def load_sca_model(model_file):
    check_file_exists(model_file)
    try:
        model = load_model(model_file)
    except:
        raise ValueError("Error: can't load Keras model file {}".format(model_file))
    return model


def load_training_data(opts):
    '''data loading function'''
    target_byte = opts.target_byte
    network_type = opts.network_type
    # start_idx and end_idx will be included in the dataset now
    # start_idx, end_idx = opts.start_idx, opts.end_idx

    data_path = opts.input
    whole_pack = np.load(data_path)
    shifted = opts.shifted

    attack_window = opts.attack_window
    if attack_window:
        tmp = attack_window.split('_')
        start_idx, end_idx = int(tmp[0]), int(tmp[1])
        attack_window = [start_idx, end_idx]

    if shifted:
        print('data will be shifted in range: ', [0, shifted])
        traces, labels, text_in, key, inp_shape = process_data.process_raw_data_shifted(whole_pack, target_byte, network_type, shifted, attack_window)
    else:
        traces, labels, text_in, key, inp_shape = process_data.process_raw_data(whole_pack, target_byte, network_type, attack_window)
    if opts.max_trace_num:
        traces = traces[:opts.max_trace_num]
        labels = labels[:opts.max_trace_num]
    print('training with {:d} traces'.format(opts.max_trace_num))
    return traces, labels, inp_shape


# Training high level function
def train_model(X_profiling, Y_profiling, model, save_file_name, epochs=150, batch_size=100, verbose=False):
    # check modelDir existence
    check_file_exists(os.path.dirname(save_file_name))

    # Save model every epoch
    log_dir = "logs/train_id_fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorBoard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpointer = ModelCheckpoint(save_file_name, monitor='val_accuracy', verbose=verbose, save_best_only=True, mode='max')
    lr_scheduler = OneCycleLR(
                    max_lr=5e-3, end_percentage=0.2, scale_percentage=0.1,
                    maximum_momentum=None,
                    minimum_momentum=None, verbose=True
                )
    earlyStopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)
    #callbacks = [checkpointer, lr_scheduler]
    callbacks = [checkpointer]

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape
    if isinstance(input_layer_shape, list):
        input_layer_shape = input_layer_shape[0]

    # Sanity check
    Reshaped_X_profiling = process_data.sanity_check(input_layer_shape, X_profiling)

    clsNum = len(set(Y_profiling))
    Y_profiling = to_categorical(Y_profiling, clsNum)
    history = model.fit(x=Reshaped_X_profiling, y=Y_profiling,
                        validation_split=0.1, batch_size=batch_size,
                        verbose=verbose, epochs=epochs,
                        shuffle=True, callbacks=callbacks)

    print('model save to path: {}'.format(save_file_name))
    return history


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-tb', '--target_byte', type=int, default=0, help='default value is 0')
    parser.add_argument('-nt', '--network_type', choices={'mlp', 'cnn', 'cnn2', 'wang', 'hw_model'}, help='')
    parser.add_argument('-s', '--shifted', type=int, default=0, help='')
    parser.add_argument('-aw', '--attack_window', default='', help='overwrite the attack window')
    parser.add_argument('-mtn', '--max_trace_num', type=int, default=0, help='')
    opts = parser.parse_args()
    return opts


class test_opts():
    def __init__(self):
        self.input = "/home/mabon/old/complete/OneDrive_datasets/original/xmega_unmasked/X1_K1_200k.npz"
        self.output = "./trained_model/unmasked_xmega_rs"
        self.verbose = 1
        self.target_byte = 2
        self.network_type = "hw_model"
        self.shifted = 0
        self.attack_window = "1800_2800"
        self.max_trace_num = 10000


if __name__ == "__main__":
    # opts = parseArgs(sys.argv)
    opts = test_opts()
    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # get the params
    network_type = opts.network_type
    resDir = opts.output
    verbose = opts.verbose
    target_byte = opts.target_byte

    # make resDir and modelDir
    modelDir = os.path.join(resDir, '{}_dir'.format(network_type))
    os.makedirs(modelDir, exist_ok=True)

    dataset_name = os.path.basename(os.path.dirname(opts.input))
    # model_save_file = os.path.join(modelDir, 'pruned_traceNum_{}_model_{}_dataset_{}_targetbyte_{}.hdf5'.format(opts.max_trace_num, network_type, dataset_name, target_byte))
    modelDir_op='/home/mabon/Tiny_power/models/original/xmega/hw/X1/pruned'
    model_save_file = os.path.join(modelDir, 'best_model_{}_trace_{}.hdf5'.format(opts. attack_window, opts.max_trace_num))
    # set all the params
    epochs = 100
    batch_size = 100

    # get the data and model
    #load traces
    X_profiling, Y_profiling, input_shape = load_training_data(opts)
    print('trace data shape is: ', X_profiling.shape)

    r = [0.6]*7
    model_file='/home/mabon/Tiny_power/models/original/xmega/hw/X1/model/best_model.h5'
    model_pruned= model_zoo.cnn_best(input_shape, r, emb_size=9, classification=True)
    model = load_model(model_file)
    ranks_path='/home/mabon/Tiny_power/l2_out/X1/unmasked_xmega_cnn_l2_idx.csv'
    model_pruned=model_zoo.copy_weights(model, model_pruned, ranks_path)
    # model = model_zoo.create_hamming_weight_model(input_shape)
    model_pruned.summary()

    if 'hw_model' == network_type:
        print('now train dnn model for HW leakage model over {} dataset...'.format(opts.input))
    else:
        print('now train dnn model for ID leakage model over {} dataset...'.format(opts.input))
    history = train_model(X_profiling, Y_profiling, model_pruned, model_save_file, epochs, batch_size, verbose)
