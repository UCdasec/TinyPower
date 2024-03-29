#!/usr/bin python3.6
import os
import sys
#sys.path.append("/home/mabon/Tiny_power/code/TinyPower")
os.sys.path.append('../MiniDrop')

import argparse
import pdb
import h5py
import time

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

    print('[LOG] -- model save to path: {}'.format(save_file_name))
    return history


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-m', '--model_dir', help='')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-cd', '--cross_dev', action='store_true', help='')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='default value is 0')
    parser.add_argument('-tb', '--target_byte', type=int, default=2, help='default value is 2')
    parser.add_argument('-lm', '--network_type', choices={'hw_model', 'ID'}, help='')
    parser.add_argument('-aw', '--attack_window', default='', help='overwrite the attack window')
    parser.add_argument('-pp', '--preprocess', default='', choices={'', 'norm', 'scailing'})
    parser.add_argument('-tn', '--max_trace_num', type=int, default=10000, help='')
    parser.add_argument('-sh', '--shifted', type=int, default=0, help='')
    parser.add_argument('-pr', '--pruning_rate', type=float, default=1, help='')
    parser.add_argument('-rp', '--ranks_path', help='')
    parser.add_argument('-CP', '--custom_pruning',default='False', help='')
    parser.add_argument('-CPP', '--custom_pruning_file',default='False', help='')



    opts = parser.parse_args()
    return opts

def train(opts):
    # get the params
    network_type = opts.network_type
    verbose = opts.verbose
    target_byte = opts.target_byte

    # make resDir and modelDir
    modelDir = os.path.join(opts.output,'model')
    os.makedirs(modelDir, exist_ok=True)
    model_save_file = os.path.join(modelDir, 'best_model.h5'.format(opts. attack_window, opts.max_trace_num))
    dataset_name = os.path.basename(os.path.dirname(opts.input))


    # set all the params
    batch_size = 100

    # get the data and model
    #load traces
    X_profiling, Y_profiling, input_shape = load_training_data(opts)
    print('trace data shape is: ', X_profiling.shape)
    if (opts.custom_pruning=='True'):
        r = np.loadtxt(opts.custom_pruning_file, delimiter=',')
        r = [1 - x for x in r]

    else:
        r = [(1-opts.pruning_rate)]*7

    print(r)

    if opts.network_type=='hw_model':
        emb_size=9
    else:
        emb_size=256

    model_file = os.path.join(opts.model_dir,'model','best_model.h5')
    model_pruned= model_zoo.cnn_best(input_shape, r, emb_size=emb_size, classification=True)
    model = load_model(model_file)
    print("ORiginal Model")
    model.summary()
    model_pruned=model_zoo.copy_weights(model, model_pruned, opts.ranks_path)
    # model = model_zoo.create_hamming_weight_model(input_shape)
    print("Pruned Model")
    model_pruned.summary()

    if 'hw_model' == network_type:
        print('now train dnn model for HW leakage model over {} dataset...'.format(opts.input))
    else:
        print('now train dnn model for ID leakage model over {} dataset...'.format(opts.input))
    t0 = time.time()
    history = train_model(X_profiling, Y_profiling, model_pruned, model_save_file, opts.epochs, batch_size, verbose)
    t1 = time.time()
    total = t1-t0
    print("RUNTIME: ",total)
    print("TIME PER EPOCH",(total/opts.epochs))


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    train(opts)
    
# This code is developed by
# Mabon Ninan and Haipeng Li 
# UC DASEC



