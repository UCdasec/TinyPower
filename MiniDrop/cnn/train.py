#!/usr/bin python3.6
import os
import sys
import argparse
import pdb
import h5py
import time

import tensorflow as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

sys.path.append('tools')
import loadData
import model_zoo


def plot_figure(x, y, fig_save_path, title_str, xlabel, ylabel):
    plt.title(title_str)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig(fig_save_path)
    plt.show(block=False)


def print_run_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print('[LOG -- RUN TIME] -- current function [{}] run time is {:f}'.format(func.__name__, end-start))
    return wrapper


def load_training_data(opts):
    '''data loading function'''
    target_byte = opts.target_byte
    leakage_model = opts.leakage_model
    data_path = opts.input
    trace_num = opts.trace_num
    method = opts.preprocess
    attack_window = opts.attack_window

    whole_pack = np.load(data_path)
    traces, text_in, key = loadData.load_data_base(whole_pack, attack_window, method, trace_num=trace_num, shifted=0)
    
    '''
    #the following code is for downsampling for training
    n1=[]
    for i in range(0,int((np.shape(traces)[0]))):
        t1=[]
        for j in range(0,4000,4):
            t1.append(traces[i][j])
        n1.append(t1)
    traces=np.array(n1)
    '''


    labels = loadData.get_labels(text_in, key[target_byte], target_byte, leakage_model)

    inp_shape = (traces.shape[1], 1)
    loadData.data_info(traces.shape, text_in.shape, key)

    clsNum = 9 if 'HW' == leakage_model else 256
    print('[LOG] -- class number is: ', clsNum)
    labels = to_categorical(labels, clsNum)

    return traces, labels, inp_shape, clsNum


# Training high level function
@print_run_time
def train_model(modelDir, X_profiling, Y_profiling, model, epochs, batch_size=100, verbose=False):
    ''' train the model '''
    # make resDir and modelDir
    model_save_file = os.path.join(modelDir, 'best_model.h5')

    # Save model every epoch
    checkpointer = ModelCheckpoint(model_save_file, monitor='val_accuracy', verbose=verbose, save_best_only=True, mode='max')
    callbacks = [checkpointer]

    # Get the input layer shape and Sanity check
    input_layer_shape = model.get_layer(index=0).input_shape
    if isinstance(input_layer_shape, list):
        input_layer_shape = input_layer_shape[0]
    Reshaped_X_profiling = loadData.sanity_check(input_layer_shape, X_profiling)

    hist = model.fit(x=Reshaped_X_profiling, y=Y_profiling,
                     validation_split=0.1, batch_size=batch_size,
                     verbose=verbose, epochs=epochs,
                     shuffle=True, callbacks=callbacks)

    print('[LOG] -- model save to path: {}'.format(model_save_file))

    loss_val = hist.history['loss']
    x = list(range(1, len(loss_val)+1))
    fig_save_path = os.path.join(modelDir, 'loss.png')
    title = 'loss curve'
    xlabel = 'loss'
    ylabel = 'epoch'
    plot_figure(x, loss_val, fig_save_path, title, xlabel, ylabel)
    print('{LOG} -- loss figure save to path: ', fig_save_path)


def main(opts):
    # get the params
    leakage_model = opts.leakage_model
    verbose = opts.verbose
    epochs = opts.epochs
    batch_size = 100

    # get the data and model and load traces
    X_profiling, Y_profiling, input_shape, clsNum = load_training_data(opts)
    print('[LOG] -- trace data shape is: ', X_profiling.shape)

    print('[LOG] -- now train dnn model for {} leakage model...'.format(leakage_model))
    best_model = model_zoo.cnn_best(input_shape, emb_size=clsNum, classification=True)
    best_model.summary()

    modelDir = os.path.join(opts.output, 'model')
    os.makedirs(modelDir, exist_ok=True)

    train_model(modelDir, X_profiling, Y_profiling, best_model, epochs, batch_size, verbose)

    print('[LOG] -- all done!')


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-tb', '--target_byte', type=int, default=0, help='default value is 0')
    parser.add_argument('-lm', '--leakage_model', choices={'HW', 'ID'}, help='')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='')
    parser.add_argument('-aw', '--attack_window', default='', help='overwrite the attack window')
    parser.add_argument('-tn', '--trace_num', type=int, default=0, help='')
    parser.add_argument('-pp', '--preprocess', default='', choices={'norm', 'scaling', ''}, help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main(opts)

