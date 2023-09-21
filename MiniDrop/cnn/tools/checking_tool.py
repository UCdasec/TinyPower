#!/usr/bin/python3
import os
import sys
import pdb
import random

from tensorflow.keras.models import load_model
import model_zoo


def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


def create_empty_model(inp_shape, clsNum=256):
    # clsNum = 256    # because we are using the identical power model
    params = model_zoo.generate_params()
    model = model_zoo.create_power_model(inp_shape, clsNum, params)
    model.summary()
    return model, params


def shuffleTheData(traces, textins, labels):
    '''shuffle the data'''
    tuple_list = []
    for i in range(traces.shape[0]):
        one_trace, one_text, one_label = traces[i,:], textins[i], labels[i]
        tmp_tuple = (one_trace, one_text, one_label)
        tuple_list.append(tmp_tuple)

    # shuffle the list
    tuple_list = random.shuffle(tuple_list)

    # change list back to data mat
    new_traces, new_textins, new_labels = [], [], []
    for i in range(len(tuple_list)):
        tmp_tuple = tuple_list[i]
        one_trace, one_text, one_label = tmp_tuple[0], tmp_tuple[1], tmp_tuple[2]
        new_traces.append(one_trace)
        new_textins.append(one_text)
        new_labels.append(one_label)

    new_traces, new_textins, new_labels = np.array(new_traces), np.array(new_textins), np.array(new_labels)
    return new_traces, new_textins, new_labels


def get_model(network_type, input_shape, emb_size, classification):
    # get network type
    if network_type == "mlp":
        best_model = model_zoo.mlp_best(emb_size, classification)
    elif network_type == "cnn1":
        best_model = model_zoo.cnn_best1(input_shape, emb_size, classification)
    elif network_type == "cnn2":
        best_model = model_zoo.cnn_best2(input_shape, emb_size, classification)
    elif network_type == "wang":
        best_model = model_zoo.create_power_model(input_shape, emb_size, classification)
    elif network_type == "hw_model":
        best_model = model_zoo.create_hamming_weight_model(input_shape)
    else:       # display an error and abort
        raise ValueError("Error: no topology found for network {}...".format(network_type))
    return best_model


def load_best_model(model_file):
    check_file_exists(model_file)
    print('loading pre-trained model...')
    model = load_model(model_file)
    return model
