import os
import pdb
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from joblib import dump, load
import loadData
import key_rank
import key_rank_hw


def get_x_feat(params, mode):
    # loading feature extractor & extract features for training
    triplet_model_dir = params['triplet_model_path']
    feat_ext_path = os.path.join(triplet_model_dir, 'best_model.h5')
    print('[LOG] -- loading the feature extractor from path: ', feat_ext_path)
    feat_model = load_model(feat_ext_path)

    tune_num = 5000
    test_num = 5000

    if 'tune' == mode:
        x_data, plaintext, key = loadData.load_tuning_data(params, tune_num)
    elif 'test' == mode:
        x_data, plaintext, key = loadData.load_test_data(params, test_num)

    x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], 1))
    x_data = feat_model.predict(x_data)
    print('[LOG] -- {} data shape is: {}'.format(mode, x_data.shape))
    return x_data, plaintext, key


def train_one_knn(params, x_train, plaintext, key):
    ''' train and save the knn model  '''
    knn_model_dir = params['knn_model_dir']
    os.makedirs(knn_model_dir, exist_ok=True)
    eval_type = params['eval_type']
    n_neighbors = params['n_neighbors']
    eval_type = params['eval_type']
    target_byte = params['target_byte']
    leakage_model = params['leakage_model']
    attack_window = params['attack_window']

    key_byte = key[target_byte]
    y_train = loadData.get_labels(plaintext, key_byte, target_byte, leakage_model)

    # make sure the data is correct
    print('[LOG] -- using {} leakage model now'.format(leakage_model))
    if 'HW' == leakage_model:
        assert(len(set(y_train))==9)
    elif 'ID' == leakage_model:
        assert(len(set(y_train))==256)
    else:
        raise ValueError()

    knn = KNeighborsClassifier(n_neighbors=10, # n-shot Number of neighbors to use by default for kneighbors queries. n for n-shot learning
                               weights='distance',
                               p=2,  # Power parameter for the Minkowski metric.
                               metric='cosine',  # the distance metric to use for the tree.
                               algorithm='brute'  # Algorithm used to compute the nearest neighbors
                               )
    knn.fit(x_train, y_train)

    acc = accuracy_score(y_train, knn.predict(x_train))
    print('[LOG] -- acc on training data set is: ', acc)

    model_name = 'knn_model.m'
    knn_model_path = os.path.join(knn_model_dir, model_name)
    dump(knn, knn_model_path)
    print('[LOG] -- model for key: {} save to path: {}'.format(key_byte, knn_model_path))
    return knn_model_path


def plot_figure(x, y, model_file_name, dataset_name, fig_save_name):
    plt.title('Performance of ' + model_file_name + ' against ' + dataset_name)
    plt.xlabel('number of traces')
    plt.ylabel('rank')
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig(fig_save_name)
    plt.show(block=False)
    plt.figure()


def compute_knn_ranks(params, x_test, plaintext, key):
    """ generate ranking raw data """
    # get all the params
    knn_model_dir = params['knn_model_dir']
    ranking_curve_dir = params["ranking_curve_dir"]
    ranking_raw_dir = params['ranking_raw_dir']

    eval_type = params['eval_type']
    test_num = params['test_num']
    target_board = params['target_board']
    attack_window = params['attack_window']
    target_byte = params['target_byte']
    leakage_model = params['leakage_model']
    rank_step = params['step_size']

    test_key = key[target_byte]
    y_test = loadData.get_labels(plaintext, test_key, target_byte, leakage_model)

    # trained knn models path
    model_name = 'knn_model.m'
    knn_model_path = os.path.join(knn_model_dir, model_name)
    knn_model = load(knn_model_path)
    print('[LOG] -- k-nn models path: ', knn_model_path)
    print('[LOG] -- making predictions of guessed key and its corresopnding k-nn models ...')

    if 'HW' == leakage_model:
        full_ranks = key_rank_hw.full_ranks
    elif 'ID' == leakage_model:
        full_ranks = key_rank.full_ranks
    else:
        raise ValueError()

    acc = accuracy_score(y_test, knn_model.predict(x_test))
    print('[LOG] -- knn model: {}, real key is: {} test acc is: {}'.format(knn_model_path, key[target_byte], acc))

    # full_ranks(model, dataset, key, plaintext_attack, min_trace_idx, max_trace_idx, target_byte, rank_step):
    min_trace_idx = 0
    max_trace_idx = 100
    pdb.set_trace()
    predictions = knn_model.predict_proba(x_test)
    tmp = predictions[:20, :]
    y_tmp = predictions[:20]
    p_tmp = [np.argmax(tmp[i]) for i in range(len(tmp))]
    ranks = full_ranks(predictions, x_test, key, plaintext, min_trace_idx, max_trace_idx, target_byte, rank_step)
    x = [ranks[k][0] for k in range(0, ranks.shape[0])]
    y = [ranks[k][1] for k in range(0, ranks.shape[0])]

    tmp_name = "{}-key-{}-{}-rank-raw.npy".format(target_board, test_key, leakage_model)
    ranks_file_path = os.path.join(ranking_raw_dir, tmp_name)
    np.savez(ranks_file_path, x=x, y=y)
    print('[LOG] -- saving raw rank data to npy file: ', ranks_file_path)

    # plot the ranking curve
    model_file_name = os.path.basename(params['triplet_model_path'])
    dataset_name = os.path.basename(params['test_data_path']).split('.')[0]
    fig_save_name = os.path.join(ranking_curve_dir, 'ranking_curve.png')
    plot_figure(x, y, model_file_name, dataset_name, fig_save_name)
    print('[LOG] -- saving ranking figure to file: ', fig_save_name)

    print('[LOG] -- all done!')
