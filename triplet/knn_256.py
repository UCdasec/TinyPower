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
    print('loading the feature extractor from path: ', feat_ext_path)
    feat_model = load_model(feat_ext_path)
    if 'tune' == mode:
        x_train, plaintext = loadData.load_tuning_data(params)

        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_train = feat_model.predict(x_train)
        print('{lOG} -- x train feat shape is: ', x_train.shape)
        return x_train, plaintext
    elif 'test' == mode:
        x_test, plaintext, real_key = loadData.load_test_data(params)

        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        x_test = feat_model.predict(x_test)
        print('[LOG] -- shape of x_test: ', x_test.shape)
        return x_test, plaintext, real_key


def train_one_knn(key, params, x_train, plaintext):
    ''' train and save the knn model  '''
    knn_model_dir = params['knn_model_dir']
    os.makedirs(knn_model_dir, exist_ok=True)
    eval_type = params['eval_type']
    n_neighbors = params['n_neighbors']
    eval_type = params['eval_type']
    target_byte = params['target_byte']
    leakage_model = params['leakage_model']
    attack_window = params['attack_window']

    y_train = loadData.get_labels(plaintext, key, target_byte, leakage_model)

    # make sure the data is correct
    print('[LOG] -- using {} leakage model now'.format(leakage_model))
    if 'HW' == leakage_model:
        assert(len(set(y_train))==9)
    elif 'ID' == leakage_model:
        assert(len(set(y_train))==256)
    else:
        raise ValueError()

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, # n-shot Number of neighbors to use by default for kneighbors queries. n for n-shot learning
                               weights='distance',
                               p=2,  # Power parameter for the Minkowski metric.
                               metric='cosine',  # the distance metric to use for the tree.
                               algorithm='brute'  # Algorithm used to compute the nearest neighbors
                               )
    knn.fit(x_train, y_train)

    acc = accuracy_score(y_train, knn.predict(x_train))
    print('[LOG] -- acc on training data set is: ', acc)

    model_name = 'knn-key-{}.model'.format(key)
    knn_model_path = os.path.join(knn_model_dir, model_name)
    dump(knn, knn_model_path)
    print('model for key: {} save to path: {}'.format(key, knn_model_path))
    return knn_model_path


def compute_knn_ranks(data_params, test_key, x_test, plaintext, real_key):
    """ generate ranking raw data """
    # get all the params
    knn_model_dir = data_params['knn_model_dir']
    ranking_curve_dir = data_params["ranking_curve_dir"]
    ranking_raw_dir = data_params['ranking_raw_dir']

    eval_type = data_params['eval_type']
    test_num = data_params['test_num']
    target_board = data_params['target_board']
    attack_window = data_params['attack_window']
    target_byte = data_params['target_byte']
    leakage_model = data_params['leakage_model']
    rank_step = data_params['step_size']

    y_test = loadData.get_labels(plaintext, test_key, target_byte, leakage_model)

    # trained knn models path
    model_name = 'knn-key-{}.model'.format(test_key)
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
    print('[LOG] -- knn model: {}, real key is: {} test acc is: {}'.format(knn_model_path, real_key, acc))

    '''
    # get the ave ranking by run it for multiple times
    ave_x, ave_y = [], []
    max_trace_idx = x_test.shape[0]

    print('[LOG] -- generating ranks and accuracies for guess key: ', test_key)
    for i in range(5):
        # generate predictions on the test dataset
        tmp_x_test, tmp_y_test, tmp_text = shuffle(x_test, y_test, plaintext)

        # full_ranks(model, dataset, key, plaintext_attack, min_trace_idx, max_trace_idx, target_byte, rank_step):
        ranks = full_ranks(feat_model, tmp_x_test, test_key, tmp_text, 0, max_trace_idx, target_byte, rank_step)

        x = [ranks[k][0] for k in range(0, ranks.shape[0])]
        y = [ranks[k][1] for k in range(0, ranks.shape[0])]
        ave_x.append(x)
        ave_y.append(y)

    ave_y = np.array(ave_y)
    ave_y = np.mean(ave_y, axis=0)

    tmp_name = "{}-key-{}-{}-rank-raw.npy".format(target_board, test_key, leakage_model)
    ranks_file_path = os.path.join(ranking_raw_dir, tmp_name)
    np.savez(ranks_file_path, x=x, y=ave_y)
    print('[LOG] -- saving raw rank data to npy file: ', ranks_file_path)
    print('#'*90)

    print('all done!')
    '''
    return acc
