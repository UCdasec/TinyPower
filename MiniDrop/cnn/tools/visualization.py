import os
import sys
import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import permutations
import seaborn as sns
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import load_model

import tools.loadData as loadData
import mytools.tools as mytools


# load up data
def process_data(fpath, target_byte, leakage_model, sample_num=100):
    wholePack = np.load(fpath)
    method = ''
    traces, plaintext, key = loadData.load_data_base(wholePack, method)
    key_byte = key[target_byte]
    labels = loadData.get_labels(plaintext, key_byte, target_byte, leakage_model)

    datas, labels = mytools.limitData(traces, labels, sampleLimit=sample_num)
    # shuffle data
    datas, labels = mytools.shuffleData(datas, labels)

    # delete all useless data to save memory
    del wholePack
    return datas, labels


def visualize(x_data, y_data, outpath, titleStr=''):
    '''t-SNE'''
    tsne = TSNE(n_components=2, init='pca', random_state=47)
    X_tsne = tsne.fit_transform(x_data)

    print("Org data dimension is {}. Embedded data dimension is {}".format(x_data.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))

    # plot the label on the dots
    label_set = set()
    for i in range(X_norm.shape[0]):
        if y_data[i] not in label_set:
            label_set.add(y_data[i])
            plt.text(X_norm[i, 0], X_norm[i, 1], str(y_data[i]), color='b', fontdict={'weight': 'bold', 'size': 20})
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y_data, cmap='viridis', marker='o')

    plt.xticks([])
    plt.yticks([])

    if titleStr:
        tmp = titleStr.split('_')
        titleStr = ' '.join(tmp)
        plt.title(titleStr)

    plt.savefig(outpath)
    plt.show()
    plt.close()


def run(dpath, target_byte, leakage_model, sampleNum, model_path, outpath, method, attack_window):
    # show original data
    x_data, y_data = process_data(dpath, target_byte, leakage_model, sampleNum)
    x_data = x_data[:, attack_window[0]:attack_window[1]]
    x_data = loadData.preprocess_data(x_data, method)
    x_data = x_data[:, :, np.newaxis]

    # show triplet data
    model_path = os.path.join(model_path, 'best_model.h5')
    model = load_model(model_path)
    feats = model.predict(x_data)

    visualize(feats, y_data, outpath)
    print('figure save to path: {}'.format(outpath))

