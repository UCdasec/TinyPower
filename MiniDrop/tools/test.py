#! /usr/bin/env python
import os
import sys
import argparse
import pdb
import time
from statistics import mean, stdev

import numpy as np

from keras.models import load_model

RootDir = os.getenv('ROOT_DIR')
toolsDir = os.path.join(RootDir, 'tools')
sys.path.append(toolsDir)
import utility
from utility import create_test_set_Wang_disjoint, kNN_accuracy
import mytools.tools as mytools

currentDir = os.path.dirname(__file__)
ResDir = os.path.join(currentDir, 'res_out')
os.makedirs(ResDir, exist_ok=True)


def Wang_Disjoint_Experment(opts, n_shot, data_dim):
    '''
    This function aims to experiment the performance of TF attack
    when the model is trained on the dataset with different distributions.
    The model is trained on AWF777 and tested on Wang100 and the set of
    websites in the training set and the testing set are mutually exclusive.
    '''
    features_model = load_model(opts.modelPath, compile=False)
    # N-MEV is the use of mean of embedded vectors as mentioned in the paper
    if opts.exp_type:
        type_exp = 'N-MEV'
    else:
        type_exp = ''

    # KNeighborsClassifier(n_neighbors=n_shot, weights='distance', p=2, metric='cosine', algorithm='brute')
    params = {'k': n_shot,
              'weights': 'distance',
              'p': 2,
              'metric': 'cosine'
              }

    print("N_shot: ", n_shot)
    acc_list_Top1, acc_list_Top5 = [], []
    for i in range(10):
        signature_dict, test_dict, sites = utility.getDataDict(opts.input, n_shot, data_dim, train_pool_size=20, test_size=70)
        if i == 0:
            size_of_problem = len(list(test_dict.keys()))
            print("Size of Problem: ", size_of_problem)
        signature_vector_dict, test_vector_dict = create_test_set_Wang_disjoint(signature_dict,
                                                                                test_dict,
                                                                                sites,
                                                                                features_model=features_model,
                                                                                type_exp=type_exp)
        # Measure the performance (accuracy)
        acc_knn_top1, acc_knn_top5 = kNN_accuracy(signature_vector_dict, test_vector_dict, params)
        acc_list_Top1.append(float("{0:.15f}".format(round(acc_knn_top1, 5))))
        acc_list_Top5.append(float("{0:.15f}".format(round(acc_knn_top5, 5))))

    print(str(acc_list_Top1).strip('[]'))
    print(str(acc_list_Top5).strip('[]'))
    rtnLine = 'n_shot: {}\tacc for top 1: {} and std is: {}'.format(n_shot, mean(acc_list_Top1), stdev(acc_list_Top1))
    rtnLine = rtnLine + '\nacc for top 5: {} and std is: {}\n\n'.format(mean(acc_list_Top5), stdev(acc_list_Top5))
    print(rtnLine)
    return rtnLine


def run(opts):
    source = os.path.basename(opts.input).split('.')[0]
    outfile = os.path.join(ResDir, 'ccs19_target{}_results.txt'.format(source))
    f = open(outfile, 'a+')
    print('\n\n#################### test time is: {} #########################'.format(time.ctime()), file=f)
    tsnList = [1, 5, 10, 15, 20]
    for tsn in tsnList:
        rtnLine = Wang_Disjoint_Experment(opts, n_shot=tsn, data_dim=5000)
        print(rtnLine, file=f)
    f.close()


class MyOpts():
    def __init__(self, input, modelPath, exp_type, data_dim):
        self.input = input
        self.modelPath = modelPath
        self.exp_type = exp_type
        self.data_dim = data_dim


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-m', '--modelPath', help='')
    parser.add_argument('-t', '--exp_type', action='store_true', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    run(opts)
