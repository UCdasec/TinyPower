from __future__ import division
import pandas as pd
import numpy as np
import pathlib
import os
import sys
import argparse


cur_path = pathlib.Path().absolute()
# from tensorflow.keras.models import load_model
# model = load_model('/home/erc/PycharmProjects/TripletPowerVanilla-master/best_whole_model.h5')
#     # model = load_model(cnn_model_path)
# model.summary()


def predefined_pr(rank_result):

    arch = list()
    for r in rank_result:
        threshold = r[1]*0.8
        t = np.where(r>threshold)
        arch.append(float("{:.2f}".format(len(t[0])/np.count_nonzero(~np.isnan(r)))))
    # a = np.nonzero(rank_result[-2])
    # b = np.nonzero(rank_result[-1])
    # arch.append(float("{:.2f}".format(len(a[0])/4096)))
    # arch.append(float("{:.2f}".format(len(b[0])/4096)))

    print(arch)


def gratitude_pr(rank_result, n):
    # aa = [0.2, 0.09, 0.16, 0.11, 0.05, 0.03, 0.06]
    gratitude = list()
    pruning_rate = list()
    idxs = list()
    for rank in rank_result:
        rank = rank[1:]
        gra = list()
        for idx, r in enumerate(rank):
            if idx == len(rank)-n:
                break
            try:
                g = (rank[idx+n] - r) / n
                gra.append(g)
            except IndexError:
                pass

        gratitude.append(np.array(gra))
    for gra in gratitude:
        for idx, g in enumerate(gra):

          if g == max(gra):
                idxs.append(int(idx + n/2))
                pruning_rate.append(float("{:.2f}".format(1-(int(idx + n/2))/len(gra))))
                break
    for i in range(len(pruning_rate)):
        if(pruning_rate[i]>.9):
            pruning_rate[i]=.9
        elif(pruning_rate[i]<0):
            pruning_rate[i]=0

    print(idxs)
    print(pruning_rate)
    # convert the list to a numpy array
    pruning_rate = np.array(pruning_rate)
    # save the numpy array to a CSV file
    np.savetxt(os.path.join(opts.output, "1-pr.csv"), pruning_rate, delimiter=',')

def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--rank_path', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-N', '--n', type=int, default=10, help='')
    opts = parser.parse_args()
    return opts

if __name__ == '__main__':
    opts = parseArgs(sys.argv)
    rank_result = pd.read_csv(opts.rank_path, header=None).values
    rr = list()
    for r in rank_result:
        r = r[~np.isnan(r)]
        rr.append(r)
    # predefined_pr(rank_result)
    # rank_result = list(rank_result)
    os.makedirs(opts.output, exist_ok=True)

    gratitude_pr(rr, opts.n)