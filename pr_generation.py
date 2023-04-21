from __future__ import division
import pandas as pd
import numpy as np
import pathlib
import os
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


def gratitude_pr(rank_result):
    # aa = [0.2, 0.09, 0.16, 0.11, 0.05, 0.03, 0.06]
    gratitude = list()
    pruning_rate = list()
    idxs = list()
    for rank in rank_result:
        rank = rank[1:]
        gra = list()
        for idx, r in enumerate(rank):
            if idx == len(rank)-10:
                break
            g = (rank[idx+10] - r) / 10
            gra.append(g)

        gratitude.append(np.array(gra))
    for gra in gratitude:
        for idx, g in enumerate(gra):

          if g == max(gra):
                idxs.append(idx+5)
                pruning_rate.append(float("{:.2f}".format(1-(idx+5)/len(gra))))
                break
    print(idxs)
    print(pruning_rate)

if __name__ == '__main__':
    rank_path = os.path.join(cur_path, "xmega_dist.csv")
    rank_result = pd.read_csv(rank_path, header=None).values
    rr = list()
    for r in rank_result:
        r = r[~np.isnan(r)]
        rr.append(r)
    # predefined_pr(rank_result)
    # rank_result = list(rank_result)
    gratitude_pr(rr)