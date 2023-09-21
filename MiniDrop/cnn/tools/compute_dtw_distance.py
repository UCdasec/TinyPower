import os
import sys
import pdb
import argparse

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def test(x=[], y=[], ifTest=False):
    if ifTest:
        x = np.array([1, 2, 3, 3, 7])
        y = np.array([1, 2, 2, 2, 2, 2, 2, 4])

    distance, path = fastdtw(x, y, dist=euclidean)

    print('the distance is: ', distance)
    #print('the path is: ', path)

    # 5.0
    # [(0, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 5), (3, 6), (4, 7)]
    return distance


def loadData(inp):
    print('load file from file: ', inp)
    whole_pack = np.load(inp)
    try:
        trace_array, textin_array, key = whole_pack['power_trace'], whole_pack['plain_text'], whole_pack['key']
    except Exception:
        try:
            trace_array, textin_array, key = whole_pack['trace_mat'], whole_pack['textin_mat'], whole_pack['key']
        except Exception:
            trace_array, textin_array, key = whole_pack['power_trace'], whole_pack['plaintext'], whole_pack['key']

    return trace_array


def get_attack_window(attack_window):
    tmp = attack_window.split('_')
    start_idx = int(tmp[0])
    end_idx = int(tmp[1])
    return [start_idx, end_idx]


def main(opts):
    f1 = opts.input_1
    f2 = opts.input_2

    t1 = loadData(f1)
    t2 = loadData(f2)

    aw1 = get_attack_window(opts.attack_window_1)
    aw2 = get_attack_window(opts.attack_window_2)

    t1 = t1[10, aw1[0]:aw1[1]]
    t2 = t2[10, aw2[0]:aw2[1]]

    test(t1, t2)

    print('all done!')


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--input_1', help='path to the first dataset')
    parser.add_argument('-i2', '--input_2', help='path to the second dataset')
    parser.add_argument('-aw1', '--attack_window_1', help='attack window of dataset 1')
    parser.add_argument('-aw2', '--attack_window_2', help='attack window of dataset 2')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    main(opts)