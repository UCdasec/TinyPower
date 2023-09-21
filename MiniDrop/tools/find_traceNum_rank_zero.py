import os
import sys
import argparse
import numpy as np
import pdb


def binary_search(arr):
    # we need to do it in a binary search style
    rtn = len(arr)
    for i in range(len(arr)):
        if arr[i] < 0.5:
            rtn = i
            break
    return rtn


def compute_min_rank(ranking_list):
    ''' try to find the last value that not convergence to 0 '''
    ranking_list = np.array(ranking_list)
    num = binary_search(ranking_list)
    return num


def main(opts):
    dpath = opts.input
    whole_pack = np.load(dpath)
    x, y = whole_pack['x'], whole_pack['y']

    trace_num = binary_search(y)
    print('[LOG] -- for dataset {}, trace num to rank zero is: {}'.format(dpath, trace_num))
    print('all done!')


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    main(opts)

