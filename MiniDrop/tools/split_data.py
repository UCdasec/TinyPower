import os
import sys
import numpy as np
import argparse


def loadData(opts):
    fpath = opts.input
    print('load file from file: ', fpath)
    whole_pack = np.load(fpath)

    try:
        trace_array, textin_array, key = whole_pack['power_trace'], whole_pack['plain_text'], whole_pack['key']
    except Exception:
        try:
            trace_array, textin_array, key = whole_pack['trace_mat'], whole_pack['textin_mat'], whole_pack['key']
        except Exception:
            trace_array, textin_array, key = whole_pack['power_trace'], whole_pack['plaintext'], whole_pack['key']

    return trace_array, textin_array, key


def main(opts):
    trace_array, textin_mat, key = loadData(opts)

    trace_array1 = trace_array[:opts.part_1, :]
    trace_array2 = trace_array[opts.part_1:, :]

    textin_mat1 = textin_mat[:opts.part_1, :]
    textin_mat2 = textin_mat[opts.part_1:, :]

    droot = os.path.dirname(opts.input)
    fname = os.path.basename(opts.input).split('.')[0]
    fpath1 = os.path.join(droot, '{}_{}_1'.format(fname, trace_array1.shape[0]))
    fpath2 = os.path.join(droot, '{}_{}_2'.format(fname, trace_array2.shape[0]))

    np.savez(fpath1, trace_mat=trace_array1, textin_mat=textin_mat1, key=key)
    np.savez(fpath2, trace_mat=trace_array2, textin_mat=textin_mat2, key=key)

    print('save data to path: {} and {}'.format(fpath1, fpath2))


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-p1', '--part_1', default=10000, type=int, help='')
    opts = parser.parse_args()
    return opts


if __name__ == '__main__':
    opts = parseArgs(sys.argv)
    main(opts)
