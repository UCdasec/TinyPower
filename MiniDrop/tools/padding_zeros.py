import os
import sys
import argparse
import pdb
import numpy as np


def main(opts):
    print('[LOG] -- load data from file: ', opts.input)
    tmp = opts.attack_window.split('_')
    attack_window = [int(tmp[0]), int(tmp[1])]
    whole_pack = np.load(opts.input)

    attack_window_size = attack_window[1] - attack_window[0]
    if attack_window_size > opts.out_dim:
        print('[LOG] -- out dim is smaller than attack window, please use downsampling instead')
        return

    start_idx, end_idx = attack_window[0], attack_window[1]
    try:
        trace_mat, textin_mat, key = whole_pack['trace_mat'], whole_pack['textin_mat'], whole_pack['key']
    except Exception as e:
        try:
            trace_mat, textin_mat, key = whole_pack['power_trace'], whole_pack['plain_text'], whole_pack['key']
        except Exception as e:
            trace_mat, textin_mat, key = whole_pack['power_trace'], whole_pack['plaintext'], whole_pack['key']

    tmp_mat = trace_mat[:, start_idx:end_idx]

    # now padding zero to the end to have the size
    padding_size = opts.out_dim - attack_window_size
    sample_num = trace_mat.shape[0]
    padding_mat = np.zeros((sample_num, padding_size), dtype=tmp_mat.dtype)
    new_trace_mat = np.concatenate((tmp_mat, padding_mat), axis=1)

    assert(new_trace_mat.shape[1] == opts.out_dim)

    tmp = os.path.basename(opts.input).split('.')[0]
    out_name = '{}_padding_{}.npz'.format(tmp, str(opts.out_dim))
    outpath = os.path.join(opts.output, out_name)
    np.savez(outpath, trace_mat=new_trace_mat, textin_mat=textin_mat, key = key)
    print('[LOG] -- file save to path: ', outpath)



def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-od', '--out_dim', type=int, help='')
    parser.add_argument('-aw', '--attack_window', help='')
    opts = parser.parse_args()
    return opts


if __name__ == '__main__':
    opts = parseArgs(sys.argv)
    main(opts)
