import os
import sys
import argparse
import random

import numpy as np
from tqdm import tqdm


def load_data(opts):
    whole_pack = np.load(opts.input)
    print('{LOG} -- will load data from path: ', opts.input)
    trace_num = opts.trace_num
    try:
        traces, plain_text, key = whole_pack['power_trace'], whole_pack['plain_text'], whole_pack['key']
    except KeyError:
        try:
            traces, plain_text, key = whole_pack['power_trace'], whole_pack['plaintext'], whole_pack['key']
        except KeyError:
            traces, plain_text, key = whole_pack['trace_mat'], whole_pack['textin_mat'], whole_pack['key']

    if trace_num:
        traces = traces[:trace_num, :]
        plain_text = plain_text[:trace_num, :]

    return traces, plain_text, key


def shift_the_data(traces, max_delay):
    # randomly shift the dataset
    print('[LOG] -- the data will be shifted in range [{}, {}]'.format(-max_delay, max_delay))
    sample_len = traces.shape[1]
    shifted_traces = []
    for i in tqdm(range(traces.shape[0])):
        random_int = random.randint(0, max_delay)
        trace_i = traces[i]
        paddings = np.zeros(abs(random_int))
        trace_i = list(paddings) + list(trace_i)
        trace_i = trace_i[:sample_len]
        shifted_traces.append(trace_i)

    shifted_traces = np.array(shifted_traces)
    return shifted_traces


def main(opts):
    # load the data
    traces, plain_text, key = load_data(opts)

    # shift the data
    shifted_traces = shift_the_data(traces, opts.max_delay)

    # save the data
    old_name = os.path.basename(opts.input).split('.')[0]
    os.makedirs(opts.output, exist_ok=True)
    save_path = os.path.join(opts.output, '{}_delay_{}'.format(old_name, opts.max_delay))
    np.savez(save_path, power_trace=shifted_traces, plain_text=plain_text, key=key)
    print('[LOG] -- data save to path: ', save_path)

    print('[LOG] -- all done!')


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-md', '--max_delay', type=int, help='')
    parser.add_argument('-tn', '--trace_num', type=int, default=5000, help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    main(opts)

