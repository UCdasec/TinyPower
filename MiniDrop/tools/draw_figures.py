#!/usr/bin/python3
import os
import sys
import argparse
import pdb
import h5py
import statistics
import math

import numpy as np
from collections import defaultdict
import ast
import matplotlib.pyplot as plt

import mytools.draw_base as draw_base


def load_data(input, opts):
    ''' load all the pre-calculate data and draw figures '''
    target_byte = opts.target_byte
    network_type = opts.network_type
    data_path = input
    shifted = opts.shifted

    if not os.path.isfile(data_path):
        raise ValueError('file did not find: {}'.format(data_path))
    whole_pack = np.load(data_path)

    attack_window = opts.attack_window
    if attack_window:
        tmp = opts.attack_window.split('_')
        start_idx, end_idx = int(tmp[0]), int(tmp[1])
        attack_window = [start_idx, end_idx]

    if shifted:
        print('data will be shifted in range: ', [0, shifted])
    traces, labels, text_in, key, inp_shape = process_data.process_raw_data(whole_pack, target_byte, network_type, attack_window)
    return traces, labels, text_in, key, inp_shape


def plot_figure(all_x, all_y, fig_save_name, method, marker_num, marker_size):
    xlabel = 'Number of traces'
    ylabel = 'Mean rank'
    showGrid = False

    if method == 'cnn':
        xTicks, yTicks = 0, [0, 64, 128, 192, 256]
        xLim, yLim = 0, [0, 256]
    else:
        xTicks, yTicks = 0, [0, 64, 128, 192, 256]
        xLim, yLim = 0, [0, 256]

    data = draw_base.DataArray(all_x, all_y)

    # default value: LABELSIZE=26, LEGENDSIZE=26, TICKSIZE=26, LWidth=2
    LABELSIZE = 30
    LEGENDSIZE = 23
    TICKSIZE = 26
    LWidth = 2

    label1 = 'opt 1'
    label2 = 'opt 2'
    label3 = 'opt 3'
    label4 = 'obfuscation'
    label5 = ''

    context = draw_base.Context(xlabel, ylabel, file2save=fig_save_name, y1Lim=yLim, xLim=xLim,
                                label1=label1, label2=label2, label3=label3, label4=label4,
                                label5=label5, xTicks=xTicks, y1Ticks=yTicks,LWidth=LWidth,
                                showGrid=showGrid, MSize=marker_size, marker_num=marker_num,
                                LABELSIZE=LABELSIZE, LEGENDSIZE=LEGENDSIZE, TICKSIZE=TICKSIZE)

    draw_base.draw_results(data, 1, context)


# Check a saved model against one of the testing databases Attack traces
def main(opts):
    # prepare the saving path
    outDir = os.path.dirname(opts.output)
    os.makedirs(outDir, exist_ok=True)
    fig_save_name = os.path.join(opts.output + '.png')
    print('figure save to file: {}'.format(fig_save_name))

    curve_num = len(opts.input)
    max_points = opts.max_points

    if 1 == curve_num:
        dpack = np.load(opts.input[0])
        x, y = dpack['x'], dpack['y']

        if max_points:
            x = x[:max_points]
            y = y[:max_points]
        all_x = [x]
        all_y = [y]
        plot_figure(all_x, all_y, fig_save_name, opts.method, opts.marker_num, opts.marker_size)
    elif 2 == curve_num:
        dpack1 = np.load(opts.input[0])
        dpack2 = np.load(opts.input[1])
        x1, y1 = dpack1['x'], dpack1['y']
        x2, y2 = dpack2['x'], dpack2['y']

        if max_points:
            x1, x2 = x1[:max_points], x2[:max_points]
            y1, y2 = y1[:max_points], y2[:max_points]

        all_x = [x1, x2]
        all_y = [y1, y2]
        plot_figure(all_x, all_y, fig_save_name, opts.method, opts.marker_num, opts.marker_size)
    elif 3 == curve_num:
        dpack1 = np.load(opts.input[0])
        dpack2 = np.load(opts.input[1])
        dpack3 = np.load(opts.input[2])
        x1, y1 = dpack1['x'], dpack1['y']
        x2, y2 = dpack2['x'], dpack2['y']
        x3, y3 = dpack3['x'], dpack3['y']

        if max_points:
            x1, x2, x3 = x1[:max_points], x2[:max_points], x3[:max_points]
            y1, y2, y3 = y1[:max_points], y2[:max_points], y3[:max_points]

        all_x = [x1, x2, x3]
        all_y = [y1, y2, y3]
        plot_figure(all_x, all_y, fig_save_name, opts.method, opts.marker_num, opts.marker_size)
    elif 4 == curve_num:
        dpack1 = np.load(opts.input[0])
        dpack2 = np.load(opts.input[1])
        dpack3 = np.load(opts.input[2])
        dpack4 = np.load(opts.input[3])
        x1, y1 = dpack1['x'], dpack1['y']
        x2, y2 = dpack2['x'], dpack2['y']
        x3, y3 = dpack3['x'], dpack3['y']
        x4, y4 = dpack4['x'], dpack4['y']

        if max_points:
            x1, x2, x3, x4 = x1[:max_points], x2[:max_points], x3[:max_points], x4[:max_points]
            y1, y2, y3, y4 = y1[:max_points], y2[:max_points], y3[:max_points], y4[:max_points]

        all_x = [x1, x2, x3, x4]
        all_y = [y1, y2, y3, y4]
        plot_figure(all_x, all_y, fig_save_name, opts.method, opts.marker_num, opts.marker_size)
    elif 5 == curve_num:
        dpack1 = np.load(opts.input[0])
        dpack2 = np.load(opts.input[1])
        dpack3 = np.load(opts.input[2])
        dpack4 = np.load(opts.input[3])
        dpack5 = np.load(opts.input[4])
        x1, y1 = dpack1['x'], dpack1['y']
        x2, y2 = dpack2['x'], dpack2['y']
        x3, y3 = dpack3['x'], dpack3['y']
        x4, y4 = dpack4['x'], dpack4['y']
        x5, y5 = dpack5['x'], dpack5['y']

        if max_points:
            x1, x2, x3, x4, x5 = x1[:max_points], x2[:max_points], x3[:max_points], x4[:max_points], x5[:max_points]
            y1, y2, y3, y4, y5 = y1[:max_points], y2[:max_points], y3[:max_points], y4[:max_points], y5[:max_points]

        all_x = [x1, x2, x3, x4, x5]
        all_y = [y1, y2, y3, y4, y5]
        plot_figure(all_x, all_y, fig_save_name, opts.method, opts.marker_num, opts.marker_size)
    else:
        raise ValueError()


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', action='append', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-mp', '--max_points', type=int, default=0, help='')
    parser.add_argument('-m', '--method', choices={'cnn', 'ada'}, help='')
    parser.add_argument('-mn', '--marker_num', type=int, default=3, help='')
    parser.add_argument('-ms', '--marker_size', type=int, default=8, help='')
    opts = parser.parse_args()
    return opts


if __name__=="__main__":
    opts = parseArgs(sys.argv)
    main(opts)
