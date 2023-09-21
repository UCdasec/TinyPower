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


# here store the data for stm32f3 unmasked

def load_data(opts):
    ''' load all the pre-calculate data and draw figures '''
    trace_num = opts.trace_num
    x = ['50', '100', '150', '200', '250', '300']
    if '1k' == trace_num:
        acc = [0.3915, 0.5083, 0.4959, 0.5092, 0.5633, 0.5687]
        training_time = [515, 1739, 3004, 4629, 5134, 6040]
    elif '2k' == trace_num:
        acc = [0.5061, 0.5088, 0.5369, 0.5103, 0.5688, 0.5578]
        training_time = [623, 1933, 3997, 6854, 8356, 9454]
    else:
        raise KeyError('[ERROR] -- key value not defined yet.')
    return x, acc, training_time


def plot_figure(all_x, all_y, fig_save_name, marker_num, marker_size):
    xlabel = 'Per class threshold'
    ylabel1 = 'Accuracy'
    ylabel2 = 'Training time (seconds)'
    showGrid = False

    xTicks, y1Ticks = 0, 0
    y2Ticks = [0, 2000, 4000, 6000, 8000, 10000]
    y2TickLabels = ['0', '2k', '4k', '6k', '8k', '10k']
    xLim, yLim = 0, [0.2, 0.8]

    data = draw_base.DataArray(all_x, all_y)

    # default value: LABELSIZE=26, LEGENDSIZE=26, TICKSIZE=26, LWidth=2
    LABELSIZE = 30
    LEGENDSIZE = 23
    TICKSIZE = 26
    LWidth = 2

    label1 = 'Accuracy'
    label2 = ''
    label3 = 'Training time'
    context = draw_base.Context(xlabel, ylabel1, ylabel2, file2save=fig_save_name, y1Lim=yLim, xLim=xLim,
                                label1=label1, label2=label2, label3=label3, xTicks=xTicks, y1Ticks=y1Ticks,
                                showGrid=showGrid, MSize=marker_size, marker_num=marker_num,
                                LABELSIZE=LABELSIZE, LEGENDSIZE=LEGENDSIZE, TICKSIZE=TICKSIZE,
                                LWidth=LWidth, y2Ticks=y2Ticks, y2TickLabels=y2TickLabels)

    #loc_pos = (0, 1.05)
    loc_pos = 0
    draw_base.draw_results(data, 2, context, loc_pos=loc_pos)


# Check a saved model against one of the testing databases Attack traces
def main(opts):
    # prepare the saving path
    outDir = os.path.join(opts.output, 'acc_compare')
    os.makedirs(outDir, exist_ok=True)
    fig_save_name = os.path.join(outDir + 'acc_cmp.png')
    data_save_name = os.path.join(outDir, 'acc_cmp.npz')

    if os.path.isfile(data_save_name):
        whole_pack = np.load(data_save_name)
        x, y1, y2 = whole_pack['x'], whole_pack['y1'], whole_pack['y2']
        print('{LOG} -- the acc data save to path: ', data_save_name)
    else:
        x, y1, y2 = load_data(opts)
        np.savez(data_save_name, x=x, y1=y1, y2=y2)

    #y_rand = [0.1963] * len(x)

    all_x = [x, [], x]
    all_y = [y1, [], y2]
    plot_figure(all_x, all_y, fig_save_name, opts.marker_num, opts.marker_size)
    print('[LOG] -- figure save to file: {}'.format(fig_save_name))


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-tn', '--trace_num', help='')
    parser.add_argument('-mn', '--marker_num', type=int, default=3, help='')
    parser.add_argument('-ms', '--marker_size', type=int, default=12, help='')
    opts = parser.parse_args()
    return opts


if __name__=="__main__":
    opts = parseArgs(sys.argv)
    main(opts)
