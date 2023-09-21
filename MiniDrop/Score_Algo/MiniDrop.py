# Import necessary modules and packages
from __future__ import division  # Ensure division behaves as expected in Python 2 and 3
import pandas as pd
import numpy as np
import pathlib
import os
import sys
import argparse  # Import argument parsing library

# Get the current working directory
cur_path = pathlib.Path().absolute()

# Define a function for predefined pruning based on a threshold
def predefined_pr(rank_result):
    arch = list()
    for r in rank_result:
        threshold = r[1] * 0.8  # Define a threshold for pruning
        t = np.where(r > threshold)
        arch.append(float("{:.2f}".format(len(t[0]) / np.count_nonzero(~np.isnan(r)))))
    print(arch)

# Define a function for gratitude-based pruning
def gratitude_pr(rank_result, n):
    gratitude = list()
    pruning_rate = list()
    idxs = list()
    for rank in rank_result:
        rank = rank[1:]
        gra = list()
        for idx, r in enumerate(rank):
            if idx == len(rank) - n:
                break
            try:
                g = (rank[idx + n] - r) / n  # Calculate gratitude
                gra.append(g)
            except IndexError:
                pass

        gratitude.append(np.array(gra))
    for gra in gratitude:
        for idx, g in enumerate(gra):
            if g == max(gra):
                idxs.append(int(idx + n / 2))  # Find the index with the maximum gratitude
                pruning_rate.append(float("{:.2f}".format(1 - (int(idx + n / 2)) / len(gra))))  # Calculate pruning rate
                break
    for i in range(len(pruning_rate)):
        if pruning_rate[i] > 0.9:
            pruning_rate[i] = 0.9
        elif pruning_rate[i] < 0:
            pruning_rate[i] = 0

    print(idxs)
    print(pruning_rate)
    # Convert the list to a numpy array
    pruning_rate = np.array(pruning_rate)
    # Save the numpy array to a CSV file
    np.savetxt(os.path.join(opts.output, "1-pr.csv"), pruning_rate, delimiter=',')

# Define a function to parse command line arguments
def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--rank_path', help='Path to the rank result file')
    parser.add_argument('-o', '--output', help='Output directory for saving results')
    parser.add_argument('-N', '--n', type=int, default=10, help='Parameter N for gratitude-based pruning')
    opts = parser.parse_args()
    return opts

if __name__ == '__main__':
    opts = parseArgs(sys.argv)
    rank_result = pd.read_csv(opts.rank_path, header=None).values
    rr = list()
    for r in rank_result:
        r = r[~np.isnan(r)]
        rr.append(r)
    os.makedirs(opts.output, exist_ok=True)

    # Call gratitude_pr function with the rank results and specified parameter N
    gratitude_pr(rr, opts.n)
