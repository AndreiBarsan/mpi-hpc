"""Analyzes the experiment data from CSC2306 Assignment 01."""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_csvs(fpaths):
    data = {}
    for fpath in fpaths:
        fname = os.path.basename(fpath)
        n_proc = int(re.split('[-.]+', fname)[1])
        df = pd.read_csv(fpath, index_col=0)
        df.rename(columns=lambda col_name: col_name.strip(), inplace=True)      # Remove spaces around column names
        data['{:03d}'.format(n_proc)] = df['time_s'][5:-5]
    aggregated_df = pd.DataFrame(data)
    aggregated_df = aggregated_df.reindex(sorted(aggregated_df.columns), axis=1)
    return aggregated_df


def main():
    # Assumes the script is ran from the directory in which it is located.
    results_dir = '../../results'
    builtin_fpaths = [os.path.join(results_dir, fname) for fname in os.listdir(results_dir) if 'builtin' in fname]
    manual_fpaths = [os.path.join(results_dir, fname) for fname in os.listdir(results_dir) if 'manual' in fname]

    builtin = parse_csvs(builtin_fpaths)
    manual = parse_csvs(manual_fpaths)
    runs = len(builtin)

    plt.figure(figsize=(8, 6))
    ax = manual.mean().plot(yerr=manual.std(), label="Custom Manual AllReduce")
    ax = builtin.mean().plot(use_index=True, yerr=builtin.std(), label="MPI AllReduce")
    int_keys = [int(x) for x in sorted(builtin.keys())]

    ax.set_xticks(range(len(int_keys)))
    ax.set_xticklabels(int_keys)
    plt.box(False)
    plt.ylim(0.0, 0.15)

    plt.xlabel("p (number of processors)")
    plt.ylabel("Time (s)")
    plt.title(r"Time to add $2^{{25}}$ numbers (average over {} runs)".format(runs))
    plt.legend()
    plt.grid(color=(0.75, 0.75, 0.75, 0.25))
    plt.show()



if __name__ == '__main__':
    main()
