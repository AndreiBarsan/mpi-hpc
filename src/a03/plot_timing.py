"""Plots the timing output produced by spline_2d_problem.cpp."""
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    root = sys.argv[1]
    print("Reading timing data from root dir: {}".format(root))

    files = [os.path.join(root, f) for f in os.listdir(root) if 'timing-' in f]

    df_chunks = []
    for fpath in files:
        fname = fpath.split('/')[-1]
        print(fname)
        pat = "timing-(.*)-(\d+)-proc-(\d+)-rep.csv"
        res = re.match(pat, fname)
        method = res.group(1)
        np = int(res.group(2))
        repetitions = int(res.group(3))

        with open(fpath, 'r') as f:
            df_part = pd.read_csv(f)

        df_part['method'] = method
        df_part['np'] = np

        df_chunks.append(df_part)

    df = pd.concat(df_chunks)


    a = df[df['method'] == 'parallel-deboor-b']
    a = a.sort_values(by=['np'])

    # b = df[df['method'] == 'parallel-deboor-b']

    ax = a[a['n'] == 30].plot('np', 'mean_ms')
    ax = a[a['n'] == 62].plot('np', 'mean_ms', ax=ax)
    ax = a[a['n'] == 126].plot('np', 'mean_ms', ax=ax)
    ax = a[a['n'] == 254].plot('np', 'mean_ms', ax=ax)
    ax = a[a['n'] == 510].plot('np', 'mean_ms', ax=ax)



    plt.show()


if __name__ == '__main__':
    main()
