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
    methods = ['parallel-deboor-a', 'parallel-deboor-b']
    # methods = ['serial-deboor']
    nps = [1, 2, 4, 8, 16]

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
    for method in methods:
        a = df[df['method'] == method]
        a = a.sort_values(by=['np'])

        plt.figure(figsize=(7.5, 3.5))
        plt.subplot(1, 2, 1)
        ax = plt.gca()
        for n in [30, 62, 126, 254]:
            ax = plot_time(a, n, ax)

        ax.set(
            xlabel="$np$ (number of processors)",
            ylabel="Time (ms)",
            ylim=(0, 5) if '-a' in method else (0, 10)
        )
        plt.title("Smaller problem sizes")
        plt.xticks(nps)

        plt.subplot(1, 2, 2)
        ax = plt.gca()
        for n in [254, 510, 1022]:      # 2046 (but same trend)
            ax = plot_time(a, n, ax=ax)
        ax.set(
            xlabel="$np$ (number of processors)",
            ylabel="Time (ms)",
            ylim=(0, 100) if '-a' in method else (0, 200)
        )
        plt.title("Larger problem sizes")
        plt.xticks(nps)
        # plt.suptitle("Results for method: {}".format(method))
        plt.tight_layout()

        plot_fpath_base = os.path.join(root, 'plot-a03p02-{}'.format(method))
        plt.savefig(plot_fpath_base + '.png')
        plt.savefig(plot_fpath_base + '.eps')

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df)

    plt.show()

    ns = [30, 62, 126, 254, 510]
    print("LaTeX table:")

    print("\\toprule")
    print(" np & ", end="")
    for mi, m in enumerate(methods):
        print("\\multicolumn{{ {} }}{{ c }}{{ {} }}".format(len(ns), m))
        if mi != len(methods) - 1:
            print(" & \phantom{xxx} & ")
    print("\\\\")


    for mi, _ in enumerate(methods):
        print(" & ", end="")
        for i, n in enumerate(ns):
            print("$ n = {} $ ".format(n), end="")
            if not (i == len(ns) - 1 and mi == len(methods) - 1):
                print(" & ", end="")
        # print(" & ", end="")

    print("\\\\ \\midrule")

    for np in nps:
        print("{} & ".format(np), end='')

        for mi, method in enumerate(methods):
            mf = df[df['method'] == method]
            for i, n in enumerate(ns):
                v = mf[mf['np'] == np]
                v = v[v['n'] == n]
                if len(v):
                    print("{: 10.2f}ms".format(list(v['mean_ms'])[0]), end="")
                else:
                    print("n/A", end="")
                if not (i == len(ns) - 1 and mi == len(methods) - 1):
                    print(" & ", end="")
            if mi != len(methods) - 1:
                print(" & ", end="")
        print(" \\\\")

    print("\\bottomrule")



    # plt.show()


def plot_time(a, n, ax):
    ax = a[a['n'] == n].plot('np', 'mean_ms',
                             yerr='std_ms', capsize=4,
                             ax=ax, label="$n = m = {}$".format(n))
    return ax


if __name__ == '__main__':
    main()
