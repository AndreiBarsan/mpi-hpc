"""Plots the timing output produced by spline_2d_problem.cpp."""
import json
import os
import re
import sys
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    root = sys.argv[1]
    print("Reading timing data from root dir: {}".format(root))
    # methods = ['parallel-deboor-a', 'parallel-deboor-b']
    methods = ['serial-deboor']
    nps = [1] #, 2, 4, 8, 16]

    files = [os.path.join(root, f) for f in os.listdir(root) if 'timing-' in f]

    df_chunks = []
    for fpath in files:
        fname = fpath.split('/')[-1]
        pat = "timing-(.*)-(\d+)-proc-(\d+)-rep.csv"
        res = re.match(pat, fname)
        method = res.group(1)
        if method not in methods:
            continue

        np = int(res.group(2))
        repetitions = int(res.group(3))

        with open(fpath, 'r') as f:
            print("Will read {}".format(fpath))
            df_part = pd.read_csv(f)

        df_part['method'] = method
        df_part['np'] = np
        df_chunks.append(df_part)

    df = pd.concat(df_chunks)
    # for method in methods:
    #     a = df[df['method'] == method]
    #     a = a.sort_values(by=['np'])
    #
    #     plt.figure(figsize=(7.5, 3.5))
    #     plt.subplot(1, 2, 1)
    #     ax = plt.gca()
    #     for n in [30, 62, 126, 254]:
    #         ax = plot_time(a, n, ax)
    #
    #     ax.set(
    #         xlabel="$np$ (number of processors)",
    #         ylabel="Time (ms)",
    #         ylim=(0, 5) if '-a' in method else (0, 10)
    #     )
    #     plt.title("Smaller problem sizes")
    #     plt.xticks(nps)
    #
    #     plt.subplot(1, 2, 2)
    #     ax = plt.gca()
    #     for n in [254, 510, 1022]:      # 2046 (but same trend)
    #         ax = plot_time(a, n, ax=ax)
    #     ax.set(
    #         xlabel="$np$ (number of processors)",
    #         ylabel="Time (ms)",
    #         ylim=(0, 100) if '-a' in method else (0, 200)
    #     )
    #     plt.title("Larger problem sizes")
    #     plt.xticks(nps)
    #     # plt.suptitle("Results for method: {}".format(method))
    #     plt.tight_layout()
    #
    #     plot_fpath_base = os.path.join(root, 'plot-a03p02-{}'.format(method))
    #     plt.savefig(plot_fpath_base + '.png')
    #     plt.savefig(plot_fpath_base + '.eps')

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df)

    plt.show()

    # ns = [30, 62, 126, 254, 510]
    ns = [8, 16, 32, 64, 128, 256]
    print("LaTeX table:")

    print("\\toprule")
    print("n = m & $\\alpha$ (nodes) & $\\alpha$ (grid) & ms & $\\beta$ (nodes) & $\\beta$ (grid) & ms \\\\")
    print("\\midrule")
    prev_se_err = None
    prev_se_err_dense = None
    prev_n = None
    ratios = []
    ratios_dense = []
    for n in ns:
        print("{} & ".format(n), end="")

        v = df[df['n'] == n]
        val = v[v['problem'] == 'quad']
        # print(val['err_nodes'])
        err = list(val['err_nodes'])[0]
        err_dense = list(val['err_dense'])[0]

        print("{:.4e} & ".format(err), end="")
        print("{:.4e} & ".format(err_dense), end="")
        print("{:.4f} & ".format(list(val['mean_ms'])[0]), end="")

        val = v[v['problem'] == 'sin-exp']

        se_err = list(val['err_nodes'])[0]
        se_err_dense = list(val['err_dense'])[0]

        if prev_se_err is not None:
            ratio = -math.log(prev_se_err / se_err) / math.log(prev_n / n)
            ratio_dense = -math.log(prev_se_err_dense / se_err_dense) / math.log(prev_n / n)
            ratios.append(ratio)
            ratios_dense.append(ratio_dense)

        prev_se_err = se_err
        prev_se_err_dense = se_err_dense
        prev_n = n

        print("{:.4e} & ".format(se_err), end="")
        print("{:.4e} & ".format(se_err_dense), end="")

        print("{:.4f} ".format(list(val['mean_ms'])[0]), end="")
        # val = v[v['problem'] == 'sin-exp']
        # print("{} & ".format(val['err_nodes'][0]), end="")
        # print("{} & ".format(val['err_dense'][0]), end="")
        print("\\\\")

    print("\\bottomrule")
    print("Ratios: {}".format(ratios))
    print("Ratios dense: {}".format(ratios_dense))
    return

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
