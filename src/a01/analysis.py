"""Analyzes the experiment data from CSC2306 Assignment 01."""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warm_up = 5
cool_down = 5


def parse_csvs(fpaths):
    data = {}
    for fpath in fpaths:
        fname = os.path.basename(fpath)
        n_proc = int(re.split('[-.]+', fname)[2])
        df = pd.read_csv(fpath, index_col=0)
        df.rename(columns=lambda col_name: col_name.strip(), inplace=True)      # Remove spaces around column names
        data['{:03d}'.format(n_proc)] = df['time_s'][warm_up:-cool_down]
    aggregated_df = pd.DataFrame(data)
    aggregated_df = aggregated_df.reindex(sorted(aggregated_df.columns), axis=1)
    return aggregated_df


def fix_df(df):
    return df.reindex(sorted(df.columns), axis=1)


def parse_csvs_p4(root):
    fpaths = [os.path.join(root, dname) for dname in os.listdir(root) if 'e04' in dname]
    data = {}
    for fpath in fpaths:
        fname = os.path.basename(fpath)
        chunks = re.split('[-.]+', fname)
        n_proc = int(chunks[4])
        n = int(chunks[3])
        method = chunks[1]

        df = pd.read_csv(fpath, index_col=0)
        df.rename(columns=lambda col_name: col_name.strip(), inplace=True)      # Remove spaces around column names
        if method not in data:
            data[method] = {}
        if n not in data[method]:
            data[method][n] = {}
        data[method][n]['{:03d}'.format(n_proc)] = df['time_s'][warm_up:-cool_down]

    for m in data.keys():
        data[m] = {n: fix_df(pd.DataFrame(data[m][n])) for n in data[m]}

    return data


def plot_problem_02():
    # Assumes the script is ran from the directory in which it is located.
    results_dir = '../../results'
    builtin_fpaths = [os.path.join(results_dir, fname) for fname in os.listdir(results_dir) if 'builtin-sum' in fname]
    manual_fpaths = [os.path.join(results_dir, fname) for fname in os.listdir(results_dir) if 'manual-sum' in fname]
    builtin_fpaths_multi = [os.path.join(results_dir, fname) for fname in os.listdir(results_dir)
                            if 'builtin-multiple' in fname]
    manual_fpaths_multi = [os.path.join(results_dir, fname) for fname in os.listdir(results_dir)
                           if 'manual-multiple' in fname]

    builtin = parse_csvs(builtin_fpaths)
    manual = parse_csvs(manual_fpaths)
    builtin_multi = parse_csvs(builtin_fpaths_multi)
    manual_multi = parse_csvs(manual_fpaths_multi)
    runs = len(builtin)

    plt.figure(figsize=(8, 6))
    ax = builtin.mean().plot(yerr=builtin.std(), label="MPI AllReduce Sum", capsize=8)
    ax = manual.mean().plot(yerr=manual.std(), label="Custom Manual AllReduce Sum", capsize=8, linestyle='--')
    ax = builtin_multi.mean().plot(yerr=builtin_multi.std(), label="MPI AllReduce Multiple Ops", capsize=8)
    ax = manual_multi.mean().plot(yerr=manual_multi.std(), label="Custom Manual AllReduce Multiple Ops", capsize=8, linestyle='--')
    int_keys = [int(x) for x in sorted(builtin.keys())]

    ax.set_xticks(range(len(int_keys)))
    ax.set_xticklabels(int_keys)
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    plt.ylim(0.0, 0.15)

    plt.xlabel("$p$ (number of processors)")
    plt.ylabel("Time (s)")
    plt.title(r"Time to add $2^{{25}}$ numbers (average over {} runs), in seconds".format(runs))
    plt.legend()
    plt.grid(color=(0.75, 0.75, 0.75, 0.25))
    plt.show()


def plot_problem_04():
    results_dir = '../../results'
    res = parse_csvs_p4(results_dir)

    methods = ['grouped', 'individ']
    for_plot_m = {m: {} for m in methods}
    for_plot_s = {m: {} for m in methods}

    for method in methods:
        print("{} results:".format(method))
        for n in sorted(res[method]):
            for procs in [2, 4, 8, 16]:
                c_data = res[method][n]['{:03d}'.format(procs)]
                # print(c_data)
                m = c_data.mean() * 1000.0
                s = c_data.std() * 1000.0

                if n not in for_plot_m[method]:
                    for_plot_m[method][n] = []
                if n not in for_plot_s[method]:
                    for_plot_s[method][n] = []
                for_plot_m[method][n].append(m)
                for_plot_s[method][n].append(s)

                print("{}, n = {}, p = {}, count = {}, mean = {:.2f}ms, std = {:.2f}ms".format(
                    method, n, procs, len(res['grouped'][n]), m, s))
        print()

    ns = for_plot_m['grouped'].keys()
    for n in ns:
        plt.figure()
        # x = proc, y = time (with error bar); 1 line for grouped, 1 line for individ
        vals_g = for_plot_m['grouped'][n]
        vals_i = for_plot_m['individ'][n]

        stds_g = for_plot_s['grouped'][n]
        stds_i = for_plot_s['individ'][n]
        xx = np.array([2, 4, 8, 16])
        plt.errorbar(xx, vals_g, yerr=stds_g, label="Grouped transfer", capsize=8)
        plt.errorbar(xx, vals_i, yerr=stds_i, label="Using n individual transfers", capsize=8)
        plt.xlabel("$p$ (Number of processors)")
        plt.ylabel("Time (ms)")
        plt.ylim(0, 25)
        plt.legend()
        plt.title("n = {}".format(n))

        for ext in ['png', 'eps']:
            plt.savefig('../../results/plots/problem-04-n-{}.{}'.format(n, ext))

    # plt.show()


def main():
    # plot_problem_02()
    plot_problem_04()



if __name__ == '__main__':
    main()
