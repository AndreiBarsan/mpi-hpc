"""Plots the output produced by spline_problem.cpp."""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    root = sys.argv[1]
    print("Reading plot data from root dir: {}".format(root))

    files = [os.path.join(root, f) for f in os.listdir(root) if 'output-problem' in f]

    # Little hack for debugging
    # files = [f for f in files if '38' in f]

    for file in files:
        data = json.load(open(file, 'r'))

        m = data['m']
        n = data['n']

        # x = data['x']
        gt_y = data['gt_y']
        interp_y = np.array(data['interp_y'])

        gt_y = np.array(gt_y)
        gt_y = gt_y.reshape((m+1)*3, (n+1)*3)
        interp_y = interp_y.reshape((m+1)*3, (n+1)*3)

        delta = interp_y - gt_y

        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        sns.heatmap(gt_y)
        plt.title("Ground truth values")
        plt.subplot(1, 3, 2)

        sns.heatmap(interp_y)
        plt.title("Interpolation result")

        plt.subplot(1, 3, 3)
        sns.heatmap(delta)
        plt.title("Delta / Error")

        plt.suptitle("Experiment: {}".format(data['name']))

        # plt.figure()
        # plt.scatter(data['control_x'], data['control_y'], label="Knots")
        # plt.plot(x, gt_y, label="Ground truth function")
        # plt.plot(x, interp_y, '--', label="Interpolation result")
        # plt.legend()
        # plt.savefig(root + "/plot-{}.png".format(file[file.rfind('/') + 1:]))

    plt.show()


if __name__ == '__main__':
    main()
