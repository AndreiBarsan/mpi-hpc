"""Plots the output produced by spline_problem.cpp."""

import json
import matplotlib.pyplot as plt
import os
import sys


def main():
    root = sys.argv[1]
    print("Reading plot data from root dir: {}".format(root))

    files = [os.path.join(root, f) for f in os.listdir(root) if 'output-problem' in f]

    for file in files:
        data = json.load(open(file, 'r'))

        x = data['x']
        gt_y = data['gt_y']
        interp_y = data['interp_y']

        plt.figure()
        plt.scatter(data['control_x'], data['control_y'], label="Knots")
        plt.plot(x, gt_y, label="Ground truth function")
        plt.plot(x, interp_y, '--', label="Interpolation result")
        plt.legend()
        plt.savefig(root + "/plot-{}.png".format(file[file.rfind('/') + 1:]))

    plt.show()


if __name__ == '__main__':
    main()
