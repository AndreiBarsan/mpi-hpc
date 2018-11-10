"""Plots the output produced by spline_problem.cpp."""

import json
import matplotlib.pyplot as plt
import os


def main():
    print(os.getcwd())
    data = json.load(open("../../results/spline_output/splines-p01.json", 'r'))

    x = data['x']
    gt_y = data['gt_y']
    interp_y = data['interp_y']

    plt.scatter(data['control_x'], data['control_y'], label="Knots")
    plt.plot(x, gt_y, label="Ground truth function")
    plt.plot(x, interp_y, '--', label="Interpolation result")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()