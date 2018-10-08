"""Solves the least-squares fit from Assignment 1, Problem 5."""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {'size': 16}
matplotlib.rc('font', **font)


def main():
    data = np.array([
        [1, 2.57e-05],
        [1000, 3.49e-04],
        [2000, 6.15e-04],
        [4000, 1.24e-03]
    ])

    t_comm = data[:, 1]
    w = data[:, 0].reshape(-1, 1)

    A = np.hstack((w, np.ones_like(w)))
    b = t_comm
    print(A)

    x, residual_sum, _, _ = np.linalg.lstsq(A, b, rcond=None)
    print(x)

    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], label="Observations")
    plt.ylim(0.0, 1.5e-03)

    line = np.linspace(-1000, 6000)
    plt.plot(line, x[1] + x[0] * line, '--k', label="Least-Squares Fit")

    plt.legend()
    plt.xlabel("$w$ (Message Size)")
    plt.ylabel("Measured Time")
    plt.tight_layout()
    plt.grid(color=(0.75, 0.75, 0.75, 0.25))
    plt.show()

    out_dir = '../../results/plots'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # print("Saving figure")
    # print(os.path.realpath(out_dir))
    # plt.savefig(os.path.join(out_dir, 'problem-05.eps'))




if __name__ == '__main__':
    main()
