"""Solves quadratic spline interpolation using a built-in numpy solver."""

import math

import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import spline as scipy_spline


def main():
    # The problem description indicates we have a grid with points indexed from 0 to n inclusive, so n+1 total points,
    # with the first (idx 0) = a, and the last (idx n) = b.
    n = 14
    a = 0.0
    b = math.pi * 12.0
    # b = 3.0
    # fn = lambda x: x ** 2
    fn = lambda x: np.sin(x)
    plot_gt_interpolation = True

    # This is the dense sample set we plot the GT and the interpolation results on
    xs = np.linspace(a, b, 1000)
    ys = fn(xs)

    step_size = (b - a) / n

    # Remember the condition in the problem spec! There are n+1 points in total, with the first = a, and the last = b,
    # so n-1 inner points.
    knots_xs = np.linspace(a, b, n + 1)
    print("Knots:", knots_xs)

    midpoints = np.array([(knots_xs[i-1] + knots_xs[i]) / 2.0 for i in range(1, n + 1)])
    print("Knots shape: {}".format(knots_xs.shape))
    print("Midpoints shape: {}".format(midpoints.shape))
    print(midpoints)

    midpoints_and_endpoints = np.zeros(n + 2)
    midpoints_and_endpoints[0] = knots_xs[0]
    midpoints_and_endpoints[1:(n+1)] = midpoints
    midpoints_and_endpoints[n+1] = knots_xs[n]

    # Function values in the mid- and end-points
    u = fn(midpoints_and_endpoints)

    # TODO(andreib): Build sparse matrix.
    A = np.zeros((n + 2, n + 2))
    b = u
    A[0, 0] = 4
    A[0, 1] = 4
    A[n + 1, n] = 4
    A[n + 1, n + 1] = 4
    for i in range(1, n + 1):
        A[i, i - 1] = 1
        A[i, i] = 6
        A[i, i + 1] = 1

    A *= 1.0 / 8.0
    print("Solving system of shape: ", A.shape)
    c = np.linalg.solve(A, b)

    def phi(x):
        if 0 <= x <= 1:
            return 0.5 * (x ** 2)
        elif 1 < x <= 2:
            return 0.5 * (-2.0 * (x - 1) ** 2 + 2 * (x - 1) + 1)
        elif 2 < x <= 3:
            return 0.5 * (3 - x) ** 2
        else:
            return 0

    def phi_i(i, x):
        assert i >= 0 and i <= n + 1
        return phi((x - a) / step_size - i + 2)

    def poly(x):
        i = int(math.ceil(x / step_size))
        print("Poly:", x, i)
        val = 0
        if i > 0:
            val += c[i - 1] * phi_i(i - 1, x)

        if i < n + 2:
            val += c[i + 1] * phi_i(i + 1, x)

        val += c[i] * phi_i(i, x)
        return val

    def poly_slow(x):
        return sum(c[i] * phi_i(i, x) for i in range(n + 2))

    plt.figure()
    plt.plot(xs, ys, label="True function")
    plt.scatter(midpoints_and_endpoints, u, label="Knots")

    # vals_slow = [poly_slow(x) for x in xs]
    # plt.plot(xs, vals_slow, label="Result (naive sum)")
    vals = [poly(x) for x in xs]
    plt.plot(xs, vals, '--', label="Result (fast intervals)")

    # Ground truth interpolation using a ready made spline interpolator
    if plot_gt_interpolation:
        xk = midpoints_and_endpoints
        yk = u
        xnew = xs
        gt_y = scipy_spline(xk, yk, xnew, order=2)
        plt.plot(xnew, gt_y, label="Computed with SciPy")

    print("Coefs:", c)
    print("b:    ", b)
    print("Step size:", step_size)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
