"""Generates the plot for Problem 1 (d)."""

import math

import matplotlib.pyplot as plt
import numpy as np


t_f = 5e-5
t_s = 1e-3
t_w = 1e-6


def fixed_eff(n, p):
    denom = (2 * (p ** 1.5) * t_s) / (n ** 3 * t_f) + (2 * (p ** 0.5) * t_w) / (n * t_f)
    return 1.0 / (1.0 + denom)


def scaled_eff(n_1, p):
    # n = p ** (1.0 / 3.0) * n_1

    denom = (2 * (p ** 0.5) * t_s) / (n_1 ** 3 * t_f) + (2 * (p ** (1.0 / 6.0)) * t_w) / (n_1 * t_f)
    return 1.0 / (1.0 + denom)


def scaled_eff_based_on_iso(n_1, p):
    n = n_1 * (p ** 0.5)
    return fixed_eff(n, p)


def detailed_scaled_eff_based_on_iso(n_1, p):
    # Weird... final efficiency does not seem to match this exactly
    k = 0.9999

    n_A = ((2 * p ** (3.0 / 2.0) * t_s) / (t_f * k)) ** (1.0 / 3.0) * n_1
    n_B = (2 * t_w * p ** 0.5) / (t_f * k) * n_1
    n = max(n_A, n_B)
    print(n_A, n_B)

    print("iso n: {} for p = {}".format(n, p))
    E = fixed_eff(n, p)
    print(E)

    return E


def main():
    ps = [2, 4, 8, 16, 32, 64]
    n_1 = 256

    fixed = [fixed_eff(n_1, p) for p in ps]
    scaled = [scaled_eff(n_1, p) for p in ps]
    iso_scaled = [scaled_eff_based_on_iso(n_1, p) for p in ps]
    d_iso_scaled = [detailed_scaled_eff_based_on_iso(n_1, p) for p in ps]
    print(iso_scaled)

    plt.figure(figsize=(9, 5))
    plt.plot(ps, fixed, '-o', label="Fixed efficiency")
    plt.plot(ps, scaled, '-o', label="Scaled workload efficiency")
    # plt.plot(ps, iso_scaled, '-o', label="Scaled efficiency based on iso-efficiency function")

    # This is experimental and may not be necessary.
    plt.plot(ps, d_iso_scaled, '-o', label="Scaled efficiency based on iso-efficiency function (picked $E_p$ = 0.9999)")

    plt.legend()
    plt.xlabel("$p$ (processors)")
    plt.ylabel("$E_p$ (efficiency)")
    plt.tight_layout()
    # plt.ylim(0.9995, 1.0)
    # plt.show()
    plt.savefig("a2p1d-efficiency.eps")
    plt.show()


if __name__ == '__main__':
    main()
