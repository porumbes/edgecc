#!/usr/bin/env python3
"""Regenerate the committed .npy fixtures of the npy_* e2e tests.

    python3 tests/e2e/npy_fixtures.py

Writes every .npy input below, plus npy_fft_split_n4/expected_output.txt
(computed with numpy.fft.fft, an oracle independent of edgecc). Needs numpy
only and never runs edgecc.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INF = float("inf")


def save(test, fname, array):
    np.save(os.path.join(HERE, test, fname), array)


def fft_split_n4():
    """One radix-2 split, N = 4: input Xs[m][b][d] = (Re, Im) of x[2m + b];
    expected Y[k1][k2][c] = (Re, Im) of X[k1 + 2*k2], X = fft(x)."""
    x = np.array([1, 2 + 1j, 3j, -1 + 2j])
    h = len(x) // 2
    xs = [[[x[2 * m + b].real, x[2 * m + b].imag] for b in range(2)]
          for m in range(h)]
    save("npy_fft_split_n4", "xs.npy", np.array(xs, dtype=float))
    X = np.fft.fft(x)
    y = [[[round(float(v), 9) + 0.0 for v in (X[k1 + h * k2].real, X[k1 + h * k2].imag)]
          for k2 in range(2)]
         for k1 in range(h)]
    with open(os.path.join(HERE, "npy_fft_split_n4", "expected_output.txt"), "w") as f:
        f.write(repr(y) + "\n")


def main():
    save("npy_load_1d", "f.npy", np.array([1, 2, 3], dtype=float))
    save("npy_load_save_3d", "a.npy",
         np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64))
    save("npy_inferred_dim_axis2", "x.npy",
         np.arange(1, 13, dtype=float).reshape(2, 2, 3))
    save("npy_argv_override", "g_arg.npy", np.array([[1, 2], [3, 4]], dtype=float))
    save("npy_inf_minplus", "w.npy",
         np.array([[INF, 1, 4], [INF, INF, 2], [INF, INF, INF]], dtype=float))
    save("npy_wrong_ndim", "a.npy", np.array([[1, 2], [3, 4]], dtype=float))
    save("npy_wrong_size", "g.npy", np.eye(4))
    save("npy_repeated_dim_mismatch", "g.npy",
         np.arange(1, 7, dtype=float).reshape(2, 3))
    save("npy_complex_rejected", "c.npy", np.array([1 + 2j, 3 - 1j]))
    fft_split_n4()


if __name__ == "__main__":
    main()
