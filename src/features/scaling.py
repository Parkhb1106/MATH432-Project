"""Detrended fluctuation analysis (DFA), Peng et al. (1995).

For a series x of length N:
1. Profile y(k) = sum_{i<=k} (x_i - mean(x)).
2. For each box size n, cut y into non-overlapping boxes of n points, taken
   from the start and again from the end (2 * floor(N/n) boxes), fit a line in
   each box, and let F(n) be the RMS of the residuals over all boxes.
3. alpha = slope of log F(n) against log n.

White noise gives alpha ~ 0.5, 1/f noise ~ 1.0, Brownian motion ~ 1.5.
Healthy sinus rhythm has short-term alpha1 (n = 4..16 beats) around 1;
AF is closer to uncorrelated noise (alpha1 -> 0.5). With 60-beat segments
only alpha1 is computable; alpha2 needs boxes of 16..64+ beats.
"""

from __future__ import annotations

import numpy as np

ALPHA1_SCALES = np.arange(4, 17)


def fluctuation(x: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """F(n) for every box size n in ``scales``."""
    x = np.asarray(x, dtype=float)
    y = np.cumsum(x - x.mean())
    N = len(y)
    F = np.empty(len(scales))
    for j, n in enumerate(scales):
        k = N // n
        if k < 1:
            raise ValueError(f"box size {n} longer than series ({N})")
        boxes = np.concatenate([y[:k * n].reshape(k, n), y[N - k * n:].reshape(k, n)])
        t = np.arange(n)
        # least-squares line per box, vectorised: residual = y - (a t + b)
        A = np.vstack([t, np.ones(n)]).T
        coef, *_ = np.linalg.lstsq(A, boxes.T, rcond=None)
        resid = boxes - (A @ coef).T
        F[j] = np.sqrt(np.mean(resid ** 2))
    return F


def dfa_alpha(x: np.ndarray, scales: np.ndarray = ALPHA1_SCALES) -> float:
    F = fluctuation(x, scales)
    if np.any(F <= 0):
        return np.nan  # constant series: log F undefined
    return float(np.polyfit(np.log(scales), np.log(F), 1)[0])


def scaling_features(rr: np.ndarray) -> dict[str, float]:
    return {"dfa_alpha1": dfa_alpha(rr, ALPHA1_SCALES)}
