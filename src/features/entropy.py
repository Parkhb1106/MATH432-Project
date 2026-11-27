"""Entropy features: sample entropy, COSEn, (multiscale) permutation entropy.

Sample entropy (Richman & Moorman, 2000)
    Templates u_i = (x_i, ..., x_{i+m-1}), i = 0..N-m-1 (the same N-m starting
    points are used for lengths m and m+1). With Chebyshev distance,
        B = #{i < j : |u_i - u_j|_inf <= r}            (length m)
        A = #{i < j : |u_i - u_j|_inf <= r}            (length m + 1)
        SampEn = -ln(A / B).
    If A = 0 (or B = 0) the value is undefined; we return the largest value a
    series of this length can certify, ln(C(N-m, 2)), and the caller can see
    it through ``sampen_undefined``.

COSEn (Lake & Moorman, 2011)
    Coefficient of sample entropy, built for AF detection on very short RR
    series: COSEn = SampEn(m, r) + ln(2r) - ln(mean RR), with r fixed in
    seconds. The ln(2r) term makes values comparable across r ("quadratic
    sample entropy") and -ln(mean RR) adds heart-rate information.

Permutation entropy (Bandt & Pompe, 2002)
    Map each window (x_i, x_{i+tau}, ..., x_{i+(n-1)tau}) to the permutation
    that sorts it, and take the Shannon entropy of the pattern frequencies,
    normalised by ln(n!) so that 0 = one pattern only, 1 = all equally likely.
    Ties are broken by position (stable sort). AFDB RR values are quantised to
    4 ms, so equal neighbours do occur in steady sinus rhythm; they all map to
    the "ascending" pattern, which lowers PE slightly for very regular rhythm.

Multiscale PE (Costa et al., 2002 coarse-graining)
    Scale s: average non-overlapping blocks of s values, then compute PE.
"""

from __future__ import annotations

from math import factorial, log

import numpy as np


def _templates(x: np.ndarray, length: int, count: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(x, length)[:count]


def _match_pairs(u: np.ndarray, r: float) -> int:
    d = np.max(np.abs(u[:, None, :] - u[None, :, :]), axis=2)
    iu = np.triu_indices(len(u), k=1)
    return int(np.count_nonzero(d[iu] <= r))


def sample_entropy_counts(x: np.ndarray, m: int, r: float) -> tuple[int, int]:
    x = np.asarray(x, dtype=float)
    n_t = len(x) - m
    B = _match_pairs(_templates(x, m, n_t), r)
    A = _match_pairs(_templates(x, m + 1, n_t), r)
    return A, B


def sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None,
                   r_factor: float = 0.2) -> tuple[float, bool]:
    """Return (SampEn, undefined_flag). r defaults to r_factor * std(x)."""
    x = np.asarray(x, dtype=float)
    if r is None:
        r = r_factor * x.std()
    A, B = sample_entropy_counts(x, m, r)
    if A == 0 or B == 0:
        n_t = len(x) - m
        return log(n_t * (n_t - 1) / 2), True
    return -log(A / B), False


def cosen(rr: np.ndarray, m: int = 1, r: float = 0.03) -> tuple[float, bool]:
    se, undefined = sample_entropy(rr, m=m, r=r)
    return se + log(2 * r) - log(float(np.mean(rr))), undefined


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    x = np.asarray(x, dtype=float)
    n_win = len(x) - (order - 1) * delay
    if n_win < 1:
        return np.nan
    idx = np.arange(order) * delay + np.arange(n_win)[:, None]
    patterns = np.argsort(x[idx], axis=1, kind="stable")
    # encode each permutation as one integer
    codes = patterns @ (order ** np.arange(order))
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / log(factorial(order)))


def coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    k = len(x) // scale
    return x[:k * scale].reshape(k, scale).mean(axis=1)


def multiscale_permutation_entropy(x: np.ndarray, scales=(1, 2, 3),
                                   order: int = 3, delay: int = 1) -> np.ndarray:
    return np.array([permutation_entropy(coarse_grain(x, s), order, delay) for s in scales])


MPE_SCALES = (1, 2, 3)


def entropy_features(rr: np.ndarray) -> dict[str, float]:
    se, se_undef = sample_entropy(rr, m=2, r_factor=0.2)
    co, co_undef = cosen(rr, m=1, r=0.03)
    mpe = multiscale_permutation_entropy(rr, MPE_SCALES, order=3)
    out = {
        "ent_sampen": se,
        "ent_sampen_undefined": float(se_undef),
        "ent_cosen": co,
        "ent_cosen_undefined": float(co_undef),
    }
    out.update({f"ent_mpe_s{s}": v for s, v in zip(MPE_SCALES, mpe)})
    return out
