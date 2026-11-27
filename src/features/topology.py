"""Topological features: Takens delay embedding -> Vietoris-Rips persistent
homology -> summaries, persistence images, persistence landscapes.

Delay embedding
    x_i = (s_i, s_{i+tau}, ..., s_{i+(d-1)tau}),  i = 0..N-(d-1)tau-1.
    For an RR series, d = 2, tau = 1 is the Poincaré plot.

Scale (``normalize``)
    "none"   : embed RR in seconds. H0 deaths then measure how far apart
               successive beats' points are, i.e. the same spread that
               RMSSD / SD1 measure.
    "zscore" : standardise the segment first, so the diagram describes shape
               only; spread is left to the HRV / entropy features.

Persistent homology
    ripser on the point cloud, H0 and H1. In a Rips filtration every H0 class
    is born at 0, and one H0 class never dies (death = inf); it is dropped.

Vectorisation (both are fitted on training diagrams only, then applied to
test diagrams, so no information leaks across folds)
    PersistenceImage (Adams et al., 2017): points mapped to
        (birth, persistence), each spread as a Gaussian weighted by a linear
        ramp in persistence, evaluated at pixel centres of a fixed grid.
    PersistenceLandscape (Bubenik, 2015): each point (b, d) gives a tent
        f(t) = max(0, min(t - b, d - t)); lambda_k(t) is the k-th largest tent
        value at t, sampled on a fixed grid.
"""

from __future__ import annotations

import numpy as np
from ripser import ripser

EMPTY = np.zeros((0, 2))


# ---------------------------------------------------------------- embedding
def delay_embed(x: np.ndarray, d: int = 3, tau: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x) - (d - 1) * tau
    if n < 1:
        raise ValueError(f"series of length {len(x)} too short for d={d}, tau={tau}")
    return x[np.arange(d) * tau + np.arange(n)[:, None]]


def _scale(x: np.ndarray, normalize: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if normalize == "none":
        return x
    if normalize == "zscore":
        s = x.std()
        return (x - x.mean()) / s if s > 0 else x - x.mean()
    raise ValueError(f"unknown normalize={normalize!r}")


# ---------------------------------------------------------------- homology
def persistence_diagrams(points: np.ndarray, maxdim: int = 1) -> list[np.ndarray]:
    """[H0, H1, ...] finite (birth, death) pairs; the infinite H0 bar is dropped."""
    dgms = ripser(points, maxdim=maxdim)["dgms"]
    return [dg[np.isfinite(dg[:, 1])] for dg in dgms]


def segment_diagrams(rr: np.ndarray, d: int = 3, tau: int = 1,
                     normalize: str = "none", maxdim: int = 1) -> list[np.ndarray]:
    return persistence_diagrams(delay_embed(_scale(rr, normalize), d, tau), maxdim)


# ---------------------------------------------------------------- summaries
def persistence_entropy(dgm: np.ndarray) -> float:
    p = dgm[:, 1] - dgm[:, 0]
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    q = p / p.sum()
    return float(-(q * np.log(q)).sum())


def diagram_summary(dgms: list[np.ndarray]) -> dict[str, float]:
    h0, h1 = dgms[0], dgms[1] if len(dgms) > 1 else EMPTY
    p0 = h0[:, 1] - h0[:, 0]
    p1 = h1[:, 1] - h1[:, 0]
    return {
        "top_h0_total": float(p0.sum()),
        "top_h0_max": float(p0.max()) if len(p0) else 0.0,
        "top_h0_mean": float(p0.mean()) if len(p0) else 0.0,
        "top_h0_entropy": persistence_entropy(h0),
        "top_h1_count": float(len(p1)),
        "top_h1_total": float(p1.sum()),
        "top_h1_max": float(p1.max()) if len(p1) else 0.0,
        "top_h1_entropy": persistence_entropy(h1),
        # largest loop relative to the cloud's connectivity scale
        "top_h1_max_rel": float(p1.max() / p0.max()) if len(p1) and p0.max() > 0 else 0.0,
    }


def topology_features(rr: np.ndarray, d: int = 3, tau: int = 1,
                      normalize: str = "none") -> dict[str, float]:
    return diagram_summary(segment_diagrams(rr, d, tau, normalize))


# ---------------------------------------------------------------- vectorisers
class PersistenceImage:
    """Fit the grid on training diagrams (one homology dimension), then transform.

    Births and persistences are mapped to a res_b x res_p grid spanning
    [0, max birth] x [0, max persistence] of the training set (the
    ``quantile`` of each, to resist outliers). In H0 of a Rips filtration all
    births are 0, so the birth axis collapses to one pixel automatically.
    """

    def __init__(self, res: int = 10, sigma: float = 1.0, quantile: float = 0.99):
        self.res, self.sigma, self.quantile = res, sigma, quantile

    def fit(self, dgms: list[np.ndarray]) -> "PersistenceImage":
        pts = np.concatenate([d for d in dgms if len(d)] or [EMPTY])
        b, p = pts[:, 0], pts[:, 1] - pts[:, 0]
        b_max = np.quantile(b, self.quantile) if len(b) else 0.0
        p_max = np.quantile(p, self.quantile) if len(p) else 1.0
        self.res_b_ = 1 if b_max <= 1e-12 else self.res
        self.p_max_ = max(p_max, 1e-12)
        self.b_edges_ = np.linspace(0, max(b_max, 1e-12), self.res_b_ + 1)
        self.p_edges_ = np.linspace(0, self.p_max_, self.res + 1)
        self.b_c_ = (self.b_edges_[:-1] + self.b_edges_[1:]) / 2
        self.p_c_ = (self.p_edges_[:-1] + self.p_edges_[1:]) / 2
        # Gaussian width in data units: sigma pixels
        self.s_b_ = self.sigma * (self.b_edges_[1] - self.b_edges_[0])
        self.s_p_ = self.sigma * (self.p_edges_[1] - self.p_edges_[0])
        return self

    def _one(self, dgm: np.ndarray) -> np.ndarray:
        if len(dgm) == 0:
            return np.zeros(self.res_b_ * self.res)
        b, p = dgm[:, 0], dgm[:, 1] - dgm[:, 0]
        w = np.minimum(p / self.p_max_, 1.0)
        gp = np.exp(-0.5 * ((self.p_c_[None, :] - p[:, None]) / self.s_p_) ** 2)
        if self.res_b_ == 1:
            img = (w[:, None] * gp).sum(axis=0)[None, :]
        else:
            gb = np.exp(-0.5 * ((self.b_c_[None, :] - b[:, None]) / self.s_b_) ** 2)
            img = np.einsum("k,kb,kp->bp", w, gb, gp)
        return img.ravel()

    def transform(self, dgms: list[np.ndarray]) -> np.ndarray:
        return np.stack([self._one(d) for d in dgms])

    def fit_transform(self, dgms):
        return self.fit(dgms).transform(dgms)


class PersistenceLandscape:
    """lambda_1..lambda_k sampled at n_grid points on [min birth, max death] of training data."""

    def __init__(self, k: int = 3, n_grid: int = 20, quantile: float = 0.99):
        self.k, self.n_grid, self.quantile = k, n_grid, quantile

    def fit(self, dgms: list[np.ndarray]) -> "PersistenceLandscape":
        pts = np.concatenate([d for d in dgms if len(d)] or [EMPTY])
        lo = pts[:, 0].min() if len(pts) else 0.0
        hi = np.quantile(pts[:, 1], self.quantile) if len(pts) else 1.0
        self.grid_ = np.linspace(lo, max(hi, lo + 1e-12), self.n_grid)
        return self

    def _one(self, dgm: np.ndarray) -> np.ndarray:
        out = np.zeros((self.k, self.n_grid))
        if len(dgm):
            t = self.grid_[None, :]
            tents = np.maximum(0, np.minimum(t - dgm[:, :1], dgm[:, 1:2] - t))
            tents = -np.sort(-tents, axis=0)          # descending per t
            m = min(self.k, len(dgm))
            out[:m] = tents[:m]
        return out.ravel()

    def transform(self, dgms: list[np.ndarray]) -> np.ndarray:
        return np.stack([self._one(d) for d in dgms])

    def fit_transform(self, dgms):
        return self.fit(dgms).transform(dgms)


# ---------------------------------------------------------------- caching
def pack_diagrams(dgms: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Ragged list of (n_i, 2) arrays -> (concatenated points, offsets)."""
    offsets = np.concatenate([[0], np.cumsum([len(d) for d in dgms])])
    pts = np.concatenate(dgms) if len(dgms) else EMPTY
    return pts.astype(np.float32), offsets.astype(np.int64)


def unpack_diagrams(pts: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [pts[offsets[i]:offsets[i + 1]].astype(float) for i in range(len(offsets) - 1)]
