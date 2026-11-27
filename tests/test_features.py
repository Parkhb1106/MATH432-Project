"""Feature checks on synthetic series with known behaviour."""

from math import log

import numpy as np
import pytest

from src.features import family_columns, scalar_features
from src.features.entropy import (coarse_grain, cosen, permutation_entropy,
                                  sample_entropy, sample_entropy_counts)
from src.features.hrv import hrv_features
from src.features.scaling import dfa_alpha
from src.features.topology import (PersistenceImage, PersistenceLandscape,
                                   delay_embed, diagram_summary, pack_diagrams,
                                   persistence_diagrams, segment_diagrams,
                                   unpack_diagrams)

rng = np.random.default_rng(0)


def sinus_rr(n=60, seed=1):
    """Regular rhythm: 0.8 s with slow respiratory modulation and tiny noise."""
    r = np.random.default_rng(seed)
    t = np.arange(n)
    return 0.8 + 0.03 * np.sin(2 * np.pi * t / 15) + 0.003 * r.standard_normal(n)


def af_rr(n=60, seed=2):
    """AF-like rhythm: independent, widely spread intervals."""
    return np.random.default_rng(seed).uniform(0.4, 1.1, n)


# ------------------------------------------------------------------ HRV
def test_hrv_known_values():
    rr = np.array([1.0, 0.9, 1.0, 0.9])
    f = hrv_features(rr)
    assert f["hrv_mean_rr"] == pytest.approx(0.95)
    assert f["hrv_rmssd"] == pytest.approx(0.1)
    assert f["hrv_pnn50"] == pytest.approx(1.0)
    # SD1 = std(diff)/sqrt(2) with ddof=1
    assert f["hrv_sd1"] == pytest.approx(np.std(np.diff(rr), ddof=1) / np.sqrt(2))


# ------------------------------------------------------------------ DFA
@pytest.mark.parametrize("make, target", [
    (lambda x: x, 0.5),              # white noise
    (lambda x: np.cumsum(x), 1.5),   # Brownian motion
])
def test_dfa_reference_processes(make, target):
    x = make(rng.standard_normal(20000))
    assert dfa_alpha(x) == pytest.approx(target, abs=0.1)


def test_dfa_separates_sinus_from_af():
    assert dfa_alpha(sinus_rr()) > 1.0 > dfa_alpha(af_rr())


# ------------------------------------------------------------------ entropy
def _sampen_bruteforce(x, m, r):
    n_t = len(x) - m
    def count(L):
        c = 0
        for i in range(n_t):
            for j in range(i + 1, n_t):
                if max(abs(x[i + k] - x[j + k]) for k in range(L)) <= r:
                    c += 1
        return c
    return count(m + 1), count(m)


def test_sampen_counts_match_bruteforce():
    x = rng.standard_normal(40)
    assert sample_entropy_counts(x, 2, 0.5) == _sampen_bruteforce(x, 2, 0.5)


def test_sampen_regular_below_random():
    se_reg, _ = sample_entropy(np.sin(np.linspace(0, 20 * np.pi, 300)))
    se_rand, _ = sample_entropy(rng.standard_normal(300))
    assert se_reg < 0.3 < 1.5 < se_rand


def test_sampen_undefined_returns_bound():
    x = np.arange(10.0)  # strictly increasing, r tiny: no matches at all
    se, undefined = sample_entropy(x, m=2, r=1e-6)
    assert undefined and se == pytest.approx(log(8 * 7 / 2))


def test_cosen_higher_for_af():
    assert cosen(af_rr())[0] > cosen(sinus_rr())[0]


def test_permutation_entropy_extremes():
    assert permutation_entropy(np.arange(50.0)) == 0.0
    assert permutation_entropy(rng.standard_normal(20000)) > 0.99


def test_coarse_grain():
    np.testing.assert_allclose(coarse_grain(np.arange(7.0), 3), [1.0, 4.0])


# ------------------------------------------------------------------ topology
def test_delay_embed_is_poincare_for_d2():
    x = np.arange(5.0)
    np.testing.assert_array_equal(delay_embed(x, 2, 1),
                                  [[0, 1], [1, 2], [2, 3], [3, 4]])
    assert delay_embed(np.arange(10.0), 3, 2).shape == (6, 3)


def test_circle_has_one_long_loop():
    th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    dgms = persistence_diagrams(np.c_[np.cos(th), np.sin(th)])
    p1 = dgms[1][:, 1] - dgms[1][:, 0]
    assert len(dgms[0]) == 59                    # n - 1 finite H0 bars
    assert p1.max() > 1.0 and np.sort(p1)[-2:-1].sum() < 0.1


def test_blob_has_no_long_loop():
    dgms = persistence_diagrams(rng.standard_normal((60, 2)) * 0.1)
    p1 = dgms[1][:, 1] - dgms[1][:, 0] if len(dgms[1]) else np.zeros(1)
    assert p1.max() < 0.1


def test_raw_scale_h0_larger_for_af():
    s_sin = diagram_summary(segment_diagrams(sinus_rr(), d=2))
    s_af = diagram_summary(segment_diagrams(af_rr(), d=2))
    assert s_af["top_h0_total"] > s_sin["top_h0_total"]


def test_zscore_removes_scale():
    a = segment_diagrams(af_rr(), normalize="zscore")
    b = segment_diagrams(af_rr() * 3 + 1, normalize="zscore")
    for da, db in zip(a, b):
        np.testing.assert_allclose(da, db, atol=1e-5)


def test_persistence_image_shape_and_h0_collapse():
    dgms = [segment_diagrams(af_rr(seed=s)) for s in range(5)]
    h1 = [d[1] for d in dgms]
    pi = PersistenceImage(res=8).fit(h1)
    assert pi.transform(h1).shape == (5, 64)
    h0 = [d[0] for d in dgms]                    # births all 0 -> 1 x res
    assert PersistenceImage(res=8).fit_transform(h0).shape == (5, 8)
    assert PersistenceImage(res=8).fit(h1).transform([np.zeros((0, 2))]).sum() == 0


def test_landscape_single_point_peak():
    pl = PersistenceLandscape(k=2, n_grid=101).fit([np.array([[0.0, 2.0]])])
    v = pl.transform([np.array([[0.0, 2.0]])]).reshape(2, 101)
    assert v[0].max() == pytest.approx(1.0)       # (d - b) / 2 at t = 1
    assert v[1].max() == 0.0


def test_pack_roundtrip():
    dgms = [np.array([[0, 1.0]]), np.zeros((0, 2)), np.array([[0, 2.0], [1, 3.0]])]
    back = unpack_diagrams(*pack_diagrams(dgms))
    for a, b in zip(dgms, back):
        np.testing.assert_allclose(a, b)


# ------------------------------------------------------------------ combined
def test_scalar_features_families():
    f = scalar_features(af_rr())
    assert all(np.isfinite(v) for v in f.values())
    assert family_columns(f, "scaling") == ["dfa_alpha1"]
    assert "ent_sampen_undefined" not in family_columns(f, "entropy")
    assert len(family_columns(f, "hrv")) == 9
