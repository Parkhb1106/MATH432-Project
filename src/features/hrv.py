"""Baseline HRV statistics of one beat-interval segment (time domain + Poincaré).

These are the clinical-style comparison features (feature set (v)). SD1/SD2
are included on purpose: SD1 measures the spread of the Poincaré plot across
the identity line, which is what the H0 persistence of a raw-scale 2-D delay
embedding also sees, so the baseline is not artificially weak.
"""

from __future__ import annotations

import numpy as np


def hrv_features(rr: np.ndarray) -> dict[str, float]:
    """rr: 1-D beat intervals in seconds."""
    rr = np.asarray(rr, dtype=float)
    d = np.diff(rr)
    mean = rr.mean()
    sdnn = rr.std(ddof=1)
    rmssd = np.sqrt(np.mean(d ** 2))
    sd1 = np.sqrt(0.5) * d.std(ddof=1)
    sd2 = np.sqrt(max(2 * sdnn ** 2 - sd1 ** 2, 0.0))
    return {
        "hrv_mean_rr": mean,
        "hrv_sdnn": sdnn,
        "hrv_rmssd": rmssd,
        "hrv_pnn50": float(np.mean(np.abs(d) > 0.05)),
        "hrv_cv": sdnn / mean,
        "hrv_nrmssd": rmssd / mean,
        "hrv_sd1": sd1,
        "hrv_sd2": sd2,
        "hrv_sd1_sd2": sd1 / sd2 if sd2 > 0 else np.nan,
    }
