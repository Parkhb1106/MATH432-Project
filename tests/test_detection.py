import numpy as np

from src.data.afdb import AFIB, N
from src.data.detection import compare_beats


def test_compare_beats_metrics():
    fs = 250.0
    ref = np.array([100, 300, 500, 700])
    det = np.array([105, 310, 900])          # 2 matches, 2 misses, 1 false beat
    ref_rhythm = np.array([N, N, AFIB, AFIB], dtype=np.int8)
    det_rhythm = np.array([N, N, AFIB], dtype=np.int8)
    row = compare_beats(ref, det, fs, ref_rhythm, det_rhythm)

    assert row["se"] == 0.5
    assert np.isclose(row["ppv"], 2 / 3)
    assert row["se_nonaf"] == 1.0 and row["se_af"] == 0.0
    assert row["ppv_nonaf"] == 1.0 and row["ppv_af"] == 0.0
    # Only the RR between ref beats 0 and 1 has both ends matched: 205 vs 200 samples.
    assert np.isclose(row["rr_err_med_ms"], 5 / fs * 1000)


def test_beyond_tolerance_is_not_matched():
    fs = 250.0
    ref = np.array([1000, 2000])
    det = np.array([1000 + 38, 2000])        # 38 samples = 152 ms > 150 ms
    rh = np.array([N, N], dtype=np.int8)
    row = compare_beats(ref, det, fs, rh, rh)
    assert row["se"] == 0.5
