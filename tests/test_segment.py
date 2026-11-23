import numpy as np

from src.data.afdb import AFIB, AFL, N, RecordRR
from src.data.segment import segment_record


def make_record(rhythm, valid=None):
    n_rr = len(rhythm)
    beats = np.arange(n_rr + 1) * 200
    return RecordRR(record="r1", fs=250.0, beat_source="qrs", has_signal=True,
                    beats=beats, rr=np.diff(beats) / 250.0,
                    rhythm=np.asarray(rhythm, dtype=np.int8),
                    valid=np.ones(n_rr, bool) if valid is None else np.asarray(valid))


def test_non_overlapping_and_tail_dropped():
    seg = segment_record(make_record([N] * 25), n=10)
    assert seg["rr"].shape == (2, 10)
    assert seg["start_samp"].tolist() == [0, 2000]
    assert seg["end_samp"].tolist() == [2000, 4000]


def test_af_fraction_and_labels():
    rhythm = [N] * 10 + [AFIB] * 5 + [N] * 5 + [AFIB] * 10 + [AFIB] * 1 + [N] * 9
    seg = segment_record(make_record(rhythm), n=10)
    assert np.allclose(seg["af_frac"], [0.0, 0.5, 1.0, 0.1])
    assert seg["label_50"].tolist() == [0, 1, 1, 0]
    assert seg["label_excl"].tolist() == [0, -1, 1, 0]


def test_drops_segments_with_afl_or_invalid_rr():
    rhythm = [N] * 10 + [N] * 9 + [AFL] + [N] * 10
    valid = [True] * 25 + [False] + [True] * 4
    seg = segment_record(make_record(rhythm, valid), n=10)
    assert seg["start_samp"].tolist() == [0]
