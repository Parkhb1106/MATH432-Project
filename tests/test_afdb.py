import numpy as np
import pytest
import wfdb

from src.data.afdb import AFIB, J, N, UNKNOWN, rhythm_at, rhythm_changes


def make_ann(samples, notes):
    return wfdb.Annotation(record_name="t", extension="atr",
                           sample=np.array(samples), symbol=["+"] * len(samples),
                           aux_note=list(notes))


def test_rhythm_changes_parses_and_sorts():
    s, c = rhythm_changes(make_ann([500, 100], ["(AFIB", "(N"]))
    assert s.tolist() == [100, 500]
    assert c.tolist() == [N, AFIB]


def test_rhythm_changes_strips_null_padding():
    _, c = rhythm_changes(make_ann([0], ["(AFIB\x00"]))
    assert c.tolist() == [AFIB]


def test_rhythm_changes_rejects_unknown_label():
    with pytest.raises(ValueError):
        rhythm_changes(make_ann([0], ["(VT"]))


def test_rhythm_at_boundaries():
    change_s = np.array([100, 500, 900])
    change_c = np.array([N, AFIB, J], dtype=np.int8)
    samples = np.array([50, 100, 499, 500, 899, 900, 10_000])
    got = rhythm_at(samples, change_s, change_c)
    # A rhythm takes effect exactly at its change sample.
    assert got.tolist() == [UNKNOWN, N, N, AFIB, AFIB, J, J]
