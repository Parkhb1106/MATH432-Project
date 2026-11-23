import numpy as np

from src.data.folds import make_folds


def test_every_record_in_exactly_one_fold():
    rng = np.random.default_rng(0)
    record = np.repeat([f"r{i}" for i in range(10)], 20)
    label = rng.integers(0, 2, len(record))
    folds = make_folds(record, label, n_splits=5)
    assert sorted(folds) == sorted(set(record))
    assert set(folds.values()) == set(range(5))


def test_deterministic():
    record = np.repeat([f"r{i}" for i in range(10)], 20)
    label = np.tile([0, 1], 100)
    assert make_folds(record, label) == make_folds(record, label)
