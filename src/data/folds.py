"""Patient-level cross-validation folds (one AFDB record = one patient).

5-fold StratifiedGroupKFold over segments, grouped by record and stratified
by ``label_50`` so every fold has a similar AF share. Fixed seed so A and B
use identical folds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

N_SPLITS = 5
SEED = 0


def make_folds(record: np.ndarray, label: np.ndarray,
               n_splits: int = N_SPLITS, seed: int = SEED) -> dict[str, int]:
    """Return {record: fold index}."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = {}
    for k, (_, test) in enumerate(cv.split(np.zeros(len(label)), label, groups=record)):
        for r in np.unique(record[test]):
            folds[str(r)] = k
    return dict(sorted(folds.items()))


def save_folds(folds: dict[str, int], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(folds, indent=2) + "\n")
