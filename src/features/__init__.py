"""Per-segment features of a beat-interval series (RR from ECG or PPI from PPG).

Every function takes one 1-D array of intervals in seconds and returns a flat
{name: value} dict, so the same code serves AFDB, Zenodo ECG and Zenodo PPG.

Feature families (for the RQ1 comparison):
    scaling   dfa_*            src.features.scaling
    entropy   ent_*            src.features.entropy
    topology  top_*            src.features.topology (summaries; images and
                               landscapes are fitted per fold from cached
                               diagrams, see PersistenceImage/Landscape)
    hrv       hrv_*            src.features.hrv (baseline)
Columns ending in ``_undefined`` are diagnostics, not model inputs.
"""

from __future__ import annotations

import numpy as np

from src.features.entropy import entropy_features
from src.features.hrv import hrv_features
from src.features.scaling import scaling_features
from src.features.topology import topology_features

FAMILY_PREFIX = {"scaling": "dfa_", "entropy": "ent_", "topology": "top_", "hrv": "hrv_"}


def scalar_features(rr: np.ndarray, d: int = 3, tau: int = 1,
                    normalize: str = "none") -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(scaling_features(rr))
    out.update(entropy_features(rr))
    out.update(topology_features(rr, d=d, tau=tau, normalize=normalize))
    out.update(hrv_features(rr))
    return out


def family_columns(columns, family: str) -> list[str]:
    pre = FAMILY_PREFIX[family]
    return [c for c in columns if c.startswith(pre) and not c.endswith("_undefined")]
