"""Split RR series into fixed-length, non-overlapping beat segments and label them.

Rules (agreed 2026-11-23):
* Segment = 60 consecutive RR intervals, no overlap; the incomplete tail of a
  record is dropped.
* A segment is dropped if it contains any implausible RR (``valid`` False),
  any AFL interval, or any interval with unknown rhythm.
* ``af_frac`` = share of the segment's RR intervals labelled AFIB (N and J
  count as non-AF).
* ``label_50``  : 1 if af_frac >= 0.5 else 0.
* ``label_excl``: 1 if af_frac >= 0.9, 0 if af_frac <= 0.1, else -1 (ambiguous).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.afdb import AFIB, AFL, UNKNOWN, RecordRR

SEGMENT_BEATS = 60


def segment_record(rec: RecordRR, n: int = SEGMENT_BEATS) -> dict[str, np.ndarray]:
    n_seg = len(rec.rr) // n
    idx = np.arange(n_seg * n).reshape(n_seg, n)
    rhythm = rec.rhythm[idx]
    keep = (rec.valid[idx].all(axis=1)
            & ~(rhythm == AFL).any(axis=1)
            & ~(rhythm == UNKNOWN).any(axis=1))
    idx, rhythm = idx[keep], rhythm[keep]

    af_frac = (rhythm == AFIB).mean(axis=1)
    label_excl = np.full(len(idx), -1, dtype=np.int8)
    label_excl[af_frac >= 0.9] = 1
    label_excl[af_frac <= 0.1] = 0
    return {
        "rr": rec.rr[idx].astype(np.float32),
        "af_frac": af_frac.astype(np.float32),
        "label_50": (af_frac >= 0.5).astype(np.int8),
        "label_excl": label_excl,
        "record": np.full(len(idx), rec.record),
        "start_samp": rec.beats[idx[:, 0]],       # beat that starts the first RR
        "end_samp": rec.beats[idx[:, -1] + 1],    # beat that ends the last RR
        "beat_source": np.full(len(idx), rec.beat_source),
    }


def segment_all(records: list[RecordRR], n: int = SEGMENT_BEATS) -> dict[str, np.ndarray]:
    parts = [segment_record(r, n) for r in records]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def save_segments(seg: dict[str, np.ndarray], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **seg)
