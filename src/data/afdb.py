"""MIT-BIH Atrial Fibrillation Database: beat loading, RR series, rhythm labels.

Conventions
-----------
* Beat positions come from ``.qrsc`` (manually corrected) when the record has
  one, otherwise from ``.qrs`` (unaudited automatic detector output). Only
  05091 and 07859 have ``.qrsc``.
* Rhythm annotations in ``.atr`` cover the first 10 hours of each record
  (9,000,000 samples at 250 Hz). Beats after that, or after the end of the
  signal when it is shorter (06453), are dropped.
* ``rr[i] = (beats[i+1] - beats[i]) / fs`` in seconds. Each RR interval takes
  the rhythm in effect at the beat that *ends* it (``beats[i+1]``).
  Intervals ending before the first rhythm annotation get ``UNKNOWN``.
* ``valid`` flags physiologically plausible intervals (RR_MIN <= rr <= RR_MAX).
  Invalid intervals usually mean missed or spurious detections; they are
  flagged here and handled during segmentation, not removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

DEFAULT_DIR = Path("data/mitbih_afdb")
ANNOTATED_SECONDS = 10 * 3600
RR_MIN, RR_MAX = 0.25, 2.5  # seconds (240 bpm, 24 bpm)

# Integer rhythm codes used throughout the project.
UNKNOWN, N, AFIB, AFL, J = -1, 0, 1, 2, 3
RHYTHM_CODES = {"(N": N, "(AFIB": AFIB, "(AFL": AFL, "(J": J}
RHYTHM_NAMES = {UNKNOWN: "unknown", N: "N", AFIB: "AFIB", AFL: "AFL", J: "J"}


@dataclass
class RecordRR:
    record: str
    fs: float
    beat_source: str      # "qrsc" or "qrs"
    has_signal: bool
    beats: np.ndarray     # (n_beats,) sample index of each beat
    rr: np.ndarray        # (n_beats - 1,) seconds
    rhythm: np.ndarray    # (n_beats - 1,) rhythm code of each RR interval
    valid: np.ndarray     # (n_beats - 1,) bool, RR within [RR_MIN, RR_MAX]


def rhythm_changes(ann: wfdb.Annotation) -> tuple[np.ndarray, np.ndarray]:
    """Return (change samples, rhythm codes) from an .atr annotation.

    Each rhythm holds from its change sample until the next change.
    Unrecognised rhythm strings raise, so new labels are never silently mixed in.
    """
    samples, codes = [], []
    for s, note in zip(ann.sample, ann.aux_note):
        note = note.strip().rstrip("\x00")
        if not note:
            continue
        if note not in RHYTHM_CODES:
            raise ValueError(f"unknown rhythm annotation {note!r} at sample {s}")
        samples.append(s)
        codes.append(RHYTHM_CODES[note])
    order = np.argsort(samples, kind="stable")
    return np.asarray(samples, dtype=np.int64)[order], np.asarray(codes, dtype=np.int8)[order]


def rhythm_at(samples: np.ndarray, change_samples: np.ndarray,
              change_codes: np.ndarray) -> np.ndarray:
    """Rhythm code in effect at each sample (UNKNOWN before the first change)."""
    idx = np.searchsorted(change_samples, samples, side="right") - 1
    out = np.full(len(samples), UNKNOWN, dtype=np.int8)
    ok = idx >= 0
    out[ok] = change_codes[idx[ok]]
    return out


def load_record_rr(record: str, data_dir: Path = DEFAULT_DIR) -> RecordRR:
    path = str(Path(data_dir) / record)
    header = wfdb.rdheader(path)
    fs = float(header.fs)
    has_signal = header.sig_len > 0

    source = "qrsc" if (Path(data_dir) / f"{record}.qrsc").exists() else "qrs"
    beats = np.asarray(wfdb.rdann(path, source).sample, dtype=np.int64)
    beats = np.unique(beats)  # sorted, no duplicates

    end = int(ANNOTATED_SECONDS * fs)
    if has_signal:
        end = min(end, header.sig_len)
    beats = beats[beats < end]

    change_samples, change_codes = rhythm_changes(wfdb.rdann(path, "atr"))
    rr = np.diff(beats) / fs
    rhythm = rhythm_at(beats[1:], change_samples, change_codes)
    valid = (rr >= RR_MIN) & (rr <= RR_MAX)
    return RecordRR(record, fs, source, has_signal, beats, rr, rhythm, valid)


def list_records(data_dir: Path = DEFAULT_DIR) -> list[str]:
    return (Path(data_dir) / "RECORDS").read_text().split()


def load_all(data_dir: Path = DEFAULT_DIR) -> list[RecordRR]:
    return [load_record_rr(r, data_dir) for r in list_records(data_dir)]


def save_rr(records: list[RecordRR], out: Path) -> None:
    """Save all records as flat arrays plus per-record offsets.

    RR intervals of record k are ``rr[offsets[k]:offsets[k+1]]``; the same
    offsets index ``rhythm``, ``valid`` and ``rr_end_sample``.
    """
    offsets = np.cumsum([0] + [len(r.rr) for r in records])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        record=np.array([r.record for r in records]),
        fs=np.array([r.fs for r in records]),
        beat_source=np.array([r.beat_source for r in records]),
        has_signal=np.array([r.has_signal for r in records]),
        offsets=offsets,
        rr=np.concatenate([r.rr for r in records]).astype(np.float32),
        rr_end_sample=np.concatenate([r.beats[1:] for r in records]),
        rhythm=np.concatenate([r.rhythm for r in records]),
        valid=np.concatenate([r.valid for r in records]),
    )


def summarize(rec: RecordRR) -> dict:
    """Per-record statistics; rhythm shares are by time (sum of RR)."""
    total = rec.rr.sum()
    row = {
        "record": rec.record,
        "beat_source": rec.beat_source,
        "has_signal": rec.has_signal,
        "hours": round(total / 3600, 2),
        "n_rr": len(rec.rr),
        "n_invalid_rr": int((~rec.valid).sum()),
        "mean_hr_bpm": round(60 / np.median(rec.rr[rec.valid]), 1),
    }
    for code, name in RHYTHM_NAMES.items():
        row[f"pct_{name}"] = round(100 * rec.rr[rec.rhythm == code].sum() / total, 2)
    row["n_af_episodes"] = int(np.sum(np.diff(np.r_[0, rec.rhythm == AFIB].astype(int)) == 1))
    return row
