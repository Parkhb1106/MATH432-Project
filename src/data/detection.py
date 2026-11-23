"""Compare automatic R-peak detection (neurokit2) with AFDB reference beats.

Rules (agreed 2026-11-23):
* Detector: ``nk.ecg_clean`` + ``nk.ecg_peaks`` (neurokit2 defaults), run on
  each lead (ECG1, ECG2) separately.
* Reference: the beats used for RR extraction (``.qrsc`` if present, else
  ``.qrs``), over the same annotated range.
* Matching: a detected beat matches a reference beat within 150 ms
  (ANSI/AAMI EC57), via ``wfdb.processing.compare_annotations``.
* Rhythm split: AF = AFIB, non-AF = N or J; AFL beats are left out of the
  split (they still count in the overall Se/PPV).
* RR error: for consecutive reference beats that are both matched, the
  difference between detected RR and reference RR, in ms.
"""

from __future__ import annotations

from pathlib import Path

import neurokit2 as nk
import numpy as np
import wfdb
from wfdb.processing import compare_annotations

from src.data.afdb import AFIB, DEFAULT_DIR, J, N, RecordRR, rhythm_at, rhythm_changes

TOLERANCE_S = 0.150


def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
    return np.asarray(info["ECG_R_Peaks"], dtype=np.int64)


def compare_beats(ref: np.ndarray, det: np.ndarray, fs: float,
                  ref_rhythm: np.ndarray, det_rhythm: np.ndarray) -> dict:
    window = int(TOLERANCE_S * fs)
    cmp = compare_annotations(ref, det, window)
    matched = cmp.matching_sample_nums  # det index for each ref beat, -1 if none
    ref_hit = matched >= 0
    det_hit = np.zeros(len(det), bool)
    det_hit[matched[ref_hit]] = True

    row = {"n_ref": len(ref), "n_det": len(det),
           "se": ref_hit.mean(), "ppv": det_hit.mean()}
    for name, codes in (("af", [AFIB]), ("nonaf", [N, J])):
        r_in = np.isin(ref_rhythm, codes)
        d_in = np.isin(det_rhythm, codes)
        row[f"se_{name}"] = ref_hit[r_in].mean() if r_in.any() else np.nan
        row[f"ppv_{name}"] = det_hit[d_in].mean() if d_in.any() else np.nan

    both = ref_hit[:-1] & ref_hit[1:]
    ref_rr = np.diff(ref)[both]
    det_rr = det[matched[1:][both]] - det[matched[:-1][both]]
    err_ms = np.abs(det_rr - ref_rr) / fs * 1000
    row["rr_err_med_ms"] = np.median(err_ms) if len(err_ms) else np.nan
    row["rr_err_p95_ms"] = np.percentile(err_ms, 95) if len(err_ms) else np.nan
    return row


def compare_record(rec: RecordRR, data_dir: Path = DEFAULT_DIR) -> list[dict]:
    """One row per lead for a record that has a signal."""
    path = str(Path(data_dir) / rec.record)
    end = int(rec.beats[-1]) + int(rec.fs)  # cover the last reference beat
    record = wfdb.rdrecord(path, sampto=min(end, wfdb.rdheader(path).sig_len))
    change_s, change_c = rhythm_changes(wfdb.rdann(path, "atr"))
    ref_rhythm = rhythm_at(rec.beats, change_s, change_c)

    rows = []
    for lead, name in enumerate(record.sig_name):
        det = detect_r_peaks(record.p_signal[:, lead], rec.fs)
        det = det[det < rec.beats[-1] + int(TOLERANCE_S * rec.fs) + 1]
        det_rhythm = rhythm_at(det, change_s, change_c)
        row = {"record": rec.record, "lead": name, "ref_source": rec.beat_source}
        row.update(compare_beats(rec.beats, det, rec.fs, ref_rhythm, det_rhythm))
        rows.append(row)
    return rows
