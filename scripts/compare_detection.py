"""Compare neurokit2 R-peak detection with AFDB reference beats (both leads).

Usage:
    python scripts/compare_detection.py

Writes:
    results/afdb_detection_compare.csv   one row per (record, lead)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.afdb import load_all  # noqa: E402
from src.data.detection import compare_record  # noqa: E402

rows = []
for rec in load_all():
    if not rec.has_signal:
        continue
    print(rec.record, flush=True)
    rows.extend(compare_record(rec))

table = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
table.to_csv("results/afdb_detection_compare.csv", index=False, float_format="%.4f")

pd.set_option("display.width", 220)
print(table.round(4).to_string(index=False))
print("\nmedian over records, by lead and reference source:")
cols = ["se", "ppv", "se_af", "ppv_af", "se_nonaf", "ppv_nonaf",
        "rr_err_med_ms", "rr_err_p95_ms"]
print(table.groupby(["lead", "ref_source"])[cols].median().round(4).to_string())
