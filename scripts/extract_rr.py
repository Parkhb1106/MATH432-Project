"""Extract RR series and rhythm labels for every AFDB record.

Usage:
    python scripts/extract_rr.py

Writes:
    data/processed/afdb_rr.npz              flat RR arrays (see src.data.afdb.save_rr)
    results/afdb_record_summary.csv         per-record rhythm shares and RR quality
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.afdb import load_all, save_rr, summarize  # noqa: E402

records = load_all()
save_rr(records, Path("data/processed/afdb_rr.npz"))

table = pd.DataFrame([summarize(r) for r in records])
Path("results").mkdir(exist_ok=True)
table.to_csv("results/afdb_record_summary.csv", index=False)

pd.set_option("display.width", 200)
print(table.to_string(index=False))
total_h = table["hours"].sum()
print(f"\ntotal {total_h:.1f} h, {table['n_rr'].sum()} RR intervals, "
      f"{table['n_invalid_rr'].sum()} invalid")
for name in ["N", "AFIB", "AFL", "J", "unknown"]:
    h = (table["hours"] * table[f"pct_{name}"] / 100).sum()
    print(f"  {name:>7}: {h:6.1f} h ({100 * h / total_h:.1f}%)")
