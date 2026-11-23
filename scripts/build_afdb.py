"""Build the AFDB segment dataset and patient-level folds.

Usage:
    python scripts/build_afdb.py

Writes:
    data/processed/T_afdb.npz     60-beat segments (see src.data.segment)
    results/folds_afdb.json       {record: fold}, shared by A and B
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.afdb import load_all  # noqa: E402
from src.data.folds import make_folds, save_folds  # noqa: E402
from src.data.segment import save_segments, segment_all  # noqa: E402

seg = segment_all(load_all())
save_segments(seg, Path("data/processed/T_afdb.npz"))

folds = make_folds(seg["record"], seg["label_50"])
save_folds(folds, Path("results/folds_afdb.json"))

df = pd.DataFrame({"record": seg["record"], "label_50": seg["label_50"],
                   "label_excl": seg["label_excl"]})
df["fold"] = df["record"].map(folds)
print(f"{len(df)} segments, AF (label_50) {df['label_50'].mean():.1%}, "
      f"ambiguous (label_excl=-1) {(df['label_excl'] == -1).mean():.1%}\n")
table = df.groupby("fold").agg(
    records=("record", lambda s: " ".join(sorted(s.unique()))),
    n_seg=("record", "size"),
    n_af=("label_50", "sum"),
    af_pct=("label_50", lambda s: round(100 * s.mean(), 1)),
)
pd.set_option("display.width", 200)
print(table.to_string())
