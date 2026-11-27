"""Compute A-part features for a segment dataset.

Usage:
    python scripts/build_features.py                       # AFDB, defaults
    python scripts/build_features.py --d 2 --normalize zscore

Reads:
    data/processed/T_afdb.npz        (from scripts/build_afdb.py)
Writes:
    data/processed/features_afdb_{tag}.csv      one row per segment: scalar
        features + record, label_50, label_excl, af_frac
    data/processed/diagrams_afdb_{tag}.npz      cached H0/H1 diagrams (packed)
        for persistence images / landscapes, which are fitted per training fold
where tag = d{d}_tau{tau}_{normalize}. Only the topology columns depend on the
tag; the other features are identical across tags.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.entropy import entropy_features  # noqa: E402
from src.features.hrv import hrv_features  # noqa: E402
from src.features.scaling import scaling_features  # noqa: E402
from src.features.topology import (diagram_summary, pack_diagrams,  # noqa: E402
                                   segment_diagrams)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="afdb")
ap.add_argument("--d", type=int, default=3)
ap.add_argument("--tau", type=int, default=1)
ap.add_argument("--normalize", default="none", choices=["none", "zscore"])
args = ap.parse_args()

src = Path(f"data/processed/T_{args.dataset}.npz")
seg = np.load(src)
rr = seg["rr"].astype(float)
print(f"{src}: {len(rr)} segments x {rr.shape[1]} intervals")

t0 = time.perf_counter()
rows, h0, h1 = [], [], []
for x in rr:
    dgms = segment_diagrams(x, d=args.d, tau=args.tau, normalize=args.normalize)
    h0.append(dgms[0])
    h1.append(dgms[1])
    f = {}
    f.update(scaling_features(x))
    f.update(entropy_features(x))
    f.update(diagram_summary(dgms))
    f.update(hrv_features(x))
    rows.append(f)
print(f"features done in {time.perf_counter() - t0:.0f} s")

tag = f"d{args.d}_tau{args.tau}_{args.normalize}"
meta = pd.DataFrame({k: seg[k] for k in ("record", "label_50", "label_excl", "af_frac")})
df = pd.concat([meta, pd.DataFrame(rows)], axis=1)
out = Path(f"data/processed/features_{args.dataset}_{tag}.csv")
df.to_csv(out, index=False)
print(f"wrote {out} ({df.shape[0]} x {df.shape[1]})")

p0, o0 = pack_diagrams(h0)
p1, o1 = pack_diagrams(h1)
dg_out = Path(f"data/processed/diagrams_{args.dataset}_{tag}.npz")
np.savez_compressed(dg_out, h0_pts=p0, h0_off=o0, h1_pts=p1, h1_off=o1,
                    record=seg["record"])
print(f"wrote {dg_out}")
print(f"SampEn undefined in {df['ent_sampen_undefined'].mean():.1%} of segments, "
      f"COSEn undefined in {df['ent_cosen_undefined'].mean():.1%}")
