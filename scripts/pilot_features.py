"""Week-1 pilot on AFDB: are the features well defined, and which embedding
settings (d, normalize) should the topology features use?

Usage:
    python scripts/pilot_features.py
    (after scripts/build_features.py for every tag in TAGS)

Writes results/pilot_afdb/:
    undefined_rates.csv     share of segments where SampEn / COSEn are undefined
    univariate_auroc.csv    per-feature AUROC for AF (label_50), direction-free
    family_cv.csv           5-fold patient-level CV (results/folds_afdb.json)
                            per feature family x model x embedding tag
and results/figures/pilot_*.png.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import family_columns  # noqa: E402
from src.features.topology import (PersistenceImage, delay_embed,  # noqa: E402
                                   segment_diagrams, unpack_diagrams)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

TAGS = ["d2_tau1_none", "d3_tau1_none", "d2_tau1_zscore", "d3_tau1_zscore"]
OUT = Path("results/pilot_afdb")
FIG = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

folds = json.loads(Path("results/folds_afdb.json").read_text())
dfs = {t: pd.read_csv(f"data/processed/features_afdb_{t}.csv", dtype={"record": str}) for t in TAGS}
base = dfs[TAGS[0]]
y = base["label_50"].to_numpy()
fold = base["record"].map(folds).to_numpy()
assert not np.isnan(fold.astype(float)).any(), "record missing from folds"
print(f"{len(base)} segments, AF {y.mean():.1%}")

# ------------------------------------------------------------- 1. undefined
und = base[["ent_sampen_undefined", "ent_cosen_undefined"]].mean().rename("share")
und_by_class = base.groupby("label_50")[["ent_sampen_undefined", "ent_cosen_undefined"]].mean()
und_by_class.index = ["non-AF", "AF"]
und_by_class.to_csv(OUT / "undefined_rates.csv")
print("\nundefined rates by class:\n", und_by_class.round(4))

# ------------------------------------------------------------- 2. univariate
rows = []
shared = [c for c in base.columns if c[:4] in ("dfa_", "ent_", "hrv_") and not c.endswith("_undefined")]
for c in shared:
    v = base[c].fillna(base[c].median())
    a = roc_auc_score(y, v)
    rows.append({"feature": c, "tag": "-", "auroc": max(a, 1 - a), "higher_in_af": a > 0.5})
for t, df in dfs.items():
    for c in family_columns(df.columns, "topology"):
        a = roc_auc_score(y, df[c].fillna(df[c].median()))
        rows.append({"feature": c, "tag": t, "auroc": max(a, 1 - a), "higher_in_af": a > 0.5})
uni = pd.DataFrame(rows).sort_values("auroc", ascending=False)
uni.to_csv(OUT / "univariate_auroc.csv", index=False)
print("\nunivariate AUROC (top 15):\n", uni.head(15).round(3).to_string(index=False))


# ------------------------------------------------------------- 3. family CV
def models():
    return {
        "LR": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(max_iter=3000, class_weight="balanced")),
        "RF": make_pipeline(SimpleImputer(strategy="median"),
                            RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                   class_weight="balanced", n_jobs=-1,
                                                   random_state=0)),
    }


def cv_eval(X_or_fn, name, tag, n_feat):
    out = []
    for mname in ("LR", "RF"):
        per = []
        for k in range(5):
            tr, te = fold != k, fold == k
            Xtr, Xte = X_or_fn(tr, te) if callable(X_or_fn) else (X_or_fn[tr], X_or_fn[te])
            m = models()[mname].fit(Xtr, y[tr])
            p = m.predict_proba(Xte)[:, 1]
            yhat = (p >= 0.5).astype(int)
            per.append([roc_auc_score(y[te], p), f1_score(y[te], yhat),
                        recall_score(y[te], yhat), recall_score(y[te], yhat, pos_label=0)])
        per = np.array(per)
        out.append({"family": name, "tag": tag, "model": mname, "n_features": n_feat,
                    "auroc": per[:, 0].mean(), "auroc_sd": per[:, 0].std(),
                    "f1": per[:, 1].mean(), "se": per[:, 2].mean(), "sp": per[:, 3].mean()})
        print(f"  {name:22s} {tag:15s} {mname}  AUROC {out[-1]['auroc']:.3f} ± {out[-1]['auroc_sd']:.3f}"
              f"  F1 {out[-1]['f1']:.3f}")
    return out


res = []
print("\nfamily CV:")
cols = {f: family_columns(base.columns, f) for f in ("hrv", "scaling", "entropy")}
for fam, cs in cols.items():
    res += cv_eval(base[cs].to_numpy(), fam, "-", len(cs))
se = cols["scaling"] + cols["entropy"]
res += cv_eval(base[se].to_numpy(), "scaling+entropy", "-", len(se))

for t, df in dfs.items():
    top = family_columns(df.columns, "topology")
    res += cv_eval(df[top].to_numpy(), "topology_summary", t, len(top))
    nl = se + top
    res += cv_eval(df[nl].to_numpy(), "scaling+entropy+topology", t, len(nl))
    ht = cols["hrv"] + top
    res += cv_eval(df[ht].to_numpy(), "hrv+topology", t, len(ht))

    dg = np.load(f"data/processed/diagrams_afdb_{t}.npz")
    h0 = unpack_diagrams(dg["h0_pts"], dg["h0_off"])
    h1 = unpack_diagrams(dg["h1_pts"], dg["h1_off"])

    def pi_features(tr, te, h0=h0, h1=h1):
        idx_tr, idx_te = np.where(tr)[0], np.where(te)[0]
        parts_tr, parts_te = [], []
        for dg_list in (h0, h1):
            pi = PersistenceImage(res=10).fit([dg_list[i] for i in idx_tr])
            parts_tr.append(pi.transform([dg_list[i] for i in idx_tr]))
            parts_te.append(pi.transform([dg_list[i] for i in idx_te]))
        return np.hstack(parts_tr), np.hstack(parts_te)

    res += cv_eval(pi_features, "topology_PI(H0+H1)", t, 10 + 100)  # H0 1x10, H1 10x10

cv = pd.DataFrame(res)
cv.to_csv(OUT / "family_cv.csv", index=False)

# ------------------------------------------------------------- 4. figures
show = ["dfa_alpha1", "ent_sampen", "ent_cosen", "ent_mpe_s1", "hrv_rmssd",
        "top_h0_total", "top_h1_count", "top_h1_max"]
df3 = dfs["d3_tau1_none"]
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for ax, c in zip(axes.ravel(), show):
    data = [df3.loc[df3.label_50 == k, c].dropna() for k in (0, 1)]
    ax.boxplot(data, tick_labels=["non-AF", "AF"], showfliers=False)
    a = uni.loc[uni.feature == c, "auroc"]
    a = a.iloc[0] if c[:4] != "top_" else uni.loc[(uni.feature == c) & (uni.tag == "d3_tau1_none"), "auroc"].iloc[0]
    ax.set_title(f"{c}\nAUROC {a:.3f}")
    ax.grid(alpha=0.3)
fig.suptitle("AFDB 특징 분포 (위상 특징은 d=3, τ=1, 초 단위)")
fig.tight_layout()
fig.savefig(FIG / "pilot_feature_boxplots.png", dpi=120)

# example segments: clearest non-AF and AF segment (af_frac 0 / 1) of one record with both
seg = np.load("data/processed/T_afdb.npz")
rec = base.groupby("record")["label_50"].mean()
r_mix = rec[(rec > 0.3) & (rec < 0.7)].index[0]
i_n = base.index[(base.record == r_mix) & (base.af_frac == 0)][0]
i_a = base.index[(base.record == r_mix) & (base.af_frac == 1)][0]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for row, (i, name) in enumerate([(i_n, "non-AF"), (i_a, "AF")]):
    rr = seg["rr"][i].astype(float)
    axes[row, 0].plot(rr, "o-", ms=3)
    axes[row, 0].set_ylim(0.3, 1.5)
    axes[row, 0].set_title(f"{name}: 기록 {r_mix}, 구간 {i} RR (초)")
    emb = delay_embed(rr, 2, 1)
    axes[row, 1].scatter(emb[:, 0], emb[:, 1], s=10)
    axes[row, 1].set_xlim(0.3, 1.5)
    axes[row, 1].set_ylim(0.3, 1.5)
    axes[row, 1].set_aspect("equal")
    axes[row, 1].set_title("지연 임베딩 d=2 (Poincaré)")
    dgms = segment_diagrams(rr, d=3)
    for dim, dgm in enumerate(dgms):
        axes[row, 2].scatter(dgm[:, 0], dgm[:, 1], s=12, label=f"H{dim}")
    axes[row, 2].plot([0, 0.6], [0, 0.6], "k--", lw=0.8)
    axes[row, 2].set_xlim(-0.02, 0.6)
    axes[row, 2].set_ylim(-0.02, 0.6)
    axes[row, 2].set_title("persistence diagram (d=3, 초 단위)")
    axes[row, 2].legend()
for ax in axes.ravel():
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "pilot_example_segments.png", dpi=120)
print("\nwrote", OUT, "and figures")
