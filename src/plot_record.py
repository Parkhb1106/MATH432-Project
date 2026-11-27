"""MIT-BIH 레코드 하나를 내려받아 신호와 주석을 함께 그린다.

사용법: python src/plot_record.py [레코드번호]   (기본값 119)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "mitdb"
FIG_DIR = ROOT / "results" / "figures"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# AAMI 클래스 매핑 (비트가 아닌 기호는 제외)
AAMI = {**dict.fromkeys("NLRej", "N"), **dict.fromkeys("aAJS", "S"),
        **dict.fromkeys("VE", "V"), "F": "F", **dict.fromkeys("/fQ", "Q")}
COLORS = {"N": "#2b6cb0", "S": "#d69e2e", "V": "#c53030", "F": "#6b46c1", "Q": "#718096"}


def main(rec_name: str = "119") -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not (DATA_DIR / f"{rec_name}.hea").exists():
        wfdb.dl_database("mitdb", str(DATA_DIR), records=[rec_name])

    rec = wfdb.rdrecord(str(DATA_DIR / rec_name))
    ann = wfdb.rdann(str(DATA_DIR / rec_name), "atr")
    fs = rec.fs
    ch = rec.sig_name.index("MLII")
    x = rec.p_signal[:, ch]

    beats = [(s, AAMI[sym]) for s, sym in zip(ann.sample, ann.symbol) if sym in AAMI]
    b_samp = np.array([s for s, _ in beats])
    b_cls = np.array([c for _, c in beats])
    print(f"record {rec_name}: fs={fs} Hz, {len(x)} samples ({len(x)/fs/60:.1f} min), leads={rec.sig_name}")
    print("beat counts:", {c: int((b_cls == c).sum()) for c in COLORS if (b_cls == c).any()})

    # 정상과 비정상 비트가 섞인 10초 구간 고르기
    win = 10 * fs
    abn = b_samp[b_cls != "N"]
    start = max(int(abn[len(abn) // 3]) - 3 * fs, 0) if len(abn) else 60 * fs
    end = start + win

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.45, wspace=0.2)

    # (a) 10초 구간 원신호 + 주석
    ax = fig.add_subplot(gs[0, :])
    t = np.arange(start, end) / fs
    ax.plot(t, x[start:end], color="black", lw=0.8)
    for s, c in zip(b_samp, b_cls):
        if start <= s < end:
            ax.plot(s / fs, x[s], "o", color=COLORS[c], ms=6)
            ax.text(s / fs, x[s] + 0.25, c, color=COLORS[c], ha="center", fontsize=11, fontweight="bold")
    ax.set_title(f"(a) 레코드 {rec_name}, MLII 리드, {start/fs:.0f}–{end/fs:.0f}초 구간 (점 = 주석의 비트 위치)")
    ax.set_xlabel("시간 (초)")
    ax.set_ylabel("전압 (mV)")
    ax.grid(alpha=0.3)

    # (b) 비트 분할: R-peak 기준 -0.25 ~ +0.45초
    pre, post = int(0.25 * fs), int(0.45 * fs)
    tb = np.arange(-pre, post) / fs * 1000
    for i, cls in enumerate(["N", "V"]):
        ax = fig.add_subplot(gs[1, i])
        idx = [s for s, c in zip(b_samp, b_cls) if c == cls and pre <= s < len(x) - post]
        segs = np.array([x[s - pre:s + post] for s in idx[:200]])
        if len(segs) == 0:
            ax.set_title(f"(b) {cls} 비트 없음")
            continue
        ax.plot(tb, segs.T, color=COLORS[cls], alpha=0.05, lw=0.8)
        ax.plot(tb, segs.mean(0), color="black", lw=2, label="평균 파형")
        name = "정상 (N)" if cls == "N" else "심실 조기수축 (V)"
        ax.set_title(f"(b{i+1}) {name} 비트 {len(segs)}개 겹쳐 그리기")
        ax.set_xlabel("R-peak 기준 시간 (ms)")
        ax.set_ylabel("전압 (mV)")
        ax.axvline(0, color="gray", ls="--", lw=0.8)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    # (c) 30분 전체 RR 간격 (심박 간격) 추이
    ax = fig.add_subplot(gs[2, :])
    rr = np.diff(b_samp) / fs
    for cls in COLORS:
        m = b_cls[1:] == cls
        if m.any():
            ax.scatter(b_samp[1:][m] / fs / 60, rr[m], s=3, color=COLORS[cls], label=cls)
    ax.set_title("(c) 30분 전체 RR 간격 (직전 비트와의 시간 간격, 색 = 비트 종류)")
    ax.set_xlabel("시간 (분)")
    ax.set_ylabel("RR 간격 (초)")
    ax.legend(markerscale=4, loc="upper right")
    ax.grid(alpha=0.3)

    out = FIG_DIR / f"record_{rec_name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "119")
