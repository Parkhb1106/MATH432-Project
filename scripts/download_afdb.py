"""Download the MIT-BIH Atrial Fibrillation Database (AFDB) from PhysioNet.

Usage:
    python scripts/download_afdb.py [--out data/mitbih_afdb]

Files are fetched from PhysioNet's open-data S3 mirror (the same place
physionet.org/files/... redirects to) using the database's SHA256SUMS.txt as
the file list, so every file is checksum-verified. Already-downloaded files
with a matching checksum are skipped. Prints a per-record inventory at the end.

(wfdb.dl_database goes through physionet.org/content/..., which can return
502 errors; this script avoids that endpoint.)
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://physionet-open.s3.amazonaws.com/afdb/1.0.0/"
# Records that have only annotation files (no ECG signal) in AFDB.
NO_SIGNAL = {"00735", "03665"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def download(out: Path) -> None:
    sums = fetch(BASE + "SHA256SUMS.txt").decode().split("\n")
    entries = [line.split(maxsplit=1) for line in sums if line.strip()]
    for i, (digest, name) in enumerate(sorted(entries, key=lambda e: e[1]), 1):
        dest = out / name
        if dest.exists() and sha256(dest) == digest:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(entries)}] {name}", flush=True)
        data = fetch(BASE + name)
        if hashlib.sha256(data).hexdigest() != digest:
            sys.exit(f"checksum mismatch: {name}")
        dest.write_bytes(data)


def inventory(out: Path) -> None:
    records = (out / "RECORDS").read_text().split()
    print(f"\n{'record':>8}  hea  dat  atr  qrs  qrsc")
    for r in records:
        flags = ["Y" if (out / f"{r}.{ext}").exists() else "-"
                 for ext in ("hea", "dat", "atr", "qrs", "qrsc")]
        print(f"{r:>8}  " + "    ".join(flags))
    n_sig = sum((out / f"{r}.dat").exists() for r in records)
    n_qrsc = sum((out / f"{r}.qrsc").exists() for r in records)
    print(f"\n{len(records)} records, {n_sig} with signal, {n_qrsc} with .qrsc")
    missing = [r for r in records
               if r not in NO_SIGNAL and not (out / f"{r}.dat").exists()]
    if missing:
        print(f"WARNING: signal missing for {missing}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/mitbih_afdb"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    download(args.out)
    inventory(args.out)


if __name__ == "__main__":
    main()
