"""Download selected subjects of the Zenodo wrist PPG / ECG AF dataset.

Usage:
    python scripts/download_zenodo.py --subjects 006 012 028
    python scripts/download_zenodo.py --subjects 001-010
    python scripts/download_zenodo.py --all
    python scripts/download_zenodo.py --subjects 001-010 --dry-run

For each selected subject both ``XXX_ECG.mat`` and ``XXX_PPG.mat`` are fetched
(AF labels live only in the ECG file). Subjects without PPG (024, 025, 034,
037) get the ECG file and a warning. ``subject_info.xlsx`` and ``LICENSE.txt``
are always fetched.

* The file list, sizes and MD5 checksums come from the Zenodo API; the record
  is pinned (default 23921320) so results do not change when a new version
  is published.
* Files that already exist with the right MD5 are skipped. Partial files are
  resumed with HTTP Range requests, and dropped connections are retried.
* Before downloading, the remaining size is compared with free disk space and
  the script stops if it does not fit.
"""

import argparse
import hashlib
import http.client
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

RECORD = "23921320"
ALWAYS = ["subject_info.xlsx", "LICENSE.txt"]
CHUNK = 1 << 20
RETRIES = 5
PROGRESS_EVERY = 200 << 20  # print progress every 200 MB


def fetch_file_list(record: str) -> dict[str, dict]:
    """{file name: {"size": bytes, "md5": hex}} for a Zenodo record."""
    url = f"https://zenodo.org/api/records/{record}"
    with urllib.request.urlopen(url, timeout=60) as r:
        meta = json.load(r)
    return {f["key"]: {"size": f["size"], "md5": f["checksum"].removeprefix("md5:")}
            for f in meta["files"]}


def parse_subjects(tokens: list[str]) -> list[str]:
    """'6', '006', '001-010' -> sorted zero-padded subject IDs."""
    ids = set()
    for tok in tokens:
        if "-" in tok:
            lo, hi = (int(x) for x in tok.split("-", 1))
            if lo > hi:
                raise ValueError(f"bad range {tok!r}")
            ids.update(range(lo, hi + 1))
        else:
            ids.add(int(tok))
    return [f"{i:03d}" for i in sorted(ids)]


def select_files(subjects: list[str], files: dict[str, dict]) -> tuple[list[str], list[str]]:
    """Return (file names to download, warnings). Raises on unknown subjects."""
    available = sorted(k[:3] for k in files if k.endswith("_ECG.mat"))
    unknown = [s for s in subjects if s not in available]
    if unknown:
        raise ValueError(f"unknown subjects {unknown}; available: {available}")
    names, warnings = list(ALWAYS), []
    for s in subjects:
        names.append(f"{s}_ECG.mat")
        if f"{s}_PPG.mat" in files:
            names.append(f"{s}_PPG.mat")
        else:
            warnings.append(f"{s}: no PPG file in this record (ECG only)")
    return names, warnings


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def is_complete(path: Path, info: dict) -> bool:
    return path.exists() and path.stat().st_size == info["size"] and md5sum(path) == info["md5"]


def remaining_bytes(path: Path, info: dict) -> int:
    have = path.stat().st_size if path.exists() else 0
    return info["size"] - have if have <= info["size"] else info["size"]


def download(url: str, dest: Path, info: dict) -> None:
    """Download with Range resume and retries, then verify MD5."""
    # Called only for incomplete files, so a full-size file has a bad MD5.
    if dest.exists() and dest.stat().st_size >= info["size"]:
        dest.unlink()
    for attempt in range(1, RETRIES + 1):
        have = dest.stat().st_size if dest.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                done = have if mode == "ab" else 0
                next_report = done + PROGRESS_EVERY
                with open(dest, mode) as f:
                    for chunk in iter(lambda: r.read(CHUNK), b""):
                        f.write(chunk)
                        done += len(chunk)
                        if done >= next_report:
                            print(f"    {done / 1e6:,.0f} / {info['size'] / 1e6:,.0f} MB", flush=True)
                            next_report += PROGRESS_EVERY
            if dest.stat().st_size == info["size"]:
                break
        except (OSError, http.client.HTTPException) as e:
            print(f"    attempt {attempt}/{RETRIES} interrupted: {e}", flush=True)
        time.sleep(min(2 ** attempt, 30))
    else:
        sys.exit(f"failed to download {dest.name} after {RETRIES} attempts")
    if md5sum(dest) != info["md5"]:
        dest.unlink()
        sys.exit(f"MD5 mismatch for {dest.name}; file removed, run again")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pick = ap.add_mutually_exclusive_group(required=True)
    pick.add_argument("--subjects", nargs="+", help="IDs or ranges, e.g. 006 012 001-010")
    pick.add_argument("--all", action="store_true", help="all subjects in the record")
    ap.add_argument("--dry-run", action="store_true", help="list files and sizes only")
    ap.add_argument("--record", default=RECORD, help=f"Zenodo record ID (default {RECORD})")
    ap.add_argument("--out", type=Path, default=Path("data/zenodo_ppg_af"))
    args = ap.parse_args()

    files = fetch_file_list(args.record)
    if args.all:
        subjects = sorted(k[:3] for k in files if k.endswith("_ECG.mat"))
    else:
        subjects = parse_subjects(args.subjects)
    try:
        names, warnings = select_files(subjects, files)
    except ValueError as e:
        sys.exit(str(e))

    total = sum(files[n]["size"] for n in names)
    need = sum(remaining_bytes(args.out / n, files[n]) for n in names)
    print(f"record {args.record}: {len(subjects)} subjects, {len(names)} files, "
          f"{total / 1e9:.2f} GB total, up to {need / 1e9:.2f} GB still to download")
    for w in warnings:
        print(f"WARNING: {w}")
    if args.dry_run:
        for n in names:
            print(f"  {n:20s} {files[n]['size'] / 1e6:10,.1f} MB")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.out).free
    if need > free:
        sys.exit(f"not enough disk space in {args.out}: need {need / 1e9:.2f} GB, "
                 f"free {free / 1e9:.2f} GB (use --out or fewer subjects)")

    for i, n in enumerate(names, 1):
        dest = args.out / n
        if is_complete(dest, files[n]):
            print(f"[{i}/{len(names)}] {n} already complete")
            continue
        print(f"[{i}/{len(names)}] {n} ({files[n]['size'] / 1e6:,.1f} MB)", flush=True)
        download(f"https://zenodo.org/records/{args.record}/files/{n}?download=1", dest, files[n])

    print(f"\n{'subject':>7}  ECG  PPG")
    for s in subjects:
        ecg = "Y" if (args.out / f"{s}_ECG.mat").exists() else "-"
        ppg = "Y" if (args.out / f"{s}_PPG.mat").exists() else ("none" if f"{s}_PPG.mat" not in files else "-")
        print(f"{s:>7}  {ecg:3s}  {ppg}")


if __name__ == "__main__":
    main()
