"""
HANCOCK Patient Slide Downloader
==================================
Downloads primary tumour and lymph node HE slide thumbnails for all 763 patients.

Image URL patterns:
  https://hancock.research.fau.eu/public/images/ThumbnailsWebsite/primarytumor_HE/PrimaryTumor_HE_NNN.jpg
  https://hancock.research.fau.eu/public/images/ThumbnailsWebsite/lymphnode_HE/LymphNode_HE_NNN.jpg

Requirements:
    pip install requests tqdm

Usage:
    python hancock_scraper.py                      # all 763 patients
    python hancock_scraper.py --start 1 --end 50   # specific range
    python hancock_scraper.py --out ./images        # custom output folder
    python hancock_scraper.py --workers 8           # more parallel downloads
"""

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
BASE_URL   = "https://hancock.research.fau.eu"
TOTAL_PATIENTS = 763
HEADERS    = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

IMAGE_TYPES = [
    {
        "name":    "primarytumor_HE",
        "url_tpl": BASE_URL + "/public/images/ThumbnailsWebsite/primarytumor_HE/PrimaryTumor_HE_{n:03d}.jpg",
        "file":    "PrimaryTumor_HE_{n:03d}.jpg",
    },
    {
        "name":    "lymphnode_HE",
        "url_tpl": BASE_URL + "/public/images/ThumbnailsWebsite/lymphnode_HE/LymphNode_HE_{n:03d}.jpg",
        "file":    "LymphNode_HE_{n:03d}.jpg",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════
def download_one(url: str, dest: Path, session: requests.Session):
    """Download a single file. Returns (url, dest, ok, status_code)."""
    if dest.exists():
        return url, dest, True, "cached"
    try:
        r = session.get(url, headers=HEADERS, timeout=60, stream=True)
        if r.status_code == 404:
            return url, dest, False, 404
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        return url, dest, True, r.status_code
    except Exception as e:
        return url, dest, False, str(e)


def build_tasks(start: int, end: int, out_root: Path):
    """Build the full list of (url, dest_path) pairs."""
    tasks = []
    for pid in range(start, end + 1):
        patient_dir = out_root / f"patient_{pid:03d}"
        for img_type in IMAGE_TYPES:
            url  = img_type["url_tpl"].format(n=pid)
            dest = patient_dir / img_type["file"].format(n=pid)
            tasks.append((url, dest))
    return tasks


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Download HANCOCK slide thumbnails")
    parser.add_argument("--start",   type=int, default=1)
    parser.add_argument("--end",     type=int, default=TOTAL_PATIENTS)
    parser.add_argument("--out",     default="hancock_images", help="Root output directory")
    parser.add_argument("--workers", type=int, default=4,      help="Parallel download threads")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"HANCOCK Slide Downloader")
    print(f"Patients {args.start}–{args.end}  ({args.end - args.start + 1} patients)")
    print(f"Image types: {', '.join(t['name'] for t in IMAGE_TYPES)}")
    print(f"Output: {out_root.resolve()}")
    print(f"Workers: {args.workers}")
    print(f"{'='*60}\n")

    tasks = build_tasks(args.start, args.end, out_root)
    print(f"Total files to fetch: {len(tasks)}\n")

    session  = requests.Session()
    manifest = []   # (patient_id, image_type, url, local_path)
    skipped  = []   # 404s

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, url, dest, session): (url, dest)
                   for url, dest in tasks}

        with tqdm(total=len(tasks), unit="file") as bar:
            for future in as_completed(futures):
                url, dest, ok, status = future.result()

                # Derive patient id and image type from the path
                parts     = dest.parts
                pid_str   = dest.parent.name          # e.g. "patient_001"
                img_label = dest.stem                  # e.g. "PrimaryTumor_HE_001"

                if ok:
                    manifest.append((pid_str, img_label, url, str(dest)))
                elif status == 404:
                    skipped.append((pid_str, url, "404"))
                else:
                    skipped.append((pid_str, url, str(status)))
                    tqdm.write(f"  [ERROR] {url} → {status}")

                bar.update(1)

    # ── Write manifest CSV ─────────────────────────────────────────────────
    manifest_path = out_root / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "image_label", "url", "local_path"])
        w.writerows(manifest)

    # ── Write skipped CSV ──────────────────────────────────────────────────
    if skipped:
        skipped_path = out_root / "skipped.csv"
        with open(skipped_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient", "url", "reason"])
            w.writerows(skipped)
        print(f"\nSkipped / missing: {len(skipped)}  (see {skipped_path})")

    print(f"\n{'='*60}")
    print(f"Done. {len(manifest)} images downloaded.")
    print(f"Manifest: {manifest_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()