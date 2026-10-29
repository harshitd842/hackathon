"""
prepare_agrifoundation.py — turn Agri-Foundation-145k into the raw/ layout this
project trains on.

Agri-Foundation-145k ships as an ImageFolder tree plus a `metadata.csv` with
these columns:

    filename, label, source, crop, disease, release_path

This script reads that CSV and copies only the crops you ask for into

    raw/<Crop>___<Condition>/<file>.jpg

which is exactly the layout `prepare_data.py` and `train_pipeline_v2.py` expect.

Usage
-----
    # 1. see what crops the dataset actually contains
    python prepare_agrifoundation.py --meta metadata.csv --list

    # 2. copy only the crops you want (names as printed by --list)
    python prepare_agrifoundation.py --meta metadata.csv --crops rice,wheat,corn,tomato --out raw

    # 3. cap each class so no single class swamps the rest
    python prepare_agrifoundation.py --meta metadata.csv --crops rice,wheat --per-class 400

Then:
    python check_dataset.py --raw raw
    python prepare_data.py --raw raw --out data
    python train_pipeline_v2.py --data data --out .
"""
import argparse
import csv
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def clean(text):
    """'Early Blight' -> 'Early_blight'; keeps the project's naming convention."""
    t = (text or "").strip().replace(" ", "_").replace("-", "_")
    while "__" in t:
        t = t.replace("__", "_")
    return t.strip("_") or "unknown"


def load_rows(meta_path):
    with open(meta_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"crop", "disease", "release_path"} - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"error: metadata.csv is missing columns: {sorted(missing)}")
        return list(reader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="metadata.csv")
    ap.add_argument("--root", default=".", help="dataset root that release_path is relative to")
    ap.add_argument("--out", default="raw")
    ap.add_argument("--crops", default="", help="comma list, e.g. rice,wheat,tomato")
    ap.add_argument("--per-class", type=int, default=0, help="0 = no cap")
    ap.add_argument("--list", action="store_true", help="print crops and exit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.meta):
        sys.exit(f"error: {args.meta} not found. Run this from the folder that has metadata.csv")

    rows = load_rows(args.meta)
    print(f"metadata rows: {len(rows)}")

    crops = Counter(r["crop"] for r in rows)
    if args.list or not args.crops:
        print(f"\ncrops in this dataset ({len(crops)}):\n")
        for c, n in sorted(crops.items(), key=lambda kv: -kv[1]):
            print(f"  {c:<28} {n:>7} images")
        if not args.crops:
            print("\nNow pick crops and re-run with --crops a,b,c")
        return

    wanted = [c.strip() for c in args.crops.split(",") if c.strip()]
    lookup = {c.lower(): c for c in crops}
    resolved = []
    for w in wanted:
        hit = lookup.get(w.lower())
        if not hit:
            print(f"  ! '{w}' not in this dataset — skipping")
            continue
        resolved.append(hit)
    if not resolved:
        sys.exit("error: none of the crops you asked for exist here. Run --list.")

    random.seed(args.seed)
    by_class = defaultdict(list)
    for r in rows:
        if r["crop"] in resolved:
            by_class[f"{r['crop']}___{clean(r['disease'])}"].append(r)

    print(f"\nclasses to build: {len(by_class)}")
    total = 0
    for cls in sorted(by_class):
        items = by_class[cls]
        if args.per_class and len(items) > args.per_class:
            items = random.sample(items, args.per_class)
        dst_dir = os.path.join(args.out, cls)
        os.makedirs(dst_dir, exist_ok=True)
        copied = 0
        for r in items:
            src = os.path.join(args.root, r["release_path"])
            if not src.lower().endswith(IMG_EXT) or not os.path.exists(src):
                continue
            name = os.path.basename(src)
            dst = os.path.join(dst_dir, name)
            if os.path.exists(dst):
                continue
            try:
                shutil.copy2(src, dst)
                copied += 1
            except OSError:
                continue
        total += copied
        print(f"  {cls:<40} {copied:>6}")

    print(f"\ncopied {total} images into {args.out}/ across {len(by_class)} classes")
    print("\nnext:")
    print(f"  python check_dataset.py --raw {args.out}")
    print(f"  python prepare_data.py --raw {args.out} --out data")
    print("  python train_pipeline_v2.py --data data --out .")


if __name__ == "__main__":
    main()
