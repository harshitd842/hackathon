"""
prepare_data.py — turn one folder of raw photos into the train/val layout the
training script expects.

    raw/<ClassName>/*.jpg          <- everything you photographed, unsorted
        |
        v
    data/train/<ClassName>/*.jpg   (~80%)
    data/val/<ClassName>/*.jpg     (~20%)

The class name is just the folder name, so name your folders exactly the way
the model should report them, using the Crop___Condition convention:

    Tomato___Late_blight
    Wheat___Yellow_rust
    Wheat___healthy

Usage
-----
    python prepare_data.py --raw raw --out data
    python prepare_data.py --raw raw --out data --val-frac 0.2 --move

Flags
-----
    --raw        folder holding one sub-folder per class  (default: raw)
    --out        output folder, gets train/ and val/      (default: data)
    --val-frac   share held out for validation            (default: 0.2)
    --min-images refuse to accept a class below this many (default: 30)
    --move       move files instead of copying (saves disk; raw is emptied)
    --seed       random seed for the split                (default: 0)

It copies by default, so your originals are never touched. Re-running is safe:
a class folder that already has files in the output is skipped.
"""
import argparse
import os
import random
import shutil
import sys

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def list_images(folder):
    out = []
    for name in os.listdir(folder):
        if name.lower().endswith(IMG_EXT):
            out.append(os.path.join(folder, name))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-images", type=int, default=30)
    ap.add_argument("--move", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.isdir(args.raw):
        sys.exit(f"error: raw folder not found: {args.raw}")

    classes = sorted(
        d for d in os.listdir(args.raw)
        if os.path.isdir(os.path.join(args.raw, d))
    )
    if not classes:
        sys.exit(f"error: no class folders inside {args.raw}")

    rng = random.Random(args.seed)
    train_root = os.path.join(args.out, "train")
    val_root = os.path.join(args.out, "val")
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)

    print(f"{'class':38s} {'total':>7s} {'train':>7s} {'val':>6s}")
    print("-" * 62)
    grand = 0
    problems = []

    for cls in classes:
        src = os.path.join(args.raw, cls)
        files = list_images(src)

        # already prepared? skip, so re-running does not duplicate work
        if os.path.isdir(os.path.join(train_root, cls)) and list_images(os.path.join(train_root, cls)):
            print(f"{cls:38s} {'-- skipped (already in output)':>38s}")
            continue

        if len(files) < args.min_images:
            problems.append(f"  {cls}: only {len(files)} images (minimum {args.min_images})")
            print(f"{cls:38s} {len(files):7d}  -- too few, skipped")
            continue

        rng.shuffle(files)
        n_val = max(1, int(round(len(files) * args.val_frac)))
        val_files = files[:n_val]
        train_files = files[n_val:]

        for split, group in (("train", train_files), ("val", val_files)):
            dst = os.path.join(args.out, split, cls)
            os.makedirs(dst, exist_ok=True)
            for path in group:
                target = os.path.join(dst, os.path.basename(path))
                if args.move:
                    shutil.move(path, target)
                else:
                    shutil.copy2(path, target)

        grand += len(files)
        print(f"{cls:38s} {len(files):7d} {len(train_files):7d} {len(val_files):6d}")

    print("-" * 62)
    print(f"total images placed: {grand}")
    print(f"classes: {len(os.listdir(train_root))}")

    if problems:
        print("\nWarnings — these classes were not added:")
        for p in problems:
            print(p)
        print("Collect more photos and run again.")

    print("\nNext: python train_pipeline_v2.py --data", args.out)


if __name__ == "__main__":
    main()
