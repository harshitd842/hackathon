"""
check_dataset.py - look at your raw photos and tell you where you stand.

Run this BEFORE you train. It answers the only question that matters at this
stage: do I have enough photos per class to train anything useful?

    python check_dataset.py --raw raw
    python check_dataset.py --raw raw --target 300

It reads your folders and reports:

  * how many classes you have, and how many photos are in each
  * which classes are trainable today and which need more photos
  * how many more photos each class needs to reach the target
  * whether any class is missing a healthy counterpart
  * photos that cannot be opened (corrupt or wrong format)
  * the image sizes, so you can spot tiny thumbnails

It never moves, edits or deletes anything. It only reads.

What the numbers mean
---------------------
  < 30 photos     not trainable - collect more
  30-99           trainable, but the class will be weak
  100-299         workable, expect mistakes
  300+            a solid class

The class name is the folder name, so name folders Crop___Condition, for example
Tomato___Late_blight or Wheat___healthy.
"""
import argparse
import os
import sys
from collections import Counter

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

TRAINABLE_MIN = 30


def images_in(folder):
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXT)
    )


def band(n):
    if n < TRAINABLE_MIN:
        return "collect more"
    if n < 100:
        return "weak"
    if n < 300:
        return "workable"
    return "solid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw", help="folder holding one sub-folder per class")
    ap.add_argument("--target", type=int, default=300, help="photos you want per class")
    args = ap.parse_args()

    if not os.path.isdir(args.raw):
        sys.exit(f"error: no such folder: {args.raw}\nCreate it and put one sub-folder per class inside.")

    classes = sorted(
        d for d in os.listdir(args.raw)
        if os.path.isdir(os.path.join(args.raw, d))
    )
    if not classes:
        sys.exit(
            f"error: {args.raw}/ has no class folders.\n"
            f"Make one folder per class, e.g. {args.raw}/Tomato___Late_blight/ and put the photos in it."
        )

    counts = {}
    broken = []
    sizes = Counter()
    total = 0

    for cls in classes:
        path = os.path.join(args.raw, cls)
        files = images_in(path)
        counts[cls] = len(files)
        total += len(files)
        for f in files:
            full = os.path.join(path, f)
            try:
                from PIL import Image
                with Image.open(full) as im:
                    sizes[im.size] += 1
            except Exception:
                broken.append(full)

    print(f"\nFolder : {args.raw}/")
    print(f"Classes: {len(classes)}")
    print(f"Photos : {total}")
    print(f"Target : {args.target} per class\n")

    print(f"{'class':40s} {'photos':>7s}  {'status':12s} {'need':>6s}")
    print("-" * 70)
    for cls in sorted(counts, key=lambda c: counts[c]):
        n = counts[cls]
        need = max(0, args.target - n)
        print(f"{cls:40s} {n:7d}  {band(n):12s} {need if need else '-':>6}")

    trainable = [c for c in counts if counts[c] >= TRAINABLE_MIN]
    print("-" * 70)
    print(f"trainable today : {len(trainable)} of {len(classes)} classes")

    if trainable:
        smallest = min(counts[c] for c in trainable)
        largest = max(counts[c] for c in trainable)
        print(f"class sizes     : smallest {smallest}, largest {largest}")
        if largest > 3 * max(1, smallest):
            print("  note: your classes are very unbalanced. A big class can drown a small one.")
    else:
        print("  No class has reached 30 photos yet, so nothing can be trained.")
        print("  Keep collecting. See the photo guide for how to shoot them.")

    # a healthy class for every crop is what stops the model calling everything diseased
    crops = {}
    for cls in classes:
        crop = cls.split("___")[0] if "___" in cls else cls
        crops.setdefault(crop, []).append(cls)
    missing_healthy = [c for c, v in crops.items() if not any("healthy" in x.lower() for x in v)]
    if missing_healthy:
        print(f"\nno healthy class for: {', '.join(sorted(missing_healthy))}")
        print("  Add a <Crop>___healthy folder, or the model may call every leaf diseased.")

    if broken:
        print(f"\n{len(broken)} photo(s) could not be opened:")
        for b in broken[:10]:
            print(f"  {b}")
        if len(broken) > 10:
            print(f"  ... and {len(broken) - 10} more")

    if sizes:
        print("\nimage sizes seen:")
        for size, n in sizes.most_common(5):
            print(f"  {size[0]}x{size[1]}  ({n} photos)")
        if min(s[0] for s in sizes) < 100:
            print("  note: some photos are very small. Small images train poorly.")

    print("\nnext steps")
    print("  1. put every photo in the folder of its class, named Crop___Condition")
    print("  2. python prepare_data.py --raw", args.raw, "--out data")
    print("  3. python train_pipeline_v2.py --data data --out .")
    print("\nsee TRAIN_GUIDE.md for the full walkthrough.")


if __name__ == "__main__":
    main()
