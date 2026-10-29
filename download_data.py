"""
download_data.py — bring public crop-disease datasets onto your disk.

This script is a one-time downloader. Nothing here runs in the app, and the app
stays fully offline. Once a dataset is on your disk, unplug the network and
everything else — prepare_data.py, training, and the app — runs offline.

    network ON, once   ->  this script  ->  raw/<ClassName>/*.jpg
    network OFF, always->  prepare_data.py -> training -> app

Privacy, plainly
----------------
* These are public research datasets. Downloading them involves no personal data
  of yours, and no data of any farmer.
* Nothing about your own photos is ever uploaded. This script only reads.
* Kaggle downloads use YOUR OWN credentials, read from ~/.kaggle/kaggle.json or
  the KAGGLE_USERNAME / KAGGLE_KEY environment variables. This script never
  asks for, prints, or hardcodes a key.
* Use `strip-exif` after downloading to remove camera and GPS metadata from the
  images, so no location data travels into your project.

Always check the licence of the dataset you use, and credit it in your README.

Usage
-----
    python download_data.py plantdoc  --out raw
    python download_data.py kaggle <owner/dataset> --out raw
    python download_data.py strip-exif --dir raw

Dependencies: requests (only for the download commands). strip-exif needs pillow.
"""
import argparse
import io
import os
import re
import shutil
import sys
import zipfile

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

PLANTDOC_ZIP = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"


def sanitize(name):
    """'Apple Scab Leaf' -> 'Apple_Scab_Leaf'. Spaces and runs of underscores go."""
    name = name.strip().replace(" ", "_")
    name = re.sub(r"_{2,}", "_", name)
    return name.strip("_")


def unique_path(folder, filename):
    """Return a path in `folder` that does not exist yet, adding _1, _2, ... if needed."""
    target = os.path.join(folder, filename)
    if not os.path.exists(target):
        return target
    stem, ext = os.path.splitext(filename)
    n = 1
    while os.path.exists(os.path.join(folder, f"{stem}_{n}{ext}")):
        n += 1
    return os.path.join(folder, f"{stem}_{n}{ext}")


def http_get(url, timeout=120):
    try:
        import requests
    except ImportError:
        sys.exit("error: pip install requests")
    print(f"downloading {url}")
    print("(PlantDoc is around 1 GB, so this can take a few minutes)")
    r = requests.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    buf = io.BytesIO()
    total = 0
    for chunk in r.iter_content(chunk_size=1 << 20):
        buf.write(chunk)
        total += len(chunk)
        if total % (50 << 20) < (1 << 20):
            print(f"  {total / 1e6:.0f} MB ...", flush=True)
    buf.seek(0)
    return buf


def cmd_plantdoc(args):
    """PlantDoc ships as train/ and test/ folders, one sub-folder per class."""
    buf = http_get(PLANTDOC_ZIP)
    os.makedirs(args.out, exist_ok=True)
    counts = {}
    with zipfile.ZipFile(buf) as z:
        for member in z.namelist():
            if not member.lower().endswith(IMG_EXT):
                continue
            parts = member.split("/")
            # .../<split>/<ClassName>/<file>
            if len(parts) < 3:
                continue
            cls = sanitize(parts[-2])
            dst_dir = os.path.join(args.out, cls)
            os.makedirs(dst_dir, exist_ok=True)
            # train/ and test/ often reuse the same filenames, so de-duplicate
            target = unique_path(dst_dir, parts[-1])
            with z.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            counts[cls] = counts.get(cls, 0) + 1

    if not counts:
        sys.exit("error: no images found in the archive")

    print(f"\nplaced {sum(counts.values())} images in {len(counts)} classes under {args.out}/")
    for cls in sorted(counts):
        print(f"  {cls:34s} {counts[cls]:5d}")
    print("\nThese names are PlantDoc's. Rename each folder to your own convention")
    print("(Crop___Condition, e.g. Tomato___Early_blight) before training, so the")
    print("app's pretty-print and treatment map recognise them.")


def cmd_kaggle(args):
    """Fetch a Kaggle dataset using your own credentials."""
    slug = args.slug
    print(f"kaggle dataset: {slug}")
    print("using credentials from ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY")
    try:
        import kagglehub  # modern, recommended
    except ImportError:
        kagglehub = None

    os.makedirs(args.out, exist_ok=True)
    if kagglehub is not None:
        path = kagglehub.dataset_download(slug)
        print(f"downloaded to {path}")
        src = path
    else:
        print("kagglehub not found; falling back to the kaggle CLI")
        rc = os.system(f'kaggle datasets download -d {slug} -p "{args.out}" --unzip')
        if rc != 0:
            sys.exit("error: kaggle CLI failed. pip install kagglehub, or set up ~/.kaggle/kaggle.json")
        print(f"extracted into {args.out}")
        return

    # move any class folders we find into raw/
    moved = 0
    for root, dirs, files in os.walk(src):
        if not dirs and any(f.lower().endswith(IMG_EXT) for f in files):
            cls = sanitize(os.path.basename(root))
            dst_dir = os.path.join(args.out, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for f in files:
                if f.lower().endswith(IMG_EXT):
                    shutil.copy2(os.path.join(root, f), unique_path(dst_dir, f))
                    moved += 1
    print(f"placed {moved} images under {args.out}/")
    print("Check the folder names and rename them to Crop___Condition before training.")


def cmd_strip_exif(args):
    """Rewrite every image without metadata, so no camera or GPS data remains."""
    try:
        from PIL import Image
    except ImportError:
        sys.exit("error: pip install pillow")

    done = skipped = 0
    for root, _dirs, files in os.walk(args.dir):
        for f in files:
            if not f.lower().endswith(IMG_EXT):
                continue
            path = os.path.join(root, f)
            try:
                im = Image.open(path).convert("RGB")
                im.save(path, quality=92)      # re-save drops all EXIF
                done += 1
            except Exception:
                skipped += 1
    print(f"stripped metadata from {done} images ({skipped} skipped) under {args.dir}/")


def main():
    ap = argparse.ArgumentParser(description="Download public crop-disease datasets.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plantdoc", help="PlantDoc, ~2,598 field photos from IIT Gandhinagar")
    p.add_argument("--out", default="raw")
    p.set_defaults(func=cmd_plantdoc)

    k = sub.add_parser("kaggle", help="any Kaggle dataset, using your own credentials")
    k.add_argument("slug", help="owner/dataset, e.g. abdallahalidev/plantvillage-dataset")
    k.add_argument("--out", default="raw")
    k.set_defaults(func=cmd_kaggle)

    s = sub.add_parser("strip-exif", help="remove camera and GPS metadata from images")
    s.add_argument("--dir", default="raw")
    s.set_defaults(func=cmd_strip_exif)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
