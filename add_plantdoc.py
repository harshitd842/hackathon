"""
add_plantdoc.py - mix PlantDoc's field photos into a PlantVillage training set.

PlantDoc ships an official train/ and test/ split. This script:
  * copies PlantDoc **train** images into an existing PlantVillage train folder
    (so the model sees real field photos during training)
  * copies PlantDoc **test** images into a separate folder that is NEVER trained on
    (so the "after" number is measured on photos the model has not seen)

Folder names are mapped to the 38 PlantVillage class names. The 10 classes that
PlantDoc does not cover are left untouched.

Usage:
    python add_plantdoc.py --zip plantdoc.zip --train data/train --val pd_val
    python add_plantdoc.py --zip plantdoc.zip --train data/train --val pd_val --dry-run
"""
import argparse
import os
import shutil
import sys
import zipfile

IMG_EXT = (".jpg", ".jpeg", ".png")

MAP = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--train", default="data/train", help="PlantVillage train folder to add into")
    ap.add_argument("--val", default="pd_val", help="where to put PlantDoc's test images")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.train):
        sys.exit(f"error: {args.train} does not exist. Run prepare_data.py first.")

    added = {"train": 0, "val": 0}
    per_class = {}
    skipped = set()

    with zipfile.ZipFile(args.zip) as z:
        for m in z.namelist():
            if not m.lower().endswith(IMG_EXT):
                continue
            parts = m.replace("\\", "/").split("/")
            if len(parts) < 3:
                continue

            # the zip nests some images (e.g. <Class>/IMG_1629.JPG/1507122477.jpg),
            # so find the split and the class by NAME, never by position
            low = [p.lower() for p in parts]
            if "train" in low:
                split = "train"
            elif "test" in low:
                split = "test"
            else:
                continue
            folder = next((p for p in parts if p in MAP), None)
            if folder is None:
                if len(parts) >= 3:
                    skipped.add(parts[-2])
                continue

            target_cls = MAP[folder]
            dest_root = args.train if split == "train" else args.val
            dest_dir = os.path.join(dest_root, target_cls)
            key = f"{split}/{target_cls}"
            per_class[key] = per_class.get(key, 0) + 1
            added["train" if split == "train" else "val"] += 1
            if args.dry_run:
                continue

            os.makedirs(dest_dir, exist_ok=True)
            base = os.path.basename(m).replace("/", "_").replace("\\", "_")
            dest = os.path.join(dest_dir, base)
            if os.path.exists(dest):
                continue                      # already copied in an earlier run
            with z.open(m) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)

    print(f"\n{'DRY RUN - nothing written' if args.dry_run else 'done'}")
    print(f"  PlantDoc train -> {args.train} : {added['train']} images")
    print(f"  PlantDoc test  -> {args.val} : {added['val']} images")
    if skipped:
        print(f"  unmapped folders skipped: {sorted(skipped)}")
    print("\nper class:")
    for k in sorted(per_class):
        print(f"  {per_class[k]:4d}  {k}")


if __name__ == "__main__":
    main()
