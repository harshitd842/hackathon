"""
evaluate_plantdoc.py — measure the CURRENT model on PlantDoc (real field photos).

PlantDoc ships an official train/ and test/ split. This script uses ONLY the
test/ folder, so nothing it measures has ever been seen in training.

It reports:
  * overall accuracy on the PlantDoc test split
  * accuracy per class (with how many photos each class had)
  * how often the Leaf Guard would refuse (confidence < 0.65)
  * the same numbers restricted to the confident predictions only

Run:
    python evaluate_plantdoc.py --zip plantdoc.zip
    python evaluate_plantdoc.py --zip plantdoc.zip --guard      (apply the 0.65 gate)
"""
import argparse
import io
import json
import zipfile
from collections import defaultdict

import numpy as np
import onnxruntime as ort
from PIL import Image

NAMES = json.load(open("class_names.json"))
MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD = np.array([0.229, 0.224, 0.225], dtype="float32")
SIZE = 96
IMG_EXT = (".jpg", ".jpeg", ".png")

# PlantDoc folder name -> our class_names.json entry
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


def preprocess(img):
    img = img.convert("RGB")
    w, h = img.size
    s = 112 / min(w, h)
    img = img.resize((round(w * s), round(h * s)))
    w, h = img.size
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    img = img.crop((left, top, left + SIZE, top + SIZE))
    x = np.asarray(img, dtype="float32") / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype("float32")


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="PlantDoc-Dataset-master.zip")
    ap.add_argument("--model", default="crop_disease.onnx")
    ap.add_argument("--guard", type=float, default=0.0,
                    help="if >0, treat predictions below this confidence as refusals")
    ap.add_argument("--limit", type=int, default=0, help="only first N images per class")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    idx_of = {n: i for i, n in enumerate(NAMES)}

    total = correct = refused = correct_conf = confident = 0
    per_class = defaultdict(lambda: [0, 0])
    unknown_folders = set()

    with zipfile.ZipFile(args.zip) as z:
        members = [m for m in z.namelist() if m.lower().endswith(IMG_EXT)]
        # keep only the official test split
        test = [m for m in members if "/test/" in m.replace("\\", "/")]
        print(f"test-split images: {len(test)}")

        seen_per_class = defaultdict(int)
        for m in test:
            parts = m.replace("\\", "/").split("/")
            folder = parts[-2]
            if folder not in MAP:
                unknown_folders.add(folder)
                continue
            true_name = MAP[folder]
            if true_name not in idx_of:
                unknown_folders.add(f"{folder} -> {true_name} (not in class_names.json)")
                continue
            if args.limit and seen_per_class[folder] >= args.limit:
                continue
            seen_per_class[folder] += 1

            with z.open(m) as fh:
                try:
                    img = Image.open(io.BytesIO(fh.read()))
                except Exception:
                    continue
            prob = softmax(sess.run(["logits"], {"input": preprocess(img)})[0][0])
            pred_i = int(prob.argmax())
            conf = float(prob[pred_i])
            ok = NAMES[pred_i] == true_name

            total += 1
            correct += int(ok)
            per_class[true_name][0] += int(ok)
            per_class[true_name][1] += 1
            if args.guard > 0 and conf < args.guard:
                refused += 1
            else:
                confident += 1
                correct_conf += int(ok)

    if unknown_folders:
        print("unmapped folders:", sorted(unknown_folders))

    if not total:
        raise SystemExit("no test images matched the mapping")

    print()
    print("=" * 62)
    print(f"PlantDoc test accuracy : {100.0*correct/total:5.1f}%   ({correct}/{total})")
    if args.guard > 0:
        print(f"Guard {args.guard:.2f} -> refused {refused} ({100.0*refused/total:.1f}%)")
        if confident:
            print(f"Accuracy on answered  : {100.0*correct_conf/confident:5.1f}%   ({correct_conf}/{confident})")
    print("=" * 62)
    print("\nper class:")
    for name in sorted(per_class, key=lambda n: per_class[n][0] / max(per_class[n][1], 1)):
        ok, n = per_class[name]
        print(f"  {100.0*ok/max(n,1):5.1f}%  {ok:3d}/{n:<3d}  {name}")


if __name__ == "__main__":
    main()
