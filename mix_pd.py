import os, re, zipfile, shutil

ZIP = r"C:\Users\dubey\Downloads\PlantDoc-Dataset-master.zip"
TRAIN = "data/train"
VAL = "pd_val"

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

n = {"train": 0, "val": 0}
skip = set()
with zipfile.ZipFile(ZIP) as z:
    for m in z.namelist():
        if not m.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parts = m.replace("\\", "/").split("/")
        low = [p.lower() for p in parts]
        split = "train" if "train" in low else ("test" if "test" in low else None)
        if split is None:
            continue
        folder = next((p for p in parts if p in MAP), None)
        if folder is None:
            skip.add(parts[-2])
            continue
        cls = MAP[folder]
        dest_dir = os.path.join(TRAIN if split == "train" else VAL, cls)
        os.makedirs(dest_dir, exist_ok=True)
        base = re.sub(r"[^A-Za-z0-9._-]", "_", parts[-1])
        dest = os.path.join(dest_dir, base)
        if os.path.exists(dest):
            continue
        with z.open(m) as s, open(dest, "wb") as o:
            shutil.copyfileobj(s, o)
        n["train" if split == "train" else "val"] += 1

print("done")
print("  train ->", TRAIN, ":", n["train"])
print("  test  ->", VAL, ":", n["val"])
if skip:
    print("  skipped:", sorted(skip)[:5])
