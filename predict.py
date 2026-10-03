"""
predict.py — run the on-device crop-disease classifier on a single image.

Dependencies: onnxruntime, numpy, pillow
Usage:  python predict.py path/to/leaf.jpg
"""
import sys, json
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL = "crop_disease.onnx"
NAMES = json.load(open("class_names.json"))

# ImageNet statistics used during training
MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD  = np.array([0.229, 0.224, 0.225], dtype="float32")
SIZE = 96

_sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])


def preprocess(path):
    """Resize shorter side to 112, center-crop 96, normalise. No torch needed."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = 112 / min(w, h)
    img = img.resize((round(w * s), round(h * s)))
    w, h = img.size
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    img = img.crop((left, top, left + SIZE, top + SIZE))
    x = np.asarray(img, dtype="float32") / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype("float32")   # (1, 3, 96, 96)


def predict(path, top_k=3):
    logits = _sess.run(["logits"], {"input": preprocess(path)})[0][0]
    e = np.exp(logits - logits.max())
    prob = e / e.sum()
    order = prob.argsort()[::-1][:top_k]
    return [(NAMES[i], float(prob[i])) for i in order]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python predict.py <image.jpg>")
        sys.exit(1)
    print(f"Image: {sys.argv[1]}")
    for name, conf in predict(sys.argv[1]):
        pretty = name.replace("___", " - ").replace("_", " ")
        print(f"  {pretty:50s} {conf*100:5.1f}%")
