"""
app.py — simple web demo (Gradio) for the on-device crop-disease classifier.

Dependencies: gradio, onnxruntime, numpy, pillow
Run:   pip install gradio
       python app.py
Then open the local URL it prints.
"""
import json
import numpy as np
import onnxruntime as ort
import gradio as gr
from PIL import Image

NAMES = json.load(open("class_names.json"))
MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD  = np.array([0.229, 0.224, 0.225], dtype="float32")
SIZE = 96
_sess = ort.InferenceSession("crop_disease.onnx", providers=["CPUExecutionProvider"])


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


def classify(img):
    logits = _sess.run(["logits"], {"input": preprocess(img)})[0][0]
    e = np.exp(logits - logits.max())
    prob = e / e.sum()
    return {NAMES[i].replace("___", " - ").replace("_", " "): float(prob[i])
            for i in prob.argsort()[::-1][:3]}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Crop Disease Detector (on-device)",
    description="Upload a photo of a plant leaf. The model runs offline and returns the top 3 likely "
                "diseases with confidence. 38 classes, 90.1% accuracy on held-out test leaves.",
)

if __name__ == "__main__":
    demo.launch()
