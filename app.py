"""
app.py — Crop Disease Detector with Leaf Guard, auto-crop and multi-leaf support.

100% offline. No internet, no cloud, no account.

Files needed in the same folder:
    crop_disease.onnx      the classifier
    class_names.json       the 38 class labels
    guard.py               Leaf Guard (input validation)
    leaf_segment.py        leaf-region detection (auto-crop + multi-leaf)
    farm_tools.py          local advisory logic
    farm_storehouse.db     local inventory (create with: python seed_inventory.py)

Run:
    python seed_inventory.py     # once
    python app.py
Then open the local URL it prints (usually http://127.0.0.1:7860).
"""
import json

import numpy as np
import onnxruntime as ort
import gradio as gr
from PIL import Image

import guard
import farm_tools
import leaf_segment

NAMES = json.load(open("class_names.json"))
MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD = np.array([0.229, 0.224, 0.225], dtype="float32")
SIZE = 96
_sess = ort.InferenceSession("crop_disease.onnx", providers=["CPUExecutionProvider"])

CONF_THRESHOLD = 0.60   # below this -> Leaf Guard rejects the image
MULTI_LEAF_MIN = 3      # 3+ separate regions -> treat as a multi-leaf photo



   
def preprocess(img):
    img = img.convert("RGB")
    w, h = img.size

    # Match the training pipeline: Resize(112)
    scale = 112 / min(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    img = img.resize(
        (new_w, new_h),
        resample=Image.Resampling.BILINEAR
    )

    # Match the training pipeline: CenterCrop(96)
    w, h = img.size
    left = (w - SIZE) // 2
    top = (h - SIZE) // 2
    img = img.crop((left, top, left + SIZE, top + SIZE))

    x = np.asarray(img, dtype="float32") / 255.0
    x = (x - MEAN) / STD

    return x.transpose(2, 0, 1)[None].astype("float32")



def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def classify(img):
    return softmax(_sess.run(["logits"], {"input": preprocess(img)})[0][0])


def _sliding_best(img, frac=0.5, n=3):
    """Try an n x n grid of overlapping windows; return (prob, conf) of the most confident.
    Helps real / compound-leaf photos, where the model expects a single-leaflet close-up."""
    W, H = img.size
    w, h = int(W * frac), int(H * frac)
    best_prob, best_conf = None, 0.0
    for iy in range(n):
        for ix in range(n):
            x0 = int(ix * (W - w) / (n - 1))
            y0 = int(iy * (H - h) / (n - 1))
            p = classify(img.crop((x0, y0, x0 + w, y0 + h)))
            c = float(p.max())
            if c > best_conf:
                best_conf, best_prob = c, p
    return best_prob, best_conf


def _leaf_line(i, prob):
    """One line for the multi-leaf report. Returns (text, is_diseased)."""
    idx = int(prob.argmax())
    name = NAMES[idx]
    conf = float(prob[idx]) * 100
    if "healthy" in name.lower():
        return f"- **Leaf {i}:** {farm_tools.pretty(name)} ({conf:.0f}%) — healthy ✅", False
    treatment, _ = farm_tools.get_treatment(name)
    return f"- **Leaf {i}:** {farm_tools.pretty(name)} ({conf:.0f}%) — {treatment}", True


def run(img, acreage):
    if img is None:
        return "Please upload a leaf photo first.", {}

    boxes = leaf_segment.find_leaves(img)

    # --- Many leaves: detect and check each region separately ---
    if len(boxes) >= MULTI_LEAF_MIN:
        lines = [f"🌿 **Detected {len(boxes)} leaf regions** — checking each one:", ""]
        affected = 0
        for i, (x0, y0, x1, y1) in enumerate(boxes, 1):
            crop = img.crop((x0, y0, x1, y1))
            prob = classify(crop)
            ok, reason = guard.check(prob, CONF_THRESHOLD, img=crop)
            if not ok:
                lines.append(f"- **Leaf {i}:** unclear / not confident ({reason})")
                continue
            line, is_diseased = _leaf_line(i, prob)
            lines.append(line)
            affected += int(is_diseased)
        lines += [
            "",
            f"**Summary:** {affected} of {len(boxes)} leaf regions look diseased.",
            "_Tip: for a full treatment plan, upload a single affected leaf on its own._",
        ]
        return "\n".join(lines), {}

    # --- Single leaf: classify the whole image, the auto-cropped leaf, and (if
    #     still unsure) a grid of sliding windows; keep the most confident. ---
    prob = classify(img)
    crop, box = leaf_segment.autocrop(img)
    if box is not None:
        prob_crop = classify(crop)
        if float(prob_crop.max()) > float(prob.max()):
            prob = prob_crop
    if float(prob.max()) < CONF_THRESHOLD:
        prob_slide, conf_slide = _sliding_best(img)
        if prob_slide is not None and conf_slide > float(prob.max()):
            prob = prob_slide

    top3 = {farm_tools.pretty(NAMES[i]): float(prob[i]) for i in prob.argsort()[::-1][:3]}

    # Leaf Guard: trust the model only when it is actually confident
    ok, reason = guard.check(prob, CONF_THRESHOLD, img=img)
    if not ok:
        # Do NOT show the top-3 here: a low-confidence guess on an unknown image
        # is meaningless and only confuses the user.
        return (
            f"⚠️ **Not confident** ({reason}).\n\n"
            "This image may not be one of the 38 supported crops, or it may be unclear.\n\n"
            "Please upload a **single leaf**, close-up, on a plain background, in good light.",
            {},
        )

    top_idx = int(prob.argmax())
    advice = farm_tools.build_advice(NAMES[top_idx], float(prob[top_idx]) * 100, float(acreage or 1.0))
    return advice, top3


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Image(type="pil", label="Leaf photo (one leaf, or a whole plant)"),
        gr.Number(value=1, label="Farm size (acres)"),
    ],
    outputs=[
        gr.Markdown(label="Diagnosis & advisory"),
        gr.Label(num_top_classes=3, label="Top 3 predictions"),
    ],
    title="Crop Disease Detector — offline, with local advisory",
    description=(
        "Upload a leaf photo. The model runs fully offline (no internet, no cloud). "
        "If the photo has several leaves it checks each one; otherwise it auto-crops to "
        "the leaf. A Leaf Guard rejects images the model isn't sure about, and a local "
        "inventory check turns the diagnosis into an actionable recommendation. "
        "38 classes, 90.1% accuracy on held-out test leaves."
    ),
)

if __name__ == "__main__":
    demo.launch()
