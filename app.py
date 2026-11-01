"""
app.py - Crop Disease Detector with Leaf Guard, auto-crop, multi-leaf support,
Hindi/English output and an optional spoken answer.

100% offline. No internet, no cloud, no account.

Files needed in the same folder:
    crop_disease.onnx      the classifier
    class_names.json       the 38 class labels
    guard.py               Leaf Guard (input validation)
    leaf_segment.py        leaf-region detection (auto-crop + multi-leaf)
    farm_tools.py          local advisory logic (English)
    i18n.py                Hindi advisory + UI strings
    farm_storehouse.db     local inventory (create with: python seed_inventory.py)

Run:
    python seed_inventory.py     # once
    python app.py
Then open the local URL it prints (usually http://127.0.0.1:7860).

Speech: the 🔊 button uses the browser's built-in speech synthesis, so it needs
no package and no internet. It uses a Hindi voice when the operating system has
one; if it does not, the button simply does nothing - the text is always there.
"""
import json
import re
from collections import Counter

import numpy as np
import onnxruntime as ort
import gradio as gr

import guard
import farm_tools
import i18n
import leaf_segment

NAMES = json.load(open("class_names.json"))
MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD = np.array([0.229, 0.224, 0.225], dtype="float32")
SIZE = 96
_sess = ort.InferenceSession("crop_disease.onnx", providers=["CPUExecutionProvider"])

CONF_THRESHOLD = 0.65   # below this -> Leaf Guard rejects the image
MULTI_LEAF_MIN = 3      # 3+ separate regions -> treat as a multi-leaf photo

HI = "हिंदी"
EN = "English"

SPEAK_JS = """
(text) => {
  if (!text || !window.speechSynthesis) return;
  const u = new SpeechSynthesisUtterance(text);
  u.lang = 'hi-IN';
  u.rate = 0.95;
  const voices = window.speechSynthesis.getVoices() || [];
  const hi = voices.find(v => (v.lang || '').toLowerCase().startsWith('hi'));
  if (hi) { u.voice = hi; }
  else { return; }               // no Hindi voice on this machine: stay silent
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
}
"""


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


def classify(img):
    return softmax(_sess.run(["logits"], {"input": preprocess(img)})[0][0])


def _sliding_vote(img, frac=0.5, n=3):
    """Classify an n x n grid of overlapping windows; return (prob, conf) ONLY if a
    majority of windows agree on the same class, else (None, 0.0).

    Helps real / compound-leaf photos (where the model wants a single-leaflet close-up),
    while refusing to invent a disease when the windows disagree -- which is exactly what
    a healthy or out-of-scope leaf looks like."""
    W, H = img.size
    w, h = int(W * frac), int(H * frac)
    preds = []
    for iy in range(n):
        for ix in range(n):
            x0 = int(ix * (W - w) / (n - 1))
            y0 = int(iy * (H - h) / (n - 1))
            preds.append(classify(img.crop((x0, y0, x0 + w, y0 + h))))
    counts = Counter(int(p.argmax()) for p in preds)
    top_idx, votes = counts.most_common(1)[0]
    if votes * 2 <= len(preds):          # no majority -> unreliable, reject
        return None, 0.0
    agree = np.mean([p for p in preds if int(p.argmax()) == top_idx], axis=0)
    return agree, float(agree[top_idx])


def _pretty(name, lang):
    return i18n.pretty_hi(name) if lang == HI else farm_tools.pretty(name)


def _leaf_line(i, prob, lang):
    """One line for the multi-leaf report. Returns (text, is_diseased)."""
    idx = int(prob.argmax())
    name = NAMES[idx]
    conf = float(prob[idx]) * 100
    if "healthy" in name.lower():
        word = "स्वस्थ" if lang == HI else "healthy"
        return f"- **{'पत्ता' if lang == HI else 'Leaf'} {i}:** {_pretty(name, lang)} ({conf:.0f}%) - {word} ✅", False
    treatment = i18n.treatment_hi(name)[0] if lang == HI else farm_tools.get_treatment(name)[0]
    return f"- **{'पत्ता' if lang == HI else 'Leaf'} {i}:** {_pretty(name, lang)} ({conf:.0f}%) - {treatment}", True


def speech_text(markdown):
    """Strip markdown and emoji so the browser reads clean sentences."""
    if not markdown:
        return ""
    t = re.sub(r"[*_`>#]+", "", markdown)
    t = re.sub(r"[⚠️✅🌿🔊]", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def _run(img, acreage, lang=HI):
    if img is None:
        return (i18n.UI["no_photo"] if lang == HI else "Please upload a leaf photo first."), {}, ""

    boxes = leaf_segment.find_leaves(img)

    # --- Many leaves: detect and check each region separately ---
    if len(boxes) >= MULTI_LEAF_MIN:
        if lang == HI:
            lines = [f"🌿 **{len(boxes)} पत्ती क्षेत्र मिले** - हर एक जाँचा जा रहा है:", ""]
        else:
            lines = [f"🌿 **Detected {len(boxes)} leaf regions** - checking each one:", ""]
        affected = 0
        for i, (x0, y0, x1, y1) in enumerate(boxes, 1):
            crop = img.crop((x0, y0, x1, y1))
            prob = classify(crop)
            ok, reason = guard.check(prob, CONF_THRESHOLD, img=crop)
            if not ok:
                if lang == HI:
                    lines.append(f"- **पत्ता {i}:** साफ़ नहीं / भरोसा कम ({reason})")
                else:
                    lines.append(f"- **Leaf {i}:** unclear / not confident ({reason})")
                continue
            line, is_diseased = _leaf_line(i, prob, lang)
            lines.append(line)
            affected += int(is_diseased)
        if lang == HI:
            lines += [
                "",
                f"**सार:** {len(boxes)} में से {affected} पत्ती क्षेत्र रोगग्रस्त लगते हैं।",
                "_पूरा इलाज प्लान देखने के लिए एक रोगग्रस्त पत्ती अलग से अपलोड करें।_",
            ]
        else:
            lines += [
                "",
                f"**Summary:** {affected} of {len(boxes)} leaf regions look diseased.",
                "_Tip: for a full treatment plan, upload a single affected leaf on its own._",
            ]
        return "\n".join(lines), {}, speech_text("\n".join(lines))

    # --- Single leaf: classify the whole image, the auto-cropped leaf, and (if
    #     still unsure) a grid of sliding windows; keep the most confident. ---
    prob = classify(img)
    crop, box = leaf_segment.autocrop(img)
    if box is not None:
        prob_crop = classify(crop)
        if float(prob_crop.max()) > float(prob.max()):
            prob = prob_crop
    if float(prob.max()) < CONF_THRESHOLD:
        prob_vote, conf_vote = _sliding_vote(img)
        if prob_vote is not None and conf_vote > float(prob.max()):
            prob = prob_vote

    top3 = {_pretty(NAMES[i], lang): float(prob[i]) for i in prob.argsort()[::-1][:3]}

    # Leaf Guard: trust the model only when it is actually confident
    ok, reason = guard.check(prob, CONF_THRESHOLD, img=img)
    if not ok:
        # Low confidence: still show the best guess, but clearly labelled, and with
        # NO treatment / economics -- so we never advise spraying on a shaky prediction.
        idx = int(prob.argmax())
        conf = float(prob[idx]) * 100
        if lang == HI:
            text = i18n.UI["low_conf"].format(best=_pretty(NAMES[idx], lang), conf=conf, reason=reason)
        else:
            text = (
                "⚠️ **Low confidence** - the model isn't sure about this photo.\n\n"
                f"**Best guess:** {farm_tools.pretty(NAMES[idx])}  ({conf:.0f}%)\n\n"
                "This may be wrong. For a reliable result, upload a **single leaf**, "
                "close-up, on a plain background, in good light.\n\n"
                f"_({reason})_"
            )
        return text, {}, speech_text(text)

    top_idx = int(prob.argmax())
    conf_pct = float(prob[top_idx]) * 100
    acres = float(acreage or 1.0)
    if lang == HI:
        advice = i18n.build_advice_hi(NAMES[top_idx], conf_pct, acres)
    else:
        advice = farm_tools.build_advice(NAMES[top_idx], conf_pct, acres)
    return advice, top3, speech_text(advice)


def run(img, acreage, lang):
    """Wrapper: any unexpected error is shown in the output instead of a blank box."""
    try:
        print("[run] image =", None if img is None else img.size, "| lang =", lang, flush=True)
        return _run(img, acreage, lang)
    except Exception as e:                     # noqa: BLE001
        import traceback
        traceback.print_exc()
        msg = (f"⚠️ **Internal error:** {type(e).__name__}: {e}\n\n"
               "Please copy this message to the developer.")
        return msg, {}, ""


with gr.Blocks(title=i18n.UI["title"]) as demo:
    gr.Markdown(f"# {i18n.UI['title']}")
    gr.Markdown(i18n.UI["desc"])
    gr.Markdown(i18n.UI["scope"])

    with gr.Row():
        with gr.Column(scale=5):
            photo = gr.Image(type="pil", label=i18n.UI["img_label"], height=420)
            acreage = gr.Number(value=1, label=i18n.UI["acre_label"])
            lang = gr.Radio([HI, EN], value=HI, label=i18n.UI["lang_label"])
            check_btn = gr.Button(i18n.UI["check_btn"], variant="primary")
            spoken = gr.Textbox(visible=False)

        with gr.Column(scale=5):
            advice = gr.Markdown(label=i18n.UI["out_label"])
            top3 = gr.Label(num_top_classes=3, label=i18n.UI["top3_label"])

    check_btn.click(fn=run, inputs=[photo, acreage, lang], outputs=[advice, top3, spoken])
    photo.change(fn=run, inputs=[photo, acreage, lang], outputs=[advice, top3, spoken])

if __name__ == "__main__":
    demo.launch()
