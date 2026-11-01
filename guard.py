"""
guard.py - Leaf Guard: reject inputs the model shouldn't be trusted on.

The classifier only knows 38 classes. For anything else (a bottle-gourd leaf,
a shoe, a blank image, a blurry photo) it still forces a guess and can look
confident while being wrong. This guard inspects BOTH the model's output and
the image itself, and refuses to answer when something looks off.

Signals used:
  1. top-1 confidence   (low  -> the model is unsure)
  2. prediction entropy (high -> probability spread across many classes)
  3. image flatness     (very low pixel variation -> blank / featureless image)

No extra model or training is needed. For a stronger guarantee, train a tiny
binary "is-this-a-leaf?" model and call it before the classifier (future work).
"""
import numpy as np


def entropy(prob):
    """Shannon entropy (nats). 0 = certain; ln(38) ~ 3.64 = uniform over 38 classes."""
    prob = np.asarray(prob, dtype="float64")
    return float(-(prob * np.log(prob + 1e-9)).sum())


def check(prob, conf_threshold=0.60, entropy_threshold=2.2, img=None, min_std=12.0):
    """
    Return (ok, reason).
      ok=True  -> trust the prediction.
      ok=False -> reject (probably not a supported crop, or an unusable image).

    Tune to taste:
      conf_threshold      raise -> stricter (fewer wrong answers, more rejections)
      entropy_threshold   lower -> stricter
      min_std             raise -> stricter on flat/blank images
    """
    top = float(np.max(prob))
    ent = entropy(prob)

    if top < conf_threshold:
        return False, f"top confidence only {top * 100:.0f}%"
    if ent > entropy_threshold:
        return False, "predictions are spread out (no clear winner)"

    if img is not None:
        arr = np.asarray(img.convert("L"), dtype="float32")
        if arr.std() < min_std:
            return False, "the image looks blank or too flat"

    return True, ""
