"""
Unit tests for the Leaf Guard (guard.py).

Run from the repo root:
    python -m pytest tests/ -v
or without pytest:
    python tests/test_guard.py
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guard import check, entropy  # noqa: E402

N = 38  # the app knows 38 disease classes


def _prob(top, n=N):
    """A probability vector whose best class has confidence `top`."""
    p = np.full(n, (1.0 - top) / (n - 1))
    p[0] = top
    return p


def test_high_confidence_is_accepted():
    ok, reason = check(_prob(0.92), conf_threshold=0.65)
    assert ok is True, reason
    assert reason == ""


def test_low_confidence_is_refused():
    ok, reason = check(_prob(0.51), conf_threshold=0.65)
    assert ok is False
    assert "51%" in reason


def test_threshold_boundary_is_accepted():
    # 0.65 is not < 0.65, so exactly at the threshold the answer is trusted
    ok, _ = check(_prob(0.65), conf_threshold=0.65)
    assert ok is True


def test_spread_out_prediction_is_refused():
    # every class equally likely -> very high entropy
    uniform = np.full(N, 1.0 / N)
    ok, reason = check(uniform, conf_threshold=0.01)
    assert ok is False
    assert "spread out" in reason


def test_blank_image_is_refused():
    blank = Image.new("L", (64, 64), color=128)
    ok, reason = check(_prob(0.95), conf_threshold=0.65, img=blank)
    assert ok is False
    assert "blank" in reason or "flat" in reason


def test_textured_image_is_accepted():
    rng = np.random.default_rng(0)
    noisy = Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8), mode="L")
    ok, reason = check(_prob(0.95), conf_threshold=0.65, img=noisy)
    assert ok is True, reason


def test_entropy_of_certain_prediction_is_zero():
    certain = np.zeros(N)
    certain[3] = 1.0
    assert entropy(certain) < 1e-6


def test_entropy_of_uniform_prediction_is_log_n():
    uniform = np.full(N, 1.0 / N)
    assert abs(entropy(uniform) - np.log(N)) < 1e-6


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
    print(f"\n{8 - failures}/8 passed")
    sys.exit(1 if failures else 0)
