"""
leaf_segment.py — find leaf region(s) in a photo, WITHOUT a trained detector.

Why this exists:
    The classifier was trained on single, close-up leaves (PlantVillage style).
    On a real photo — a whole plant, or a pile of many leaves — feeding the
    entire image gives poor results. This module segments the image by colour
    into leaf-like regions, so app.py can:
      * auto-crop a single leaf (a big accuracy win on real photos), and
      * handle photos that contain MANY leaves (classify each region separately).

Pure numpy + Pillow. No extra dependencies.
"""
import numpy as np

_GRID = 160             # segment on a small grid for speed
_MIN_AREA_FRAC = 0.012  # ignore regions smaller than ~1.2% of the image
_MAX_LEAVES = 12        # never report more than this many regions


def _bg_color(arr):
    """Estimate the background colour from the four corners."""
    h, w, _ = arr.shape
    c = max(4, min(h, w) // 8)
    corners = np.concatenate([
        arr[:c, :c].reshape(-1, 3), arr[:c, -c:].reshape(-1, 3),
        arr[-c:, :c].reshape(-1, 3), arr[-c:, -c:].reshape(-1, 3),
    ])
    return np.median(corners, axis=0)


def _mask(arr, tol=55.0):
    """Leaf pixels = far from the background colour, and not sky-blue."""
    bg = _bg_color(arr)
    dist = np.sqrt(((arr - bg) ** 2).sum(axis=2))
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    sky = (b > r + 20) & (b > g + 10)          # bluish -> likely sky
    return (dist > tol) & (~sky)


def _components(mask):
    """4-connected components via iterative flood fill. Returns [(x0,y0,x1,y1,area)]."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            x0 = x1 = sx
            y0 = y1 = sy
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                if cx < x0:
                    x0 = cx
                elif cx > x1:
                    x1 = cx
                if cy < y0:
                    y0 = cy
                elif cy > y1:
                    y1 = cy
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((x0, y0, x1, y1, area))
    return comps


def find_leaves(img, max_leaves=_MAX_LEAVES):
    """Return bounding boxes [(x0,y0,x1,y1), ...] of leaf-like regions, biggest first."""
    W, H = img.size
    small = img.convert("RGB").resize((_GRID, _GRID))
    arr = np.asarray(small, dtype="float32")
    comps = _components(_mask(arr))
    total = _GRID * _GRID
    comps = [c for c in comps if c[4] >= _MIN_AREA_FRAC * total]
    comps.sort(key=lambda c: -c[4])
    sx, sy = W / _GRID, H / _GRID
    boxes = []
    for (x0, y0, x1, y1, _a) in comps[:max_leaves]:
        boxes.append((int(x0 * sx), int(y0 * sy), int((x1 + 1) * sx), int((y1 + 1) * sy)))
    return boxes


def autocrop(img, pad=0.06):
    """Crop to the largest leaf region (with a little padding). Returns (cropped_img, box)."""
    boxes = find_leaves(img, max_leaves=1)
    if not boxes:
        return img, None
    x0, y0, x1, y1 = boxes[0]
    W, H = img.size
    pw, ph = int((x1 - x0) * pad), int((y1 - y0) * pad)
    x0, y0 = max(0, x0 - pw), max(0, y0 - ph)
    x1, y1 = min(W, x1 + pw), min(H, y1 + ph)
    if x1 - x0 < 24 or y1 - y0 < 24:      # too small to be a real crop
        return img, None
    return img.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)
