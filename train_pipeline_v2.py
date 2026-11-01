"""
train_pipeline_v2.py - train the crop-disease classifier from scratch.

This is an upgraded version of train_pipeline.py. It keeps the same idea - a
frozen ImageNet backbone used as a feature extractor, plus a small classifier
head trained on the cached features - and adds the four things that matter most
for real-world accuracy:

    1. Augmentation    each training photo is turned into several augmented
                       views, so the head sees more variety than you photographed
    2. Class balance   a weighted loss, so a small class is not ignored
    3. Honest metrics  per-class accuracy and the top confusions, not just one
                       overall number
    4. Reproducible    every run writes metrics.json next to the model

Data layout expected
--------------------
    data/train/<ClassName>/*.jpg
    data/val/<ClassName>/*.jpg

The class name is the folder name, so training a new class needs no code change.

Usage
-----
    python train_pipeline_v2.py --data data --out .
    python train_pipeline_v2.py --data data --out build --epochs 120 --aug 6

Outputs (in --out)
------------------
    crop_disease.onnx   the model the app loads
    class_names.json    class names, in the model's output order
    metrics.json        accuracy, per-class accuracy, top confusions
    head.pt             the trained head plus the feature normalisation

Dependencies: torch, torchvision, onnx, onnxruntime, pillow, numpy
"""
import argparse
import glob
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def eval_transform(img):
    """Deterministic: used for validation features."""
    return transforms.Compose([
        transforms.Resize(int(img * 1.16)),
        transforms.CenterCrop(img),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def aug_transform(img):
    """Random: used to grow the training set from the photos you have."""
    return transforms.Compose([
        transforms.Resize(int(img * 1.16)),
        transforms.RandomResizedCrop(img, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomRotation(25),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def files_in(folder):
    return sorted(
        f for f in glob.glob(os.path.join(folder, "*"))
        if f.lower().endswith(IMG_EXT)
    )


def extract(backbone, folder, names, tf, views, batch=32):
    """Forward-only feature extraction. `views` passes over the training set."""
    feats, labs = [], []
    for ci, cn in enumerate(names):
        files = files_in(os.path.join(folder, cn))
        if not files:
            print(f"  warning: no images for class {cn} in {folder}")
            continue
        for _ in range(views):
            buf = []
            for path in files:
                try:
                    im = Image.open(path).convert("RGB")
                except Exception:
                    continue
                buf.append(tf(im))
                if len(buf) == batch:
                    with torch.no_grad():
                        feats.append(backbone(torch.stack(buf)).numpy())
                    labs += [ci] * len(buf)
                    buf = []
            if buf:
                with torch.no_grad():
                    feats.append(backbone(torch.stack(buf)).numpy())
                labs += [ci] * len(buf)
    if not feats:
        raise SystemExit(f"no images found under {folder}")
    return np.concatenate(feats).astype("float32"), np.array(labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="folder with train/ and val/")
    ap.add_argument("--out", default=".", help="where to write the outputs")
    ap.add_argument("--img", type=int, default=96, help="input size (keep 96 for the app)")
    ap.add_argument("--aug", type=int, default=5, help="augmented views per training image")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)
    os.makedirs(args.out, exist_ok=True)

    train_dir = os.path.join(args.data, "train")
    val_dir = os.path.join(args.data, "val")
    if not os.path.isdir(train_dir):
        raise SystemExit(f"error: {train_dir} not found - run prepare_data.py first")

    names = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    if not names:
        raise SystemExit(f"error: no class folders in {train_dir}")
    C = len(names)
    missing = [n for n in names if not os.path.isdir(os.path.join(val_dir, n))]
    if missing:
        raise SystemExit(f"error: these classes have no val folder: {missing}")
    print(f"classes: {C}")

    # ---------- 1. frozen backbone ----------
    backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    backbone.classifier = nn.Identity()          # 576-d features
    backbone.eval()

    # ---------- 2. features (with augmentation on train) ----------
    t0 = time.time()
    print(f"extracting train features ({args.aug} augmented views per image) ...", flush=True)
    Xtr, ytr = extract(backbone, train_dir, names, aug_transform(args.img), args.aug)
    print("extracting val features ...", flush=True)
    Xva, yva = extract(backbone, val_dir, names, eval_transform(args.img), 1)
    print(f"  train {Xtr.shape[0]} feature rows, val {Xva.shape[0]}  ({time.time()-t0:.0f}s)")

    # ---------- 3. normalise features ----------
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr)
    Xva, yva = torch.tensor(Xva), torch.tensor(yva)

    # ---------- 4. train the head ----------
    head = nn.Sequential(
        nn.Linear(576, 256), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(256, C),
    )
    opt = torch.optim.AdamW(head.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # class weights: a small class counts as much as a big one
    counts = np.bincount(ytr.numpy(), minlength=C).astype("float32")
    counts[counts == 0] = 1.0
    weights = torch.tensor(counts.sum() / (C * counts))
    lossf = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    best, best_state, best_ep = 0.0, None, -1
    for ep in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 256):
            idx = perm[i:i + 256]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            lossf(head(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            acc = (head(Xva).argmax(1) == yva).float().mean().item()
        if acc > best:
            best, best_ep = acc, ep
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs:
            print(f"  epoch {ep:3d}  val accuracy {acc*100:5.2f}%  (best {best*100:5.2f}%)")

    head.load_state_dict(best_state)
    head.eval()
    print(f"best validation accuracy: {best*100:.2f}% at epoch {best_ep}")

    torch.save({"state": head.state_dict(), "mu": mu, "sd": sd, "names": names},
               os.path.join(args.out, "head.pt"))

    # ---------- 5. honest metrics ----------
    with torch.no_grad():
        logits = head(Xva)
        pred = logits.argmax(1).numpy()
    ytrue = yva.numpy()
    per_class = {}
    for ci, cn in enumerate(names):
        mask = ytrue == ci
        if mask.sum() == 0:
            per_class[cn] = None
            continue
        per_class[cn] = round(float((pred[mask] == ci).mean()), 4)

    # most common mistakes: true class -> what it was called instead
    confusions = {}
    for t, p in zip(ytrue, pred):
        if t != p:
            key = f"{names[t]}  ->  {names[p]}"
            confusions[key] = confusions.get(key, 0) + 1
    top_confusions = sorted(confusions.items(), key=lambda kv: -kv[1])[:10]

    metrics = {
        "classes": C,
        "train_rows": int(len(ytr)),
        "val_rows": int(len(yva)),
        "aug_views": args.aug,
        "best_val_accuracy": round(float(best), 4),
        "per_class_accuracy": per_class,
        "top_confusions": [{"pair": k, "count": v} for k, v in top_confusions],
        "images_per_class": {names[i]: int(counts[i]) for i in range(C)},
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, "class_names.json"), "w") as f:
        json.dump(names, f)

    weakest = sorted(
        ((k, v) for k, v in per_class.items() if v is not None), key=lambda kv: kv[1]
    )[:5]
    print("weakest classes:", ", ".join(f"{k} {v*100:.0f}%" for k, v in weakest))

    # ---------- 6. export ONNX ----------
    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Linear(576, 256), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(256, C),
            )

        def forward(self, x):
            return self.head(self.backbone(x))

    full = Full()
    full.head.load_state_dict(head.state_dict())
    mu_t = torch.tensor(mu, dtype=torch.float32)
    sd_t = torch.tensor(sd, dtype=torch.float32)
    with torch.no_grad():                       # fold (f - mu) / sd into the first Linear
        W1, b1 = full.head[0].weight.clone(), full.head[0].bias.clone()
        full.head[0].weight.copy_(W1 / sd_t)
        full.head[0].bias.copy_(b1 - (W1 / sd_t) @ mu_t.squeeze(0))
    full.eval()

    onnx_path = os.path.join(args.out, "crop_disease.onnx")
    torch.onnx.export(
        full, torch.randn(1, 3, args.img, args.img), onnx_path,
        input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False,
    )
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"exported {onnx_path} ({size_mb:.2f} MB)")
    print("\nnext: copy crop_disease.onnx and class_names.json into the app folder")


if __name__ == "__main__":
    main()
