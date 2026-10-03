"""
train_pipeline.py — end-to-end training pipeline for the crop-disease model.

Approach (chosen to run on modest hardware, e.g. a laptop CPU):
  1. Use a frozen ImageNet-pretrained MobileNetV3-Small as a feature extractor.
  2. Extract 576-d features once for every training/validation image (fast: forward only).
  3. Train a small classifier head on the cached features (the "linear probe").
  4. Attach the head back to the backbone and export the whole thing to ONNX.

Data layout expected:
    data/train/<class_name>/*.jpg
    data/val/<class_name>/*.jpg

Dependencies: torch, torchvision, onnx, onnxruntime, pillow, numpy
Run:  python train_pipeline.py
Outputs: crop_disease.onnx, class_names.json, head.pt, model_metrics.json
"""
import os, glob, json, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

torch.set_num_threads(2)
torch.manual_seed(0)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG = 96
T = transforms.Compose([transforms.Resize(112), transforms.CenterCrop(IMG),
                        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
NAMES = sorted(os.listdir("data/train"))
json.dump(NAMES, open("class_names.json", "w"))
C = len(NAMES)

# ---------- 1. frozen backbone ----------
W = MobileNet_V3_Small_Weights.IMAGENET1K_V1
backbone = mobilenet_v3_small(weights=W)
backbone.classifier = nn.Identity()          # outputs 576-d features
backbone.eval()

# ---------- 2. extract features ----------
def extract(sub):
    feats, labs = [], []
    for ci, cn in enumerate(NAMES):
        batch = []
        for f in sorted(glob.glob(f"data/{sub}/{cn}/*.jpg")):
            batch.append(T(Image.open(f).convert("RGB")))
            if len(batch) == 32:
                with torch.no_grad():
                    feats.append(backbone(torch.stack(batch)).numpy())
                labs += [ci] * len(batch); batch = []
        if batch:
            with torch.no_grad():
                feats.append(backbone(torch.stack(batch)).numpy())
            labs += [ci] * len(batch)
    return np.concatenate(feats).astype("float32"), np.array(labs)

print("extracting features ...", flush=True)
Xtr, ytr = extract("train")
Xva, yva = extract("val")
mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
Xtr, ytr, Xva, yva = map(torch.tensor, (Xtr, ytr, Xva, yva))

# ---------- 3. train head ----------
head = nn.Sequential(nn.Linear(576, 256), nn.BatchNorm1d(256), nn.ReLU(),
                     nn.Dropout(0.3), nn.Linear(256, C))
opt = torch.optim.AdamW(head.parameters(), 1e-3, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
best, best_state = 0.0, None
for ep in range(80):
    head.train(); perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 256):
        idx = perm[i:i + 256]
        opt.zero_grad(); lossf(head(Xtr[idx]), ytr[idx]).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        acc = (head(Xva).argmax(1) == yva).float().mean().item()
    if acc > best:
        best, best_state = acc, {k: v.clone() for k, v in head.state_dict().items()}
head.load_state_dict(best_state); head.eval()
print(f"best validation accuracy: {best*100:.2f}%", flush=True)
torch.save({"state": head.state_dict(), "mu": mu, "sd": sd, "names": NAMES}, "head.pt")

# ---------- 4. export to ONNX ----------
class Full(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(576, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(256, C))
    def forward(self, x):
        return self.head(self.backbone(x))

full = Full(); full.head.load_state_dict(head.state_dict())
with torch.no_grad():                          # fold (f-mu)/sd into first Linear
    W1, b1 = full.head[0].weight.clone(), full.head[0].bias.clone()
    full.head[0].weight.copy_(W1 / sd)
    full.head[0].bias.copy_(b1 - (W1 / sd) @ mu.squeeze(0))
full.eval()
torch.onnx.export(full, torch.randn(1, 3, IMG, IMG), "crop_disease.onnx",
                  input_names=["input"], output_names=["logits"],
                  opset_version=17, dynamo=False)
print("exported crop_disease.onnx", flush=True)
