import argparse, json, os
import torch, torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ap = argparse.ArgumentParser()
ap.add_argument("--head", default="head.pt")
ap.add_argument("--out", default="crop_disease.onnx")
ap.add_argument("--img", type=int, default=96)
args = ap.parse_args()

ck = torch.load(args.head, map_location="cpu", weights_only=False)
mu, sd, names = ck["mu"], ck["sd"], ck["names"]
C = len(names)
print("loaded", args.head, ":", C, "classes")

backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
backbone.classifier = nn.Identity()
backbone.eval()

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
full.head.load_state_dict(ck["state"])
mu_t = torch.tensor(mu, dtype=torch.float32)
sd_t = torch.tensor(sd, dtype=torch.float32)
with torch.no_grad():
    W1, b1 = full.head[0].weight.clone(), full.head[0].bias.clone()
    full.head[0].weight.copy_(W1 / sd_t)
    full.head[0].bias.copy_(b1 - (W1 / sd_t) @ mu_t.squeeze(0))
full.eval()

torch.onnx.export(full, torch.randn(1, 3, args.img, args.img), args.out,
    input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False)
print("exported", args.out, round(os.path.getsize(args.out)/1e6, 2), "MB")
if not os.path.exists("class_names.json"):
    json.dump(names, open("class_names.json", "w"))
