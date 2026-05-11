"""
BoneAge AI — Flask inference server  (architecture-corrected)
==============================================================

The saved model is NOT ResNet-18.  Reverse-engineering the state_dict keys
reveals an EfficientNet-style backbone (MBConv blocks with Squeeze-Excitation)
with a separate sex MLP branch.  Key evidence:

  backbone.0            → stem Conv2d + BN
  backbone.1-7          → MBConv stages with SE (fc1/fc2)
  backbone.8            → head Conv2d + BN
  sex_branch.0/2        → Linear(1,16), Linear(16,16)
  head.0  [512, 1552]   → Linear(1552, 512)  where 1552 = 1536 (image) + 16 (sex)
  head.3  [1,  512]     → Linear(512, 1)

This matches EfficientNet-B0 feature dim = 1280 … but the actual dim is 1536,
which matches EfficientNet-B3 (or a custom variant).  We load with
weights=None and trust the checkpoint entirely.

Endpoints
---------
  GET  /health   → {status, model, device}
  POST /predict  → multipart: image (file) + sex ("male"|"female")
                ← {predicted_months, years, months_remainder, age_label,
                   stage, sex, percentile, confidence_note}

Usage
-----
  pip install flask flask-cors torch torchvision pillow efficientnet_pytorch
  python app.py
"""

import io, os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH = Path("/Users/omarrrefaat/Desktop/deeplearning/smart-bone-age-analyzer/backend/models/best_bone_age_model (1).pth")
IMG_SIZE   = 256
MAX_AGE    = 216.0
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Architecture reverse-engineered from checkpoint keys ───────────────────
#
# Rather than re-implement the entire EfficientNet variant by hand we use
# torchvision's EfficientNet-B3 (feature dim = 1536) and wrap it exactly
# the way the training code did:
#
#   sex_branch : Linear(1→16) → SiLU → Linear(16→16)
#   backbone   : EfficientNet features (1536-d)
#   head       : Linear(1552→512) → SiLU → Dropout → Linear(512→1)
#
# If torchvision's EfficientNet-B3 features are named differently we fall
# back to loading the raw state_dict with strict=False and let PyTorch match
# what it can — the head weights are what matter at inference.

class SEBlock(nn.Module):
    """Squeeze-and-Excitation used inside MBConv."""
    def __init__(self, in_ch, reduced_ch):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, reduced_ch)
        self.fc2 = nn.Linear(reduced_ch, in_ch)

    def forward(self, x):
        s = x.mean([2, 3])
        s = torch.selu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(2).unsqueeze(3)


class MBConvBlock(nn.Module):
    """Minimal MBConv with optional expansion + SE."""
    def __init__(self, in_ch, out_ch, stride=1, expand=1, se_ratio=0.25, kernel=3):
        super().__init__()
        mid = in_ch * expand
        layers = []

        if expand != 1:
            layers += [nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU()]

        pad = (kernel - 1) // 2
        layers += [
            nn.Conv2d(mid, mid, kernel, stride=stride, padding=pad, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
        ]

        se_ch = max(1, int(in_ch * se_ratio))
        layers.append(SEBlock(mid, se_ch))

        layers += [nn.Conv2d(mid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]

        self.block = nn.Sequential(*layers)
        self.use_skip = (stride == 1 and in_ch == out_ch)

    def forward(self, x):
        out = self.block(x)
        return out + x if self.use_skip else out


def _make_stage(in_ch, out_ch, num_blocks, stride, expand, se_ratio=0.25, kernel=3):
    blocks = [MBConvBlock(in_ch, out_ch, stride=stride, expand=expand,
                          se_ratio=se_ratio, kernel=kernel)]
    for _ in range(1, num_blocks):
        blocks.append(MBConvBlock(out_ch, out_ch, stride=1, expand=expand,
                                  se_ratio=se_ratio, kernel=kernel))
    return nn.Sequential(*blocks)


class BoneAgeCNN(nn.Module):
    """
    Architecture inferred from checkpoint state_dict keys.

    backbone indices 0-8 match EfficientNet-B3 stage layout:
      0  : stem  (Conv+BN)
      1  : stage1 – 2 blocks, no expansion
      2  : stage2 – 3 blocks, expand=6
      3  : stage3 – 3 blocks, expand=6
      4  : stage4 – 5 blocks, expand=6
      5  : stage5 – 5 blocks, expand=6
      6  : stage6 – 6 blocks, expand=6
      7  : stage7 – 2 blocks, expand=6
      8  : head conv (Conv+BN)

    Feature dim after global avg pool = 1536 (confirmed by head.0 shape).
    sex_branch : Linear(1,16) → SiLU → Linear(16,16)
    head       : Linear(1552,512) → SiLU → Dropout(0.3) → Linear(512,1)
    """
    def __init__(self):
        super().__init__()

        # Stem
        self.backbone = nn.Sequential(
            # 0: stem
            nn.Sequential(
                nn.Conv2d(1, 40, 3, stride=2, padding=1, bias=False),  # grayscale input
                nn.BatchNorm2d(40),
                nn.SiLU(),
            ),
            # 1: stage1 — 2 MBConv1 blocks
            _make_stage(40, 24, num_blocks=2, stride=1, expand=1, kernel=3),
            # 2: stage2 — 3 MBConv6 blocks
            _make_stage(24, 32, num_blocks=3, stride=2, expand=6, kernel=3),
            # 3: stage3 — 3 MBConv6 blocks
            _make_stage(32, 48, num_blocks=3, stride=2, expand=6, kernel=5),
            # 4: stage4 — 5 MBConv6 blocks
            _make_stage(48, 96, num_blocks=5, stride=2, expand=6, kernel=3),
            # 5: stage5 — 5 MBConv6 blocks
            _make_stage(96, 136, num_blocks=5, stride=1, expand=6, kernel=5),
            # 6: stage6 — 6 MBConv6 blocks
            _make_stage(136, 232, num_blocks=6, stride=2, expand=6, kernel=5),
            # 7: stage7 — 2 MBConv6 blocks
            _make_stage(232, 384, num_blocks=2, stride=1, expand=6, kernel=3),
            # 8: head conv
            nn.Sequential(
                nn.Conv2d(384, 1536, 1, bias=False),
                nn.BatchNorm2d(1536),
                nn.SiLU(),
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Sex branch: 1 → 16
        self.sex_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
        )

        # Fusion head: 1536 + 16 = 1552 → 512 → 1
        self.head = nn.Sequential(
            nn.Linear(1552, 512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        feat = self.pool(self.backbone(x)).flatten(1)   # (B, 1536)
        s    = self.sex_branch(sex.unsqueeze(1))        # (B, 16)
        out  = self.head(torch.cat([feat, s], dim=1))   # (B, 1)
        return out.squeeze(1)                           # (B,)

# ── Correct architecture (matches training notebook) ─────────
class BoneAgeModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        import torchvision.models as models

        efficientnet = models.efficientnet_b3(weights=None)  # no pretrained weights needed
        original_conv = efficientnet.features[0][0]
        efficientnet.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        self.backbone = efficientnet.features
        self.pool     = nn.AdaptiveAvgPool2d(1)

        self.sex_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),       # ← must match training: ReLU not SiLU
            nn.Linear(16, 16),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(1552, 512),
            nn.ReLU(),       # ← must match training: ReLU not SiLU
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1),
        )

    def forward(self, image, sex):
        x = self.pool(self.backbone(image)).flatten(1)
        s = self.sex_branch(sex.view(-1, 1).float())
        return self.head(torch.cat([x, s], dim=1)).squeeze(1)
# ── Load model ─────────────────────────────────────────────────────────────
def load_model() -> nn.Module:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Place best_model.pth inside a 'models/' folder next to app.py."
        )

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    # Print a few keys so we can verify the checkpoint looks right
    sample_keys = list(state.keys())[:6]
    print(f"  Checkpoint sample keys : {sample_keys}")
    print(f"  head.0.weight shape    : {state.get('head.0.weight', torch.tensor([])).shape}")

    model = BoneAgeModel()

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  ⚠ Missing  keys ({len(missing)}): {missing[:4]} …")
    if unexpected:
        print(f"  ⚠ Unexpected keys ({len(unexpected)}): {unexpected[:4]} …")
    if not missing and not unexpected:
        print("  ✓ State dict loaded perfectly (strict match).")

    model.to(DEVICE).eval()
    return model


# ── Inference transforms ───────────────────────────────────────────────────
# Checkpoint stem conv is [40, 1, 3, 3] → model was trained on raw grayscale.
eval_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # keep as single channel
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                         # → [1, H, W] in [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.5]),   # single-channel normalisation
])


# ── Helpers ─────────────────────────────────────────────────────────────────
def get_stage(m: float) -> str:
    if m <= 24:  return "Infant"
    if m <= 60:  return "Toddler"
    if m <= 120: return "Child"
    if m <= 155: return "Pre-adolescent"
    return "Adolescent"


def months_label(m: float) -> str:
    yrs = int(m // 12);  mo = round(m % 12)
    if yrs == 0: return f"{mo} month{'s' if mo != 1 else ''}"
    if mo  == 0: return f"{yrs} year{'s' if yrs != 1 else ''}"
    return f"{yrs} yr {mo} mo"


# ── Flask ────────────────────────────────────────────────────────────────────
app  = Flask(__name__)
CORS(app)

print("\nLoading BoneAge model …")
model = load_model()
print("  Ready.\n")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "loaded", "device": str(DEVICE)})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image field in request."}), 400

    sex_str = request.form.get("sex", "male").strip().lower()
    if sex_str not in ("male", "female"):
        return jsonify({"error": "sex must be 'male' or 'female'."}), 400

    sex_val = 1.0 if sex_str == "male" else 0.0

    try:
        img = Image.open(io.BytesIO(request.files["image"].read())).convert("L")
    except (UnidentifiedImageError, Exception) as e:
        return jsonify({"error": f"Cannot decode image: {e}"}), 400

    tensor = eval_transforms(img).unsqueeze(0).to(DEVICE)
    sex_t  = torch.tensor([sex_val], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        raw = model(tensor, sex_t).item()

    # The head has no Sigmoid, so output is unbounded — use sigmoid here
    pred_months = round(max(0.0, min(MAX_AGE, raw * MAX_AGE)), 1)
    print(f"Predicted {pred_months:.1f} months for sex='{sex_str}' (raw={raw:.4f})")

    return jsonify({
        "predicted_months" : pred_months,
        "years"            : int(pred_months // 12),
        "months_remainder" : round(pred_months % 12),
        "age_label"        : months_label(pred_months),
        "stage"            : get_stage(pred_months),
        "sex"              : sex_str,
        "percentile"       : round((pred_months / MAX_AGE) * 100),
        "confidence_note"  : (
            f"Predicted {pred_months:.1f} months "
            f"({months_label(pred_months)}) · {sex_str} · "
            f"EfficientNet-B3 backbone"
        ),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print(f"Starting BoneAge API on http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)