#!/usr/bin/env python3
"""
Rebuild Part 3 -- Literature Comparison notebooks (v2, protocol-identical to Part 2).

Every generated notebook reuses the EXACT data/aug/pre-resize-cache cell, the EXACT
loss cell and the EXACT metrics cell from step2.ipynb (the notebook that produced the
FSCG-UNet central result, test mIoU 0.5366 at 320px). Only the model architecture and a
model-agnostic train loop are swapped, so literature models are trained under identical
conditions: same 2560-sample oversampled train set + two-pipeline augmentation, same
CE+Dice+Focal+Lovasz composite, same AdamW/CAWR schedule/AMP/grad-clip, same 8h budget,
same held-out REAL test split with TTA. The only difference vs FSCG-UNet is the
architecture itself -- a controlled comparison.

Run:  python3 build_comparison_v2.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
STEP2 = os.path.join(HERE, "..", "part2", "step2.ipynb")
OUT_DIR = os.path.join(HERE, "generated")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Extract the three reusable cells verbatim from step2.ipynb (proven on Kaggle)
# ----------------------------------------------------------------------------
with open(os.path.join(HERE, "..", "part2", "step2.ipynb")) as f:
    _step2 = json.load(f)
_cells = {i: "".join(c.get("source", [])) for i, c in enumerate(_step2["cells"])
          if c["cell_type"] == "code"}
KEY_CELLS = {"data": 8, "loss": 17, "metrics": 19}
extracted = {k: _cells[v] for k, v in KEY_CELLS.items()}
print("step2 cells extracted:",
      {k: len(v.splitlines()) for k, v in extracted.items()})

CELL_DATA = extracted["data"]
CELL_LOSS = extracted["loss"]

# adapt the metrics cell to single-output models via a _logits helper
CELL_METRICS = extracted["metrics"]
assert "main, _, _, _ = model(images)" in CELL_METRICS
assert "main_h, _, _, _ = model(h)" in CELL_METRICS
assert "main_v, _, _, _ = model(v)" in CELL_METRICS
assert "main_hv, _, _, _ = model(hv)" in CELL_METRICS
assert "torch.argmax(model(imgs)[0], dim=1)" in CELL_METRICS
CELL_METRICS = (
    "\ndef _logits(model, x):\n"
    "    y = model(x)\n"
    "    return y[0] if isinstance(y, (tuple, list)) else y\n"
    + CELL_METRICS
    .replace("main, _, _, _ = model(images)", "main = _logits(model, images)")
    .replace("main_h, _, _, _ = model(h)", "main_h = _logits(model, h)")
    .replace("main_v, _, _, _ = model(v)", "main_v = _logits(model, v)")
    .replace("main_hv, _, _, _ = model(hv)", "main_hv = _logits(model, hv)")
    .replace("torch.argmax(model(imgs)[0], dim=1)", "torch.argmax(_logits(model, imgs), dim=1)")
)

# ----------------------------------------------------------------------------
# Shared cells (config, loaders, MainLoss, train loop, final eval, package)
# ----------------------------------------------------------------------------
CELL_INSTALL = """import subprocess, sys
packages = [
    "segmentation-models-pytorch==0.3.3",
    "albumentations==1.3.1",
    "timm",
]
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
print("Packages ready.")"""

CELL_IMPORTS = """import os, sys, json, random, warnings, time, pickle, copy, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

warnings.filterwarnings("ignore")
print("Imports OK. PyTorch", torch.__version__)"""

CELL_CONFIG = """SMOKE_TEST  = os.environ.get("FSCG_SMOKE") == "1"
DEBUG_SHORT = os.environ.get("FSCG_DEBUG") == "1" or SMOKE_TEST

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

IN_KAGGLE = os.path.isdir("/kaggle/input")
if IN_KAGGLE:
    WORKING_DIR = "/kaggle/working"
else:
    WORKING_DIR = os.path.join(os.path.dirname(os.getcwd()), "working")
    os.makedirs(WORKING_DIR, exist_ok=True)

OUTPUT_DIR = os.path.join(WORKING_DIR, "fed_salgate_outputs")
CKPT_DIR   = os.path.join(WORKING_DIR, "checkpoints")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")
for d in [OUTPUT_DIR, CKPT_DIR, PLOT_DIR, os.path.join(OUTPUT_DIR, "tables")]:
    os.makedirs(d, exist_ok=True)

HAS_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")
if HAS_CUDA:
    torch.backends.cudnn.benchmark = True
print("Device:", DEVICE)


def _find_part1_dir():
    if os.path.exists(os.path.join(OUTPUT_DIR, "config.json")):
        return OUTPUT_DIR
    _pin = "/kaggle/input/models/YOUR_KAGGLE_USERNAME/fedall/tensorflow2/default/1/fed_salgate_outputs"
    if IN_KAGGLE and os.path.exists(os.path.join(_pin, "config.json")):
        return _pin
    if IN_KAGGLE:
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files and "client_indices.pkl" in files:
                if os.path.basename(root) == "fed_salgate_outputs":
                    return root
    return None


PART1_DIR = os.environ.get("PART1_DIR", "") or (_find_part1_dir() or "")
print(f"Part 1 outputs: {PART1_DIR}")
if not PART1_DIR or not os.path.exists(os.path.join(PART1_DIR, "config.json")):
    raise FileNotFoundError("Part-1 outputs not found. Attach the feds1 model (Part 1 output).")

with open(os.path.join(PART1_DIR, "config.json")) as f:
    CFG = json.load(f)

NUM_CLASSES = int(CFG.get("NUM_CLASSES", 10))
CLASS_NAMES = CFG.get("CLASS_NAMES", ['Background', 'Building-Flooded', 'Building-NonFlooded',
                                      'Road-Flooded', 'Road-NonFlooded', 'Water',
                                      'Tree', 'Vehicle', 'Pool', 'Grass'])
IMG_SIZE   = int(os.environ.get("FSCG_IMG", "320"))
BATCH_SIZE = int(os.environ.get("FSCG_BATCH", "4")) if not SMOKE_TEST else 2
class_weights = CFG.get("class_weights")
NUM_CLIENTS = int(CFG.get("NUM_CLIENTS", 3))
print(f"IMG_SIZE={IMG_SIZE}  BATCH={BATCH_SIZE}  NUM_CLASSES={NUM_CLASSES}")

FSCG_EPOCHS    = os.environ.get("FSCG_EPOCHS", "3" if SMOKE_TEST else "100")
FSCG_MAX_HOURS = os.environ.get("FSCG_MAX_HOURS", "8.0")
FSCG_VALLOG    = os.environ.get("FSCG_VALLOG", "64")
FSCG_WORKERS   = os.environ.get("FSCG_WORKERS", "4")
EPOCHS    = int(FSCG_EPOCHS)
MAX_HOURS = float(FSCG_MAX_HOURS)
VAL_LOG_N = int(FSCG_VALLOG)
WORKERS   = int(FSCG_WORKERS)
print(f"EPOCHS={EPOCHS} MAX_HOURS={MAX_HOURS}h VAL_LOG_N={VAL_LOG_N} WORKERS={WORKERS}")
print(">>> Literature baseline: SAME data/aug/loss/optimizer/budget as Part-2 FSCG-UNet (320px). "
      "Only the architecture differs.")

_client_pkl = os.path.join(PART1_DIR, "client_indices.pkl")
CLIENT_INDICES = None
if os.path.exists(_client_pkl):
    with open(_client_pkl, "rb") as f:
        CLIENT_INDICES = pickle.load(f)
    print(f"client sizes: {[len(c) for c in CLIENT_INDICES]}")"""

CELL_LOADERS = """_wk = WORKERS if (not SMOKE_TEST and HAS_CUDA) else 0
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=_wk, pin_memory=True,
                          persistent_workers=(_wk > 0), drop_last=True)
val_loader   = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=_wk, pin_memory=True,
                          persistent_workers=(_wk > 0))
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=_wk, pin_memory=True,
                          persistent_workers=(_wk > 0))
print(f"Loaders ready (workers={_wk}).")"""

CELL_MAINLOSS = """class MainLoss(nn.Module):
    \"\"\"Main term of the Part-2 composite loss (CE + Dice + Focal + Lovasz),
    model-agnostic (takes plain logits + target).\"\"\"
    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
        self.ce    = nn.CrossEntropyLoss(weight=class_weights)
        self.dice  = DiceLoss(num_classes)
        self.focal = FocalLoss(alpha=0.75, gamma=3.0, weight=class_weights)

    def forward(self, logits, target):
        return (0.4 * self.ce(logits, target)
                + 0.35 * self.dice(logits, target)
                + 0.15 * self.focal(logits, target)
                + 0.4 * lovasz_softmax(logits, target))
print("MainLoss ready (same composite as Part-2 FSCG-UNet).")"""

CELL_TRAIN = """def train_centralized(name, model, epochs=EPOCHS, lr=3e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    criterion = MainLoss(NUM_CLASSES, class_weights).to(DEVICE)
    scaler = torch.amp.GradScaler("cuda", enabled=HAS_CUDA) if HAS_CUDA else None
    history = {"epoch": [], "train_loss": [], "val_miou": []}
    best_iou, best_state = -1.0, None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        if (time.time() - t0) / 3600 >= MAX_HOURS:
            print(f"  [TIME-BUDGET] {MAX_HOURS:.1f}h reached after {epoch-1} epochs; stopping (best checkpoint kept).")
            break
        model.train()
        total_loss, n_seen = 0.0, 0
        loop = tqdm(train_loader, desc=f"[{name}] epoch {epoch}")
        for imgs, msks, _ in loop:
            imgs = imgs.to(DEVICE, non_blocking=True)
            msks = msks.long().to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=HAS_CUDA):
                logits = model(imgs)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                loss = criterion(logits, msks)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            total_loss += loss.item() * imgs.size(0); n_seen += imgs.size(0)
            loop.set_postfix(loss=loss.item())
        scheduler.step()
        vm, per_class, _ = compute_miou(model, val_loader, use_tta=True)  # real val GT, deterministic
        history["epoch"].append(epoch)
        history["train_loss"].append(total_loss / max(1, n_seen))
        history["val_miou"].append(vm)
        if vm > best_iou:
            best_iou = vm
            best_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_val_iou.pth"))
        print(f"[{name}] epoch {epoch:3d}/{epochs}  loss={history['train_loss'][-1]:.4f}  "
              f"val_mIoU={vm:.4f}  best={best_iou:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_iou


def _sanity(model, tag):
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE))
        if isinstance(y, (tuple, list)):
            y = y[0]
        print(f"{tag}: out={tuple(y.shape)} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        assert y.shape[-2:] == (IMG_SIZE, IMG_SIZE), "model output resolution must equal IMG_SIZE"


print("Training functions ready.")
model = make_model().to(DEVICE)
_sanity(model, "MODEL_DISPLAY")"""

CELL_FINAL_EVAL = """model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best_val_iou.pth"),
                                 map_location=DEVICE))
test_miou, test_per_class, test_pix = compute_miou(model, test_loader, use_tta=True)
print(f"\\n===== MODEL_DISPLAY -- TEST (TTA, real GT) =====")
print(f"  mIoU      : {test_miou:.4f}")
print(f"  Pixel Acc : {test_pix:.4f}")

result = {
    "model_id": "MODEL_ID",
    "display_name": "MODEL_DISPLAY",
    "type": "centralized (literature)",
    "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
    "epochs_trained": len(history["epoch"]),
    "heldout_miou": test_miou,
    "pixel_acc": test_pix,
    "best_val_miou": best_val,
    "per_class_iou": {str(k): v for k, v in test_per_class.items()},
    "history": history,
    "config": {"img_size": IMG_SIZE, "batch": BATCH_SIZE, "lr": 3e-5,
               "epochs": EPOCHS, "n_train": len(train_dataset)},
    "source_ref": SOURCE_REF,
}
with open(os.path.join(OUTPUT_DIR, "result_MODEL_ID.json"), "w") as f:
    json.dump(result, f, indent=2, default=float)
print(f"Saved: result_MODEL_ID.json")

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(history["epoch"], history["train_loss"], "o-", color="tab:red", label="train loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss", color="tab:red")
ax2 = ax1.twinx()
ax2.plot(history["epoch"], history["val_miou"], "s-", color="tab:blue", label="val mIoU (real GT)")
ax2.set_ylabel("val mIoU", color="tab:blue")
plt.title("MODEL_DISPLAY -- Training Curve")
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "31_curve_MODEL_ID.png"), dpi=150, bbox_inches="tight")
plt.show(); print("31_curve_MODEL_ID.png saved.")"""

CELL_FINAL_TRAIN = """t0 = time.time()
model, history, best_val = train_centralized('MODEL_DISPLAY', model)
print(f'Training done in {(time.time()-t0)/60:.1f} min | best val mIoU = {best_val:.4f}')"""

CELL_PACKAGE = """import shutil, datetime, zipfile
_t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_zip = os.path.join(WORKING_DIR, f"Part3_MODEL_ID_{_t}.zip")
with zipfile.ZipFile(_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for _r, _, _fs in os.walk(OUTPUT_DIR):
        for _f in _fs:
            if _f.endswith(".pth"):
                continue
            _p = os.path.join(_r, _f)
            z.write(_p, os.path.relpath(_p, WORKING_DIR))
    z.write(os.path.join(CKPT_DIR, "best_val_iou.pth"), "checkpoints/best_val_iou.pth")
print(f"Done: {_zip} ({os.path.getsize(_zip)/1e6:.2f} MB)")
if IN_KAGGLE:
    print("Kaggle: /kaggle/working/ is auto-saved. Download from the Output tab.")"""

# ----------------------------------------------------------------------------
# Model architectures (kept from the original builder)
# ----------------------------------------------------------------------------
MODEL_UNET_R34 = """def _make_model():
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name='resnet34', encoder_weights='imagenet',
                    in_channels=3, classes=NUM_CLASSES)"""

MODEL_UNET_EB0 = """def _make_model():
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name='efficientnet-b0', encoder_weights='imagenet',
                    in_channels=3, classes=NUM_CLASSES)"""

MODEL_SEGNET_R34 = """class SegNetStyle(nn.Module):
    \"\"\"SegNet-style decoder (no skip connections) on a ResNet-34 backbone.\"\"\"
    def __init__(self, encoder_name='resnet34', num_classes=NUM_CLASSES):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=3, weights='imagenet')
        chs = self.encoder.out_channels[1:]
        self.decoder = nn.ModuleList()
        in_ch = chs[-1]
        for out_ch in reversed(chs):
            self.decoder.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)))
            in_ch = out_ch
        self.head = nn.Conv2d(chs[0], num_classes, 1)
    def forward(self, x):
        feats = self.encoder(x)
        x = feats[-1]
        for blk in self.decoder:
            x = blk(x)
        return self.head(x)

def _make_model():
    return SegNetStyle('resnet34', NUM_CLASSES)"""

MODEL_SEGNET_EB0 = """class SegNetStyle(nn.Module):
    \"\"\"SegNet-style decoder (no skip connections) on an EfficientNet-b0 backbone.\"\"\"
    def __init__(self, encoder_name='efficientnet-b0', num_classes=NUM_CLASSES):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=3, weights='imagenet')
        chs = self.encoder.out_channels[1:]
        self.decoder = nn.ModuleList()
        in_ch = chs[-1]
        for out_ch in reversed(chs):
            self.decoder.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)))
            in_ch = out_ch
        self.head = nn.Conv2d(chs[0], num_classes, 1)
    def forward(self, x):
        feats = self.encoder(x)
        x = feats[-1]
        for blk in self.decoder:
            x = blk(x)
        return self.head(x)

def _make_model():
    return SegNetStyle('efficientnet-b0', NUM_CLASSES)"""

MODEL_DLV3P_EB0 = """def _make_model():
    import segmentation_models_pytorch as smp
    return smp.DeepLabV3Plus(encoder_name='efficientnet-b0', encoder_weights='imagenet',
                             in_channels=3, classes=NUM_CLASSES)"""

MODEL_FLOODNETLITE = """class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, padding=k//2, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(4, ch//r), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(4, ch//r), ch, 1), nn.Sigmoid())
    def forward(self, x): return x * self.se(x)

class FloodNetLite(nn.Module):
    \"\"\"FloodNet-Lite (Thirugnanasammandamoorthi et al. 2025, JSTARS):
    MobileNetV3-small encoder + depthwise-separable decoder + SE attention.\"\"\"
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        import torchvision
        self.backbone = torchvision.models.mobilenet_v3_small(
            weights='IMAGENET1K_V1' if pretrained else None).features
        self.stages = [0, 1, 2, 4, 7, 11]
        enc_chs = [16, 16, 24, 40, 48, 96]
        self.decoder = nn.ModuleList()
        in_ch = 96
        for out_ch in reversed(enc_chs[:-1]):
            self.decoder.append(nn.ModuleList([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                DepthwiseSeparableConv(in_ch, out_ch),
                SEBlock(out_ch)]))
            in_ch = out_ch
        self.head = nn.Conv2d(enc_chs[0], num_classes, 1)
    def forward(self, x):
        feats = []
        blks = list(self.backbone)
        prev = 0
        for s in self.stages:
            for b in blks[prev:s]:
                x = b(x)
            feats.append(x)
            prev = s
        x = feats[-1]
        for up, dw, se in self.decoder:
            x = up(x); x = dw(x); x = se(x)
        return self.head(x)

def _make_model():
    return FloodNetLite(NUM_CLASSES, pretrained=True)"""

# ----------------------------------------------------------------------------
# FedSeg baseline (Part 3.07) -- federated DeepLabV3+ with pixel contrastive
# ----------------------------------------------------------------------------
MODEL_FEDSEG = """def _make_model():
    import segmentation_models_pytorch as smp
    return smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights='imagenet',
                             in_channels=3, classes=NUM_CLASSES)"""

CELL_FEDSEG_CONFIG = """FL_ROUNDS        = int(os.environ.get("FSCG_FL_ROUNDS", "3" if SMOKE_TEST else "20"))
FL_LOCAL_EPOCHS  = int(os.environ.get("FSCG_FL_EPOCHS", "1" if SMOKE_TEST else "2"))
CLIENT_SUBSAMPLE = int(os.environ.get("FSCG_FL_SUBSAMPLE", "60" if SMOKE_TEST else "120"))
CONTRAST_TEMP    = 0.1
CONTRAST_LAMBDA  = 0.1
NUM_NEGATIVES    = 64
print(f"FedSeg FL: rounds={FL_ROUNDS} local_epochs={FL_LOCAL_EPOCHS} subsample={CLIENT_SUBSAMPLE}")

class PixelContrastMemory:
    def __init__(self, num_classes=NUM_CLASSES, feat_dim=256, momentum=0.999):
        self.num_classes = num_classes; self.feat_dim = feat_dim; self.momentum = momentum

def back_ce_loss(logits, masks):
    return F.cross_entropy(logits, masks)

def pixel_contrastive_loss(feats, masks, protos, temperature=CONTRAST_TEMP,
                           num_neg=NUM_NEGATIVES, num_pos=16):
    B, C, H, W = feats.shape
    feats = F.normalize(feats, dim=1)
    masks = masks.view(B, H * W)
    feats = feats.view(B, C, H * W)
    protos = F.normalize(protos, dim=1)
    loss_tot = feats.new_zeros(())
    n_acc = 0
    for b in range(B):
        m = masks[b]; f = feats[b]
        for c in range(protos.shape[0]):
            idx = (m == c).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            n_anch = min(num_pos, idx.numel())
            anch_idx = idx[torch.randperm(idx.numel())[:n_anch]]
            anchors = f[:, anch_idx].t()
            pos_sim = (anchors * protos[c:c+1]).sum(dim=1) / temperature
            neg_idx = torch.randperm(protos.shape[0])[:num_neg]
            neg_sim = torch.matmul(anchors, protos[neg_idx].t()) / temperature
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(n_anch, dtype=torch.long, device=feats.device)
            loss_tot = loss_tot + F.cross_entropy(logits, labels)
            n_acc += n_anch
    return loss_tot / max(1, n_acc)
print("FedSeg loss ready.")"""

CELL_FEDSEG_TRAIN = """def fed_aggregate(weights_list, client_weights):
    w = torch.tensor(client_weights, dtype=torch.float32, device=DEVICE)
    w = w / (w.sum() + 1e-8)
    g = {}
    for key in weights_list[0].keys():
        st = torch.stack([ww[key].float() for ww in weights_list])
        shape = [len(weights_list)] + [1] * (st.dim() - 1)
        g[key] = (st * w.view(shape)).sum(dim=0)
    return g

def _local_train(global_model, client_idx, backbone_fn, local_epochs, lr, protos, round_idx=1):
    idx_all = np.array(CLIENT_INDICES[client_idx])
    rng = np.random.RandomState(SEED + client_idx * 7919 + round_idx * 104729)
    if len(idx_all) > CLIENT_SUBSAMPLE:
        idx = idx_all[rng.choice(len(idx_all), CLIENT_SUBSAMPLE, replace=False)]
    else:
        idx = idx_all
    sub_ds = Subset(train_dataset, idx.tolist())
    loader = DataLoader(sub_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=WORKERS if HAS_CUDA else 0, pin_memory=True)
    client_model = backbone_fn().to(DEVICE)
    client_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
    optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    client_model.train()
    total_loss = 0.0; n_seen = 0
    for _ in range(local_epochs):
        for imgs, msks, _ in loader:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            optimizer.zero_grad()
            logits = _logits(client_model, imgs)
            seg_loss = back_ce_loss(logits, msks)
            feats = client_model.encoder(imgs)[-1]
            feats = F.interpolate(feats, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
            con_loss = pixel_contrastive_loss(feats, msks, protos)
            loss = seg_loss + CONTRAST_LAMBDA * con_loss
            loss.backward(); optimizer.step()
            total_loss += loss.item() * imgs.size(0); n_seen += imgs.size(0)
    del loader
    return client_model, total_loss / max(1, n_seen)

def run_fedseg(name, backbone_fn, rounds=FL_ROUNDS, local_epochs=FL_LOCAL_EPOCHS, lr=3e-4):
    global_model = backbone_fn().to(DEVICE)
    protos = F.normalize(torch.randn(NUM_CLASSES, 256, device=DEVICE), dim=1)
    history = {"round": [], "avg_train_loss": [], "val_miou": []}
    client_sizes = [len(CLIENT_INDICES[c]) for c in range(NUM_CLIENTS)] if CLIENT_INDICES else [1.0] * NUM_CLIENTS
    best_miou, best_state = -1.0, None
    for round_idx in range(1, rounds + 1):
        local_weights, local_models, local_losses = [], [], []
        for c in range(NUM_CLIENTS):
            cm, al = _local_train(global_model, c, backbone_fn, local_epochs, lr, protos, round_idx)
            local_weights.append(copy.deepcopy(cm.state_dict())); local_models.append(cm); local_losses.append(al)
        global_model.load_state_dict(fed_aggregate(local_weights, client_sizes))
        vm, per_class, _ = compute_miou(global_model, val_loader, use_tta=True)
        history["round"].append(round_idx)
        history["avg_train_loss"].append(float(np.mean(local_losses)))
        history["val_miou"].append(vm)
        if vm > best_miou:
            best_miou = vm; best_state = copy.deepcopy(global_model.state_dict())
            torch.save(global_model.state_dict(), os.path.join(CKPT_DIR, "best_val_iou.pth"))
        for m in local_models:
            del m
        torch.cuda.empty_cache()
        if round_idx % 5 == 0 or round_idx == 1:
            print(f"[{name}] R{round_idx:3d}/{rounds}  loss={history['avg_train_loss'][-1]:.4f}  val_mIoU={vm:.4f}")
    if best_state is not None:
        global_model.load_state_dict(best_state)
    return global_model, history

print("FedSeg FL functions ready.")"""

CELL_FEDSEG_RUN = """print("=" * 70)
print("FedSeg baseline: FL with back-CE + pixel-contrastive (Miao et al., CVPR 2023)")
print("=" * 70)
t0 = time.time()
fedseg_model, fedseg_history = run_fedseg('FedSeg', make_model)
print(f'FedSeg done in {(time.time()-t0)/60:.1f} min | best round mIoU = {max(fedseg_history["val_miou"]):.4f}')

model = fedseg_model
model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best_val_iou.pth"), map_location=DEVICE))
test_miou, test_per_class, test_pix = compute_miou(model, test_loader, use_tta=True)
print(f"\\n===== DeepLabV3+ / FedSeg -- TEST (TTA, real GT) =====")
print(f"  mIoU      : {test_miou:.4f}")
print(f"  Pixel Acc : {test_pix:.4f}")

result = {
    "model_id": "fedseg_deeplabv3p",
    "display_name": "DeepLabV3+ / FedSeg (FL baseline)",
    "type": "federated (literature)",
    "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
    "rounds": len(fedseg_history["round"]),
    "heldout_miou": test_miou,
    "pixel_acc": test_pix,
    "best_val_miou": max(fedseg_history["val_miou"]),
    "per_class_iou": {str(k): v for k, v in test_per_class.items()},
    "history": fedseg_history,
    "config": {"img_size": IMG_SIZE, "batch": BATCH_SIZE, "rounds": FL_ROUNDS,
               "local_epochs": FL_LOCAL_EPOCHS, "contrast_lambda": CONTRAST_LAMBDA},
    "source_ref": "[37] Miao et al., *CVPR*, 2023 -- FedSeg: federated semantic segmentation via class-balanced CE + pixel contrastive",
}
with open(os.path.join(OUTPUT_DIR, "result_fedseg_deeplabv3p.json"), "w") as f:
    json.dump(result, f, indent=2, default=float)
print('\\nSaved: result_fedseg_deeplabv3p.json')

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(fedseg_history["round"], fedseg_history["avg_train_loss"], "o-", color="tab:red", label="avg train loss")
ax1.set_xlabel("Round"); ax1.set_ylabel("Avg train loss", color="tab:red")
ax2 = ax1.twinx()
ax2.plot(fedseg_history["round"], fedseg_history["val_miou"], "s-", color="tab:blue", label="val mIoU (real GT)")
ax2.set_ylabel("val mIoU", color="tab:blue")
plt.title("DeepLabV3+ / FedSeg -- Federated Training Curve")
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "31_curve_fedseg_deeplabv3p.png"), dpi=150, bbox_inches="tight")
plt.show(); print('31_curve_fedseg_deeplabv3p.png saved.')"""

# ----------------------------------------------------------------------------
# Notebook builders
# ----------------------------------------------------------------------------
def cell_code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}

def cell_md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def header_md(title, desc, ref, federated=False):
    mode = "Federated" if federated else "Centralized"
    return f"""# {title}

**Part 3 literature comparison ({mode})** -- protocol-identical to the Part-2 FSCG-UNet run:
same 320px input, same Part-1 sample JSONs + pre-resize cache, same oversampled 2560 train set
and two-pipeline augmentation, same composite loss (CE+Dice+Focal+Lovasz), same AdamW /
CosineAnnealingWarmRestarts / AMP / grad-clip / 8h budget, and final evaluation on the **same
held-out REAL test split with TTA**. Only the model architecture differs from FSCG-UNet.

{desc}

**Reference:** {ref}

---
### What you need attached on Kaggle
1. The FloodNet dataset (`aerial-imagery-dataset-floodnet-challenge` + `floodnet_fixed`).
2. The **feds1 model** (Part 1 + Part 2 outputs) as an input -- auto-discovered at
   `/kaggle/input/models/YOUR_KAGGLE_USERNAME/fedall/...`. If missing, nothing will run."""

def build_central(model_def, mid, display, ref):
    cells = [
        cell_md(header_md(f"Part 3.{MODELS.index(md) + 1} -- {display}",
                          f"Centralized training of **{display}** under the exact Part-2 recipe.",
                          ref)),
        cell_md("## ⚙️ Install packages"), cell_code(CELL_INSTALL),
        cell_md("## 📚 Imports"), cell_code(CELL_IMPORTS),
        cell_md("## ⚙️ Config + Part-1 discovery"), cell_code(CELL_CONFIG),
        cell_md("## 🗂️ Dataset (exact Part-2 pipeline: samples + cache)"), cell_code(CELL_DATA),
        cell_md("## 🚚 Loaders"), cell_code(CELL_LOADERS),
        cell_md("## 📊 Loss (Part-2 composite)"), cell_code(CELL_LOSS),
        cell_md("## 📊 Metrics + TTA"), cell_code(CELL_METRICS),
        cell_md("## 🧠 Model architecture"), cell_code(model_def),
        cell_md("## 🏋️ Train (identical recipe to Part-2 FSCG-UNet)"), cell_code(CELL_MAINLOSS),
    ]
    cells.append(cell_code(CELL_TRAIN
                           .replace("MODEL_DISPLAY", display)))
    cells.append(cell_code(CELL_FINAL_TRAIN.replace("MODEL_DISPLAY", display)))
    cells.append(cell_md("## 🎯 Final evaluation on held-out REAL test (TTA)"))
    cells.append(cell_code(CELL_FINAL_EVAL
                           .replace("MODEL_ID", mid)
                           .replace("MODEL_DISPLAY", display)
                           .replace('"source_ref": SOURCE_REF,', f'"source_ref": {json.dumps(ref)},')))
    cells.append(cell_md("## 📦 Package outputs"))
    cells.append(cell_code(CELL_PACKAGE.replace("MODEL_ID", mid).replace("MODEL_DISPLAY", display)))
    return cells


def build_fedseg():
    display = "DeepLabV3+ / FedSeg (FL baseline)"
    cells = [
        cell_md(header_md("Part 3.07 -- DeepLabV3+ / FedSeg (Federated Learning baseline)",
                          "Federated training of **DeepLabV3+ (ResNet-34)** with the FedSeg loss "
                          "(back-CE + pixel-contrastive, Miao et al., CVPR 2023) on the SAME "
                          "Part-1 client splits, evaluated on the SAME held-out REAL test split as Fed-SalGate.",
                          "[37] Miao et al., *CVPR*, 2023 -- FedSeg", federated=True)),
        cell_md("## ⚙️ Install packages"), cell_code(CELL_INSTALL),
        cell_md("## 📚 Imports"), cell_code(CELL_IMPORTS),
        cell_md("## ⚙️ Config + Part-1 discovery"), cell_code(CELL_CONFIG),
        cell_md("## 🗂️ Dataset (exact Part-2 pipeline)"), cell_code(CELL_DATA),
        cell_md("## 🚚 Loaders"), cell_code(CELL_LOADERS),
        cell_md("## 📊 Metrics + TTA"), cell_code(CELL_METRICS),
        cell_md("## 🧠 FedSeg config + pixel-contrastive loss"), cell_code(CELL_FEDSEG_CONFIG),
        cell_md("## 🧠 Model architecture (DeepLabV3+ ResNet-34)"), cell_code(MODEL_FEDSEG),
        cell_md("## 🔄 Federated training (FedSeg)"), cell_code(CELL_FEDSEG_TRAIN),
        cell_code(CELL_FEDSEG_RUN),
        cell_md("## 📦 Package outputs"),
        cell_code(CELL_PACKAGE.replace("MODEL_ID", "fedseg_deeplabv3p").replace("MODEL_DISPLAY", display)),
    ]
    return cells


MODELS = [
    dict(fname="part3_01_unet_resnet34",        id="unet_resnet34",        display="UNet (ResNet-34)",        model=MODEL_UNET_R34,   ref="[36] Karcı et al., *Natural Hazards*, 2026 -- UNet+ResNet comparison"),
    dict(fname="part3_02_unet_efficientnet_b0", id="unet_efficientnet_b0", display="UNet (EfficientNet-b0)",  model=MODEL_UNET_EB0,   ref="[36] Karcı et al., *Natural Hazards*, 2026 -- best combo (IoU 0.9271)"),
    dict(fname="part3_03_segnet_resnet34",      id="segnet_resnet34",      display="SegNet (ResNet-34)",      model=MODEL_SEGNET_R34, ref="[36] Karcı et al., *Natural Hazards*, 2026 -- SegNet+ResNet"),
    dict(fname="part3_04_segnet_efficientnet_b0", id="segnet_efficientnet_b0", display="SegNet (EfficientNet-b0)", model=MODEL_SEGNET_EB0, ref="[36] Karcı et al., *Natural Hazards*, 2026 -- SegNet+EfficientNet"),
    dict(fname="part3_05_deeplabv3p_efficientnet_b0", id="deeplabv3p_efficientnet_b0", display="DeepLabV3+ (EfficientNet-b0)", model=MODEL_DLV3P_EB0, ref="[36] Karcı et al., *Natural Hazards*, 2026 -- DeepLabV3+ + EfficientNet"),
    dict(fname="part3_06_floodnet_lite",        id="floodnet_lite",        display="FloodNet-Lite",           model=MODEL_FLOODNETLITE, ref="[35] Thirugnanasammandamoorthi et al., *JSTARS*, 2025 -- lightweight optimized UNet"),
]

for md in MODELS:
    cells = build_central(md["model"], md["id"], md["display"], md["ref"])
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.10.12"}},
          "nbformat": 4, "nbformat_minor": 2}
    with open(os.path.join(OUT_DIR, md["fname"] + ".ipynb"), "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {md['fname']}.ipynb  ({len(cells)} cells)")

cells = build_fedseg()
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10.12"}},
      "nbformat": 4, "nbformat_minor": 2}
with open(os.path.join(OUT_DIR, "part3_07_fedseg_deeplabv3p.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote part3_07_fedseg_deeplabv3p.ipynb  ({len(cells)} cells)")

print("\nAll Part 3 literature-comparison notebooks regenerated (v2, protocol-identical).")
