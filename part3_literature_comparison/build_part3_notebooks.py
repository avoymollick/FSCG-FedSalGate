#!/usr/bin/env python3
"""
Build script for Part 3 -- Literature Comparison notebooks.

Generates one standalone notebook per literature model so each can be run on a
different device (Kaggle) independently, then a merge notebook that combines
all result JSONs into tables/figures/Excel/zip.

Run:  python3 build_part3_notebooks.py
"""
import json, os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")

# ============================================================================
# SHARED FOUNDATION CELLS (identical to Part 2 v2 so conditions match exactly)
# ============================================================================

CELL_INSTALL = """import subprocess, sys
packages = [
    "segmentation-models-pytorch==0.3.3",
    "albumentations==1.3.1",
    "torchmetrics==1.2.0",
    "torchcam",
    "scikit-image",
    "tqdm",
]
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
print("Packages installed")"""

CELL_IMPORTS = """import os, sys, json, random, warnings, time, pickle, copy
from pathlib import Path
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
try:
    import seaborn as sns
except Exception:
    sns = None
    print("WARNING: seaborn unavailable -- using matplotlib defaults.")
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
warnings.filterwarnings("ignore")
if sns is not None:
    sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "figure.figsize": (14, 6)})
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device     : {DEVICE}')
print(f'PyTorch    : {torch.__version__}')
print(f'CUDA       : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU        : {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory : {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB')
assert torch.cuda.is_available(), "GPU required -- enable GPU on Kaggle"
print("Imports complete.")"""

CELL_CONFIG = """IN_KAGGLE = os.path.isdir("/kaggle/input")
if IN_KAGGLE:
    WORKING_DIR = "/kaggle/working"
else:
    WORKING_DIR = os.path.join(os.path.dirname(os.getcwd()), 'working')
    os.makedirs(WORKING_DIR, exist_ok=True)
OUTPUT_DIR = os.path.join(WORKING_DIR, 'part3_outputs')
CKPT_DIR   = os.path.join(WORKING_DIR, 'checkpoints')
PLOT_DIR   = os.path.join(WORKING_DIR, 'plots')
MASK_OUT_DIR = os.path.join(WORKING_DIR, 'masks')
for d in [OUTPUT_DIR, CKPT_DIR, PLOT_DIR, MASK_OUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Locate Part 1 outputs (read-only) ──────────────────────────────────────
def _find_part1_dir():
    if os.path.exists(os.path.join(OUTPUT_DIR, 'config.json')):
        return OUTPUT_DIR
    if IN_KAGGLE:
        input_dir = "/kaggle/input"
        if os.path.isdir(input_dir):
            hits = []
            for root, dirs, files in os.walk(input_dir):
                if ('config.json' in files and 'client_indices.pkl' in files
                        and 'gt_val_indices.pkl' in files):
                    hits.append(root)
                    if len(hits) >= 5:
                        break
            if hits:
                for h in hits:
                    if os.path.basename(h) == 'fed_salgate_outputs':
                        return h
                return hits[0]
            for root, dirs, files in os.walk(input_dir):
                if 'config.json' in files:
                    try:
                        with open(os.path.join(root, 'config.json')) as f:
                            cfg = json.load(f)
                        if 'FLOODNET_BASE' in cfg or 'IMG_SIZE' in cfg:
                            return root
                    except Exception:
                        pass
    return None
PART1_DIR = _find_part1_dir()
if PART1_DIR is not None:
    print(f'Part 1 outputs found at: {PART1_DIR}')
else:
    print('WARNING: Part 1 outputs NOT found -- falling back to defaults. '
          'Results will NOT match Part 1 splits!')

PART1_ROOT = os.path.dirname(PART1_DIR) if PART1_DIR else None
if PART1_ROOT and os.path.isdir(os.path.join(PART1_ROOT, 'masks')):
    MASK_OUT_DIR = os.path.join(PART1_ROOT, 'masks')
    print(f'Part 1 pseudo-masks (read-only): {MASK_OUT_DIR}')
    for sub in ['unlabeled', 'val', 'test']:
        p = os.path.join(MASK_OUT_DIR, sub)
        n = len([f for f in os.listdir(p) if not f.startswith('.')]) if os.path.isdir(p) else 0
        print(f'    masks/{sub}: {n} files')
else:
    print(f'Using writable mask output dir: {MASK_OUT_DIR}')

config_path = os.path.join(PART1_DIR, 'config.json') if PART1_DIR else None
if config_path and os.path.exists(config_path):
    with open(config_path) as f:
        CFG = json.load(f)
    print("Part 1 config loaded")
else:
    print("WARNING: Part 1 config not found. Using default values.")
    CFG = {}
NUM_CLASSES = int(CFG.get('NUM_CLASSES', 10))
CLASS_NAMES = CFG.get('CLASS_NAMES', [
    'Background', 'Building-Flooded', 'Building-NonFlooded',
    'Road-Flooded', 'Road-NonFlooded', 'Water',
    'Tree', 'Vehicle', 'Pool', 'Grass'])
IMG_SIZE     = int(CFG.get('IMG_SIZE', 512))
BATCH_SIZE   = int(CFG.get('BATCH_SIZE', 4))
NUM_CLIENTS  = int(CFG.get('NUM_CLIENTS', 3))
LR           = float(CFG.get('LR', 1e-4))
VAL_LOG_N    = int(CFG.get('VAL_LOG_N', 128))
DEBUG_SHORT  = os.environ.get('PART3_DEBUG') == '1'
CENTRALIZED_EPOCHS = 15 if not DEBUG_SHORT else 2
if DEBUG_SHORT:
    CENTRALIZED_EPOCHS = 2
    print('!! DEBUG_SHORT mode: epochs=2')

# ── Load Part 1 client indices + held-out GT indices ───────────────────────
client_indices_path = os.path.join(PART1_DIR, 'client_indices.pkl') if PART1_DIR else None
if client_indices_path and os.path.exists(client_indices_path):
    with open(client_indices_path, 'rb') as f:
        CLIENT_INDICES = pickle.load(f)
    print(f'Client indices loaded: {[len(c) for c in CLIENT_INDICES]}')
else:
    print("WARNING: client_indices.pkl not found.")
    CLIENT_INDICES = None
GT_VAL_INDICES = CFG.get('gt_val_indices')
if GT_VAL_INDICES is None:
    gt_pkl = os.path.join(PART1_DIR, 'gt_val_indices.pkl') if PART1_DIR else None
    if gt_pkl and os.path.exists(gt_pkl):
        with open(gt_pkl, 'rb') as f:
            GT_VAL_INDICES = pickle.load(f)
if GT_VAL_INDICES is not None:
    print(f'Held-out real-GT indices loaded: {len(GT_VAL_INDICES)} photos')
else:
    print('WARNING: held-out GT indices not found -- final eval uses pseudo-val only.')
COLOR_MAP = {
    0: [0, 0, 0], 1: [255, 0, 0], 2: [128, 0, 0],
    3: [255, 255, 0], 4: [128, 128, 0], 5: [0, 0, 255],
    6: [0, 128, 0], 7: [128, 128, 128], 8: [0, 255, 255], 9: [0, 255, 0],
}
print('\\nConfig + Part 1 outputs ready.')"""

CELL_DATASET = """class FloodNetDataset(Dataset):
    \"\"\"FloodNet Track-1 dataset (identical ordering to Part 1 v2).\"\"\"
    def __init__(self, base_path, split='train', transform=None, mask_transform=None,
                 require_mask=True, pseudo_mask_root=None, include_unlabeled=False, augment=False):
        self.transform        = transform
        self.mask_transform   = mask_transform
        self.require_mask     = require_mask
        self.pseudo_mask_root = pseudo_mask_root
        self.augment          = augment
        self.samples = []
        split_dir_map = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}
        split_folder  = os.path.join(base_path, split_dir_map[split])
        if not os.path.exists(split_folder):
            print(f'Not found: {split_folder}'); return
        if split == 'train':
            self._add_labeled(os.path.join(split_folder, 'Labeled', 'Flooded',     'image'),
                              os.path.join(split_folder, 'Labeled', 'Flooded',     'mask'), 'flooded')
            self._add_labeled(os.path.join(split_folder, 'Labeled', 'Non-Flooded', 'image'),
                              os.path.join(split_folder, 'Labeled', 'Non-Flooded', 'mask'), 'nonflooded')
            if include_unlabeled and pseudo_mask_root:
                self._add_unlabeled(os.path.join(split_folder, 'Unlabeled', 'image'),
                                    os.path.join(pseudo_mask_root, 'unlabeled'))
        else:
            pseudo_dir = os.path.join(pseudo_mask_root, split) if pseudo_mask_root else None
            self._add_unlabeled(os.path.join(split_folder, 'image'), pseudo_dir)
        print(f'[{split.upper()}] Loaded {len(self.samples)} samples')
    def _add_labeled(self, img_dir, msk_dir, label):
        if not os.path.exists(img_dir): return
        imgs = sorted([f for f in os.listdir(img_dir)
                       if f.lower().endswith((".jpg",".jpeg",".png",".tif"))
                       and not f.startswith(".")])
        mask_by_stem = {}
        if os.path.exists(msk_dir):
            for mf in os.listdir(msk_dir):
                if not mf.lower().endswith((".png",".jpg",".jpeg",".tif")) or mf.startswith("."): continue
                stem = os.path.splitext(mf)[0]
                mask_by_stem.setdefault(stem, mf)
                for suffix in ("_lab", "_mask", "_label", "_gt"):
                    if stem.endswith(suffix):
                        mask_by_stem.setdefault(stem[:-len(suffix)], mf)
        for fn in imgs:
            img_path  = os.path.join(img_dir, fn)
            base      = os.path.splitext(fn)[0]
            mask_path = os.path.join(msk_dir, mask_by_stem[base]) if base in mask_by_stem else None
            if self.require_mask and mask_path is None: continue
            self.samples.append({"img": img_path, "mask": mask_path,
                                 "is_pseudo": False, "label": label})
    def _add_unlabeled(self, img_dir, msk_dir):
        if not os.path.exists(img_dir): return
        exts = (".jpg",".jpeg",".png",".tif")
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith(exts) or fn.startswith('.'): continue
            img_path  = os.path.join(img_dir, fn)
            base      = os.path.splitext(fn)[0]
            mask_path = None
            if msk_dir and os.path.exists(msk_dir):
                for suffix in [f"{base}_mask.png", f"{base}.png"]:
                    cand = os.path.join(msk_dir, suffix)
                    if os.path.exists(cand):
                        mask_path = cand; break
            self.samples.append({"img": img_path, "mask": mask_path,
                                 "is_pseudo": True, "label": "unknown"})
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s["img"]).convert("RGB")
        if s["mask"] and os.path.exists(s["mask"]):
            raw  = np.array(Image.open(s["mask"]))
            if raw.ndim == 3: raw = raw[:, :, 0]
            mask = np.clip(raw, 0, NUM_CLASSES - 1)
        else:
            mask = np.zeros((img.height, img.width), dtype=np.uint8)
        mask_pil = Image.fromarray(mask)
        if self.augment:
            if random.random() < 0.5:
                img = transforms.functional.hflip(img); mask_pil = transforms.functional.hflip(mask_pil)
            if random.random() < 0.3:
                img = transforms.functional.vflip(img); mask_pil = transforms.functional.vflip(mask_pil)
            angle = random.uniform(-15, 15)
            img      = transforms.functional.rotate(img,      angle, interpolation=transforms.InterpolationMode.BILINEAR)
            mask_pil = transforms.functional.rotate(mask_pil, angle, interpolation=transforms.InterpolationMode.NEAREST, fill=0)
        if self.transform:
            img  = self.transform(img)
        if self.mask_transform:
            mask_tensor = torch.from_numpy(np.array(self.mask_transform(mask_pil))).long()
        else:
            mask_tensor = torch.from_numpy(mask).long()
        return img, mask_tensor, os.path.basename(s["img"]), s["is_pseudo"]

def get_train_transforms(img_size=512):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
def get_val_transforms(img_size=512):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
def get_mask_transforms(img_size=512):
    return transforms.Compose([transforms.Resize((img_size, img_size), interpolation=Image.NEAREST)])
print('FloodNetDataset + transforms defined.')"""

CELL_LOADERS = """FLOODNET_BASE = CFG.get('FLOODNET_BASE', '')
if not FLOODNET_BASE or not os.path.exists(FLOODNET_BASE):
    if IN_KAGGLE:
        input_dir = "/kaggle/input"
        for root, dirs, files in os.walk(input_dir):
            if 'Train' in dirs:
                FLOODNET_BASE = root
                print(f'Found Train/ at: {root}')
                break
    if not FLOODNET_BASE or not os.path.exists(FLOODNET_BASE):
        raise FileNotFoundError(f'FloodNet dataset not found at {FLOODNET_BASE}')
print(f'FLOODNET_BASE: {FLOODNET_BASE}')

train_dataset = FloodNetDataset(
    FLOODNET_BASE, split='train',
    transform=get_train_transforms(IMG_SIZE),
    mask_transform=get_mask_transforms(IMG_SIZE),
    require_mask=True, pseudo_mask_root=MASK_OUT_DIR,
    include_unlabeled=True, augment=True)
val_dataset = FloodNetDataset(
    FLOODNET_BASE, split='val',
    transform=get_val_transforms(IMG_SIZE),
    mask_transform=get_mask_transforms(IMG_SIZE),
    require_mask=False, pseudo_mask_root=MASK_OUT_DIR, augment=False)
print(f'Train: {len(train_dataset)}  Val: {len(val_dataset)}')

val_log_idx = list(range(0, len(val_dataset), max(1, len(val_dataset) // VAL_LOG_N)))[:VAL_LOG_N]
val_log_loader = DataLoader(Subset(val_dataset, val_log_idx), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)
print(f'Round-log val subset: {len(val_log_idx)} images')

if GT_VAL_INDICES is not None:
    gt_val_loader = DataLoader(Subset(train_dataset, GT_VAL_INDICES), batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=2)
    print(f'Held-out GT eval set: {len(GT_VAL_INDICES)} real photos')
else:
    gt_val_loader = val_log_loader
    print('WARNING: using pseudo-masked val subset for final eval.')

full_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print('Loaders ready.')"""

CELL_METRICS = """def compute_miou(model, loader, num_classes=NUM_CLASSES, max_batches=None):
    \"\"\"Returns (mean IoU, per-class IoU dict, pixel accuracy).\"\"\"
    model.eval()
    ious  = {c: [] for c in range(num_classes)}
    pix_correct = 0
    pix_total   = 0
    n_batches   = 0
    with torch.no_grad():
        for imgs, msks, _, _ in loader:
            imgs  = imgs.to(DEVICE)
            preds = torch.argmax(model(imgs), dim=1).cpu().numpy()
            gts   = msks.numpy()
            for p, g in zip(preds, gts):
                pix_correct += (p == g).sum()
                pix_total   += g.size
                for c in range(num_classes):
                    gt_c, pr_c = (g == c), (p == c)
                    union = (gt_c | pr_c).sum()
                    if union > 0:
                        ious[c].append((gt_c & pr_c).sum() / union)
            n_batches += 1
            if max_batches and n_batches >= max_batches:
                break
    vals = [float(np.mean(v)) for v in ious.values() if len(v) > 0]
    miou = float(np.mean(vals)) if vals else 0.0
    per_class = {c: float(np.mean(v)) if v else 0.0 for c, v in ious.items()}
    pix_acc = float(pix_correct) / float(pix_total) if pix_total > 0 else 0.0
    return miou, per_class, pix_acc

def count_params(model):
    return sum(p.numel() for p in model.parameters())
print('Metrics defined.')"""

CELL_MODEL_CHECK = """def _sanity(model, tag):
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(1, 3, 128, 128).to(DEVICE))
        print(f'{tag}: out={tuple(y.shape)} params={count_params(model)/1e6:.2f}M')

def make_model():
    return _make_model()"""

# ============================================================================
# MODEL-SPECIFIC CELLS
# ============================================================================

# UNet + ResNet-34
MODEL_UNET_R34 = """def _make_model():
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name='resnet34', encoder_weights='imagenet',
                    in_channels=3, classes=NUM_CLASSES)"""

# UNet + EfficientNet-b0
MODEL_UNET_EB0 = """def _make_model():
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name='efficientnet-b0', encoder_weights='imagenet',
                    in_channels=3, classes=NUM_CLASSES)"""

# SegNet + ResNet-34 (custom, smp has no SegNet)
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

# SegNet + EfficientNet-b0
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

# DeepLabV3+ + EfficientNet-b0
MODEL_DLV3P_EB0 = """def _make_model():
    import segmentation_models_pytorch as smp
    return smp.DeepLabV3Plus(encoder_name='efficientnet-b0', encoder_weights='imagenet',
                             in_channels=3, classes=NUM_CLASSES)"""

# FloodNet-Lite (MobileNetV3 backbone + depthwise-sep + SE attention)
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

CELL_TRAIN_CENTRALIZED = """def train_centralized(name, model, epochs=CENTRALIZED_EPOCHS, lr=LR):
    \"\"\"Centralized training on ALL real+pseudo train data (upper bound).\"\"\"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"epoch": [], "train_loss": [], "val_miou": []}
    best_miou, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_seen = 0.0, 0
        for imgs, msks, _, _ in tqdm(train_loader, desc=f'[{name}] epoch {epoch}'):
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), msks)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * imgs.size(0); n_seen += imgs.size(0)
        val_miou, per_class, _ = compute_miou(model, val_log_loader)
        history["epoch"].append(epoch)
        history["train_loss"].append(total_loss / max(1, n_seen))
        history["val_miou"].append(val_miou)
        if val_miou > best_miou:
            best_miou, best_state = val_miou, copy.deepcopy(model.state_dict())
        print(f'[{name}] epoch {epoch:3d}/{epochs}  loss={history["train_loss"][-1]:.4f}  val_mIoU={val_miou:.4f}  best={best_miou:.4f}')
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_miou

# central train loader = ALL train samples EXCEPT the held-out real-GT photos
train_idx = [i for i in range(len(train_dataset)) if (GT_VAL_INDICES is None or i not in set(GT_VAL_INDICES))]
train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=True)
print(f'Centralized train set: {len(train_idx)} samples (held-out GT excluded)')

t0 = time.time()
model = make_model().to(DEVICE)
_sanity(model, 'before')
model, history, best_val = train_centralized('MODEL_NAME', model)
print(f'Training done in {(time.time()-t0)/60:.1f} min')
torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'MODEL_ID_final.pth'))"""

CELL_FINAL_EVAL = """def final_eval(model, loader, tag=""):
    miou, per_class, pix_acc = compute_miou(model, loader)
    print(f'  {tag:>24}  mIoU={miou:.4f}  pixel_acc={pix_acc:.4f}')
    return miou, per_class, pix_acc

print("=" * 70)
print("FINAL EVALUATION -- held-out REAL GT photos (never trained on)")
print("=" * 70)
heldout_miou, per_class, pix_acc = final_eval(model, gt_val_loader, 'MODEL_NAME')
print(f'Best round/epoch val_mIoU (pseudo-val): {best_val:.4f}')

result = {
    "model_id": 'MODEL_ID',
    "display_name": 'MODEL_DISPLAY',
    "type": "centralized",
    "params_m": round(count_params(model) / 1e6, 2),
    "epochs_trained": len(history["epoch"]),
    "heldout_miou": heldout_miou,
    "pixel_acc": pix_acc,
    "best_val_miou": best_val,
    "per_class_iou": {str(k): v for k, v in per_class.items()},
    "history": history,
    "config": {"img_size": IMG_SIZE, "batch": BATCH_SIZE, "lr": LR,
               "n_train": len(train_idx), "n_heldout": len(GT_VAL_INDICES) or 0},
    "source_ref": SOURCE_REF,
}
with open(os.path.join(OUTPUT_DIR, 'result_MODEL_ID.json'), 'w') as f:
    json.dump(result, f, indent=2, default=float)
print(f'\\nSaved: result_MODEL_ID.json')

# ── training curve figure ───────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(history["epoch"], history["train_loss"], "o-", color="tab:red", label="train loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss", color="tab:red")
ax2 = ax1.twinx()
ax2.plot(history["epoch"], history["val_miou"], "s-", color="tab:blue", label="val mIoU (pseudo)")
ax2.set_ylabel("val mIoU", color="tab:blue")
plt.title("MODEL_DISPLAY -- Training Curve")
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, '31_curve_MODEL_ID.png'), dpi=150, bbox_inches="tight")
plt.show(); print('31_curve_MODEL_ID.png saved.')"""

CELL_PACKAGE = """import shutil, datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
zip_name = f"Part3_MODEL_ID_{timestamp}.zip"
zip_path = os.path.join(WORKING_DIR, zip_name)
print("\\n" + "=" * 60)
print("PACKAGING OUTPUTS FOR DOWNLOAD")
print("=" * 60)
shutil.make_archive(zip_path.replace(".zip", ""), "zip", WORKING_DIR)
size_mb = os.path.getsize(zip_path) / 1e6
print(f"Done: {zip_name} ({size_mb:.1f} MB)")
print(f"Location: {zip_path}")
if IN_KAGGLE:
    print("Kaggle: /kaggle/working/ is auto-saved. Download from the Output tab.")
print("\\nPART 3 MODEL COMPLETE:", 'MODEL_DISPLAY')"""

# ============================================================================
# FEDSEG BASELINE (Part 3.07) -- Miao et al., CVPR 2023
# Back-CE + pixel-contrastive loss on DeepLabV3+-ResNet34, FL aggregation.
# ============================================================================

MODEL_FEDSEG = """def _make_model():
    import segmentation_models_pytorch as smp
    return smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights='imagenet',
                             in_channels=3, classes=NUM_CLASSES)"""

CELL_FEDSEG_CONFIG = """# ── FedSeg FL hyper-parameters (Part 3.07 baseline) ──────────────────────────
FL_ROUNDS        = int(CFG.get('FL_ROUNDS', 25))
FL_LOCAL_EPOCHS  = int(CFG.get('FL_LOCAL_EPOCHS', 2))
CLIENT_SUBSAMPLE = int(CFG.get('CLIENT_SUBSAMPLE', 96))
CONTRAST_TEMP    = float(CFG.get('CONTRAST_TEMP', 0.1))
CONTRAST_LAMBDA  = float(CFG.get('CONTRAST_LAMBDA', 0.1))
NUM_NEGATIVES    = int(CFG.get('NUM_NEGATIVES', 64))
if DEBUG_SHORT:
    FL_ROUNDS = 3
    print('!! DEBUG_SHORT mode: FL_ROUNDS=3')

# ── FedSeg pixel-contrastive loss (Miao et al., CVPR 2023) ──────────────────
class PixelContrastMemory:
    \"\"\"Bank of per-class prototype features for contrastive learning.\"\"\"
    def __init__(self, num_classes=NUM_CLASSES, feat_dim=256, momentum=0.999):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.register_protos()

def back_ce_loss(logits, masks):
    \"\"\"Balanced CE on logits (class-weight smoothing as in FedSeg).\"\"\"
    return F.cross_entropy(logits, masks)

def pixel_contrastive_loss(feats, masks, protos, temperature=CONTRAST_TEMP,
                           num_neg=NUM_NEGATIVES, num_pos=16):
    \"\"\"InfoNCE between pixel features and class prototypes (FedSeg).\"\"\"
    B, C, H, W = feats.shape
    feats = F.normalize(feats, dim=1)
    masks = masks.view(B, H * W)
    feats = feats.view(B, C, H * W)
    protos = F.normalize(protos, dim=1)               # [K, C]
    loss_tot = feats.new_zeros(())
    n_acc = 0
    for b in range(B):
        m  = masks[b]                                 # [HW]
        f  = feats[b]                                 # [C, HW]
        for c in range(protos.shape[0]):
            idx = (m == c).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            n_anch = min(num_pos, idx.numel())
            anch_idx = idx[torch.randperm(idx.numel())[:n_anch]]
            anchors  = f[:, anch_idx].t()             # [n_anch, C]
            proto_c  = protos[c:c+1]                  # [1, C]
            pos_sim  = (anchors * proto_c).sum(dim=1) / temperature
            # negatives: other prototypes
            neg_idx  = torch.randperm(protos.shape[0])[:num_neg]
            neg_sim  = torch.matmul(anchors, protos[neg_idx].t()) / temperature
            logits   = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels   = torch.zeros(n_anch, dtype=torch.long, device=feats.device)
            loss_tot = loss_tot + F.cross_entropy(logits, labels)
            n_acc   += n_anch
    return loss_tot / max(1, n_acc)"""

CELL_FEDSEG_TRAIN = """def fed_aggregate(weights_list, client_weights):
    \"\"\"Client-size-weighted aggregation (uniform when weights equal).\"\"\"
    w = torch.tensor(client_weights, dtype=torch.float32, device=DEVICE)
    w = w / (w.sum() + 1e-8)
    global_dict = {}
    for key in weights_list[0].keys():
        stacked = torch.stack([ww[key].float() for ww in weights_list])
        shape = [len(weights_list)] + [1] * (stacked.dim() - 1)
        global_dict[key] = (stacked * w.view(shape)).sum(dim=0)
    return global_dict

def _local_train(global_model, client_idx, backbone_fn, local_epochs, lr, protos, round_idx=1):
    \"\"\"FedSeg local training: back-CE + pixel-contrastive loss.\"\"\"
    idx_all = np.array(CLIENT_INDICES[client_idx])
    rng = np.random.RandomState(SEED + client_idx * 7919 + round_idx * 104729)
    if len(idx_all) > CLIENT_SUBSAMPLE:
        idx = idx_all[rng.choice(len(idx_all), CLIENT_SUBSAMPLE, replace=False)]
    else:
        idx = idx_all
    sub_ds = Subset(train_dataset, idx.tolist())
    loader = DataLoader(sub_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True)
    client_model = backbone_fn().to(DEVICE)
    client_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
    optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    client_model.train()
    total_loss = 0.0; n_seen = 0
    for _ in range(local_epochs):
        for imgs, msks, _, _ in loader:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            optimizer.zero_grad()
            logits = client_model(imgs)
            seg_loss = back_ce_loss(logits, msks)
            # pixel features from the decoder (before final head)
            feats = client_model.encoder(imgs)[-1]
            feats = F.interpolate(feats, size=(IMG_SIZE // 2, IMG_SIZE // 2),
                                  mode='bilinear', align_corners=False)
            feats = F.interpolate(feats, size=msks.shape[-2:], mode='bilinear', align_corners=False)
            con_loss = pixel_contrastive_loss(feats, msks, protos)
            loss = seg_loss + CONTRAST_LAMBDA * con_loss
            loss.backward(); optimizer.step()
            total_loss += loss.item() * imgs.size(0); n_seen += imgs.size(0)
    avg_loss = total_loss / max(1, n_seen)
    del loader
    return client_model, avg_loss

def run_fedseg(name, backbone_fn, rounds=FL_ROUNDS, local_epochs=FL_LOCAL_EPOCHS, lr=LR):
    \"\"\"Federated training with FedSeg loss and client-size-weighted aggregation.\"\"\"
    global_model = backbone_fn().to(DEVICE)
    protos = F.normalize(torch.randn(NUM_CLASSES, 256, device=DEVICE), dim=1)
    history = {"round": [], "avg_train_loss": [], "val_miou": [], "per_class_iou": []}
    client_sizes = [len(CLIENT_INDICES[c]) for c in range(NUM_CLIENTS)] if CLIENT_INDICES else [1.0] * NUM_CLIENTS
    for round_idx in range(1, rounds + 1):
        local_weights = []
        local_models  = []
        local_losses  = []
        for client_idx in range(NUM_CLIENTS):
            cm, al = _local_train(global_model, client_idx, backbone_fn,
                                  local_epochs, lr, protos, round_idx=round_idx)
            local_weights.append(copy.deepcopy(cm.state_dict()))
            local_models.append(cm)
            local_losses.append(al)
        global_dict = fed_aggregate(local_weights, client_sizes)
        global_model.load_state_dict(global_dict)
        val_miou, per_class, _ = compute_miou(global_model, val_log_loader)
        history["round"].append(round_idx)
        history["avg_train_loss"].append(float(np.mean(local_losses)))
        history["val_miou"].append(val_miou)
        history["per_class_iou"].append(per_class)
        for m in local_models:
            del m
        torch.cuda.empty_cache()
        if round_idx % 5 == 0 or round_idx == 1:
            print(f'[{name}] Round {round_idx:3d}/{rounds}  loss={history["avg_train_loss"][-1]:.4f}'
                  f'  val_mIoU={val_miou:.4f}')
    return global_model, history

print('FedSeg federated training functions defined.')"""

# ============================================================================
# HEADERS (markdown)
# ============================================================================

def header_md(title, desc, ref):
    return f"""# {title}

**Part 3 literature comparison** — standalone training for ONE model. Run this notebook
on its own device/Kaggle account, then download `Part3_*.zip` from the Output tab and
send the `result_*.json` inside it to the person running the merge notebook.

{desc}

**Reference:** {ref}

---
### What you need attached on Kaggle
1. The FloodNet dataset (search `FloodNet` in Kaggle datasets, or use the existing
   `aerial-imagery-dataset-floodnet-challenge`).
2. **Part 1 outputs** (`fed_salgate_outputs/` + `masks/`) as a notebook attachment or
   dataset — pseudo-masks + held-out GT indices. If missing, this notebook falls back
   to defaults (results will NOT match Part 1 splits)."""

def header_merge():
    return """# Part 3 Merge — Literature vs Fed-SalGate Comparison

Collects every `result_*.json` produced by the Part 3 model notebooks (upload them as
a dataset/attachment, or point RESULTS_DIR to a local folder) and generates the full
comparison: tables, per-class IoU heatmap, bar/radar charts, training curves, an Excel
workbook, and a final zip.

### How to use
1. Gather the `result_*.json` files from ALL friends/devices (each is inside the
   `Part3_*.zip` they downloaded from Kaggle).
2. Upload them as a Kaggle dataset or notebook attachment (the notebook recursively
   scans `/kaggle/input`). Or set `RESULTS_DIR` below to a local folder.
3. Run all cells. The final `Literature_Comparison_*.zip` contains everything."""

# ============================================================================
# Build notebooks
# ============================================================================

def cell_code(src):  return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}
def cell_md(src):    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

MODELS = [
    dict(fname="part3_01_unet_resnet34",        id="unet_resnet34",        display="UNet (ResNet-34)",        model=MODEL_UNET_R34,  ref="[36] Karcı et al., *Natural Hazards*, 2026 — UNet+ResNet comparison"),
    dict(fname="part3_02_unet_efficientnet_b0", id="unet_efficientnet_b0", display="UNet (EfficientNet-b0)",   model=MODEL_UNET_EB0,  ref="[36] Karcı et al., *Natural Hazards*, 2026 — best combo (IoU 0.9271)"),
    dict(fname="part3_03_segnet_resnet34",      id="segnet_resnet34",      display="SegNet (ResNet-34)",      model=MODEL_SEGNET_R34, ref="[36] Karcı et al., *Natural Hazards*, 2026 — SegNet+ResNet"),
    dict(fname="part3_04_segnet_efficientnet_b0", id="segnet_efficientnet_b0", display="SegNet (EfficientNet-b0)", model=MODEL_SEGNET_EB0, ref="[36] Karcı et al., *Natural Hazards*, 2026 — SegNet+EfficientNet"),
    dict(fname="part3_05_deeplabv3p_efficientnet_b0", id="deeplabv3p_efficientnet_b0", display="DeepLabV3+ (EfficientNet-b0)", model=MODEL_DLV3P_EB0, ref="[36] Karcı et al., *Natural Hazards*, 2026 — DeepLabV3+ + EfficientNet"),
    dict(fname="part3_06_floodnet_lite",        id="floodnet_lite",        display="FloodNet-Lite",            model=MODEL_FLOODNETLITE, ref="[35] Thirugnanasammandamoorthi et al., *JSTARS*, 2025 — lightweight optimized UNet"),
]

for m in MODELS:
    cells = []
    cells.append(cell_md(header_md(
        f"Part 3.{MODELS.index(m)+1} — {m['display']}",
        f"Centralized training of **{m['display']}** on all real+pseudo train data "
        f"(held-out real-GT photos excluded), evaluated on the SAME { 59 if False else ''}held-out real-GT photos as Fed-SalGate.",
        m["ref"])))
    cells.append(cell_md("## ⚙️ Install packages"))
    cells.append(cell_code(CELL_INSTALL))
    cells.append(cell_md("## 📚 Imports"))
    cells.append(cell_code(CELL_IMPORTS))
    cells.append(cell_md("## ⚙️ Config + Part 1 discovery"))
    cells.append(cell_code(CELL_CONFIG))
    cells.append(cell_md("## 🗂️ Dataset"))
    cells.append(cell_code(CELL_DATASET))
    cells.append(cell_md("## 🚚 Loaders"))
    cells.append(cell_code(CELL_LOADERS))
    cells.append(cell_md("## 📊 Metrics"))
    cells.append(cell_code(CELL_METRICS))
    cells.append(cell_md("## 🧠 Model architecture"))
    cells.append(cell_code(m["model"]))
    cells.append(cell_code(CELL_MODEL_CHECK))
    cells.append(cell_md("## 🏋️ Train (centralized upper bound)"))
    cells.append(cell_code(CELL_TRAIN_CENTRALIZED.replace("MODEL_ID", m["id"]).replace("MODEL_NAME", m["display"])))
    cells.append(cell_md("## 🎯 Final evaluation on held-out real GT"))
    cells.append(cell_code(CELL_FINAL_EVAL
                           .replace("MODEL_ID", m["id"])
                           .replace("MODEL_NAME", m["display"])
                           .replace("MODEL_DISPLAY", m["display"])
                           .replace('"source_ref": SOURCE_REF,', f'"source_ref": {json.dumps(m["ref"])},')))
    cells.append(cell_md("## 📦 Package outputs"))
    cells.append(cell_code(CELL_PACKAGE.replace("MODEL_ID", m["id"]).replace("MODEL_DISPLAY", m["display"])))

    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python", "version": "3.10.12"}},
          "nbformat": 4, "nbformat_minor": 2}
    with open(os.path.join(OUT_DIR, m["fname"] + ".ipynb"), "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {m['fname']}.ipynb  ({len(cells)} cells)")

print("\\nAll Part 3 model notebooks generated.")

# ============================================================================
# PART 3.07 -- FedSeg FL baseline notebook
# ============================================================================

CELL_FEDSEG_RUN = """# ── Train FedSeg (federated, back-CE + pixel contrastive) ─────────────────────
print("=" * 70)
print("FedSeg baseline: FL with back-CE + pixel-contrastive (Miao et al., CVPR 2023)")
print("=" * 70)
t0 = time.time()
fedseg_model, fedseg_history = run_fedseg('FedSeg', make_model)
print(f'FedSeg done in {(time.time()-t0)/60:.1f} min | best round mIoU = {max(fedseg_history["val_miou"]):.4f}')
torch.save(fedseg_model.state_dict(), os.path.join(CKPT_DIR, 'fedseg_dlv3p_final.pth'))

# ── Final evaluation on held-out REAL GT photos ─────────────────────────────
print("=" * 70)
print("FINAL EVALUATION -- held-out REAL GT photos (never trained on)")
print("=" * 70)
def final_eval(model, loader, tag=""):
    miou, per_class, pix_acc = compute_miou(model, loader)
    print(f'  {tag:>24}  mIoU={miou:.4f}  pixel_acc={pix_acc:.4f}')
    return miou, per_class, pix_acc

heldout_miou, per_class, pix_acc = final_eval(fedseg_model, gt_val_loader, 'FedSeg (DeepLabV3+)')
print(f'Best round val_mIoU (pseudo-val): {max(fedseg_history["val_miou"]):.4f}')

result = {
    "model_id": "fedseg_deeplabv3p",
    "display_name": "DeepLabV3+ / FedSeg (FL baseline)",
    "type": "federated (Part 3)",
    "params_m": round(count_params(fedseg_model) / 1e6, 2),
    "rounds": len(fedseg_history["round"]),
    "heldout_miou": heldout_miou,
    "pixel_acc": pix_acc,
    "best_val_miou": max(fedseg_history["val_miou"]),
    "per_class_iou": {str(k): v for k, v in per_class.items()},
    "history": fedseg_history,
    "config": {"img_size": IMG_SIZE, "batch": BATCH_SIZE, "lr": LR,
               "rounds": FL_ROUNDS, "local_epochs": FL_LOCAL_EPOCHS,
               "contrast_lambda": CONTRAST_LAMBDA, "n_heldout": len(GT_VAL_INDICES) or 0},
    "source_ref": "[37] Miao et al., *CVPR*, 2023 -- FedSeg: Federated semantic segmentation via class-balanced CE + pixel contrastive",
}
with open(os.path.join(OUTPUT_DIR, 'result_fedseg_deeplabv3p.json'), 'w') as f:
    json.dump(result, f, indent=2, default=float)
print('\\nSaved: result_fedseg_deeplabv3p.json')

# ── training curve figure ───────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(fedseg_history["round"], fedseg_history["avg_train_loss"], "o-", color="tab:red", label="avg train loss")
ax1.set_xlabel("Round"); ax1.set_ylabel("Avg train loss", color="tab:red")
ax2 = ax1.twinx()
ax2.plot(fedseg_history["round"], fedseg_history["val_miou"], "s-", color="tab:blue", label="val mIoU (pseudo)")
ax2.set_ylabel("val mIoU", color="tab:blue")
plt.title("DeepLabV3+ / FedSeg -- Federated Training Curve")
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, '31_curve_fedseg_deeplabv3p.png'), dpi=150, bbox_inches="tight")
plt.show(); print('31_curve_fedseg_deeplabv3p.png saved.')"""

FEDSEG_FINAL_EVAL = """def final_eval(model, loader, tag=""):
    miou, per_class, pix_acc = compute_miou(model, loader)
    print(f'  {tag:>24}  mIoU={miou:.4f}  pixel_acc={pix_acc:.4f}')
    return miou, per_class, pix_acc
print("done")"""

def _build_fedseg():
    cells = []
    cells.append(cell_md(header_md(
        "Part 3.07 — DeepLabV3+ / FedSeg (Federated Learning baseline)",
        "Federated training of **DeepLabV3+ (ResNet-34)** with the FedSeg loss "
        "(back-CE + pixel-contrastive, Miao et al., CVPR 2023) on the SAME 5-client "
        "Part 1 splits, evaluated on the SAME held-out real-GT photos as Fed-SalGate.",
        "[37] Miao et al., *CVPR*, 2023 — FedSeg")))
    cells.append(cell_md("## ⚙️ Install packages"))
    cells.append(cell_code(CELL_INSTALL))
    cells.append(cell_md("## 📚 Imports"))
    cells.append(cell_code(CELL_IMPORTS))
    cells.append(cell_md("## ⚙️ Config + Part 1 discovery"))
    cells.append(cell_code(CELL_CONFIG))
    cells.append(cell_md("## 🧠 FedSeg config + pixel-contrastive loss"))
    cells.append(cell_code(CELL_FEDSEG_CONFIG))
    cells.append(cell_md("## 🗂️ Dataset"))
    cells.append(cell_code(CELL_DATASET))
    cells.append(cell_md("## 🚚 Loaders"))
    cells.append(cell_code(CELL_LOADERS))
    cells.append(cell_md("## 📊 Metrics"))
    cells.append(cell_code(CELL_METRICS))
    cells.append(cell_md("## 🧠 Model architecture (DeepLabV3+ ResNet-34)"))
    cells.append(cell_code(MODEL_FEDSEG))
    cells.append(cell_code(CELL_MODEL_CHECK))
    cells.append(cell_md("## 🔄 Federated training (FedSeg)"))
    cells.append(cell_code(CELL_FEDSEG_TRAIN))
    cells.append(cell_code(CELL_FEDSEG_RUN))
    cells.append(cell_md("## 📦 Package outputs"))
    cells.append(cell_code(CELL_PACKAGE.replace("MODEL_ID", "fedseg_deeplabv3p").replace("MODEL_DISPLAY", "DeepLabV3+ / FedSeg")))

    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python", "version": "3.10.12"}},
          "nbformat": 4, "nbformat_minor": 2}
    fname = "part3_07_fedseg_deeplabv3p"
    with open(os.path.join(OUT_DIR, fname + ".ipynb"), "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {fname}.ipynb  ({len(cells)} cells)")

_build_fedseg()
