#!/usr/bin/env python3
"""
Build script for Part 3 -- Merge notebook.

Generates part3_99_merge_results.ipynb which reads all result_*.json produced by the
per-model notebooks (or the Part 2 summary) and produces the full comparison package.
"""
import json, os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")

def cell_code(src):  return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}
def cell_md(src):    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

HEADER = """# Part 3 Merge — Literature Review Models vs Fed-SalGate

Collects every `result_*.json` produced by the Part 3 model notebooks (plus the optional
Part 2 `part2_v2_summary.json`) and generates the full comparison: summary tables,
per-class IoU heatmap, bar/radar charts, training-curve overlay, an Excel workbook, and
a final zip with everything.

### How to use
1. Each friend/device that ran a Part 3 notebook returns its `result_<model>.json`
   (it is inside their `Part3_*.zip`). Place all of them in one place.
2. Either upload that folder as a Kaggle dataset / notebook attachment (the notebook
   recursively scans `/kaggle/input`) or set `RESULTS_DIR` below to a local path.
3. Optionally also provide the Part 2 summary JSON so FedAvg / FedProx / Fed-SalGate
   appear in the same table. It is recursively discovered as `part2_v2_summary.json`.
4. Run all cells. The final `Literature_Comparison_*.zip` contains everything."""

IMPORTS = """import os, json, glob, datetime, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    import seaborn as sns
except Exception:
    sns = None
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "figure.figsize": (14, 6)})
try:
    import openpyxl
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openpyxl"])
    import openpyxl
print("Imports complete.")"""

CONFIG = """WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "part3_merged")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")
EXCEL_DIR  = os.path.join(OUTPUT_DIR, "excel")
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
for d in [OUTPUT_DIR, PLOT_DIR, EXCEL_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Where the result_*.json files live ──────────────────────────────────────
RESULTS_DIR = os.environ.get("PART3_RESULTS_DIR", None)  # set to a local path if not on Kaggle
if RESULTS_DIR and os.path.isdir(RESULTS_DIR):
    print(f"Using explicit RESULTS_DIR: {RESULTS_DIR}")
else:
    RESULTS_DIR = None

def discover_json(fname, root=None):
    root = root or RESULTS_DIR
    if root:
        for p in glob.glob(os.path.join(root, "**", fname), recursive=True):
            return p
    if os.path.isdir("/kaggle/input"):
        for root_dir, dirs, files in os.walk("/kaggle/input"):
            if fname in files:
                return os.path.join(root_dir, fname)
    return None

# ── Collect all Part 3 model results ────────────────────────────────────────
result_files = []
if RESULTS_DIR:
    result_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "**", "result_*.json"), recursive=True))
else:
    seen = set()
    for root_dir, dirs, files in os.walk("/kaggle/input"):
        for fn in files:
            if fn.startswith("result_") and fn.endswith(".json") and fn not in seen:
                result_files.append(os.path.join(root_dir, fn))
                seen.add(fn)
if not result_files:
    # also look in the working dir (e.g. a local run)
    result_files = sorted(glob.glob(os.path.join(WORKING_DIR, "**", "result_*.json"), recursive=True))
print(f"Found {len(result_files)} Part 3 result JSONs")
for p in result_files:
    print("  ", os.path.basename(os.path.dirname(p)) or p, "/", os.path.basename(p))

# ── Part 2 summary (FedAvg / FedProx / Fed-SalGate) ─────────────────────────
part2_path = discover_json("part2_v2_summary.json")
part2 = {}
if part2_path:
    with open(part2_path) as f:
        part2 = json.load(f)
    print(f"\\nPart 2 summary found: {part2_path}")
    print("  models:", list(part2.get("models", {}).keys()))
else:
    print("\\nPart 2 summary NOT found (FedAvg/FedProx/Fed-SalGate will be absent).")"""

LOAD = """CLASS_NAMES = ['Background', 'Building-Flooded', 'Building-NonFlooded',
                'Road-Flooded', 'Road-NonFlooded', 'Water',
                'Tree', 'Vehicle', 'Pool', 'Grass']

def normalize_per_class(pc):
    out = {}
    if isinstance(pc, dict):
        for k, v in pc.items():
            idx = k if isinstance(k, int) else int(float(k))
            out[idx] = float(v)
    return out

rows = []
per_class = {}   # model -> {class_idx: iou}
histories = {}   # model -> history dict
for fp in result_files:
    with open(fp) as f:
        r = json.load(f)
    mid = r["model_id"]
    pc = normalize_per_class(r.get("per_class_iou", {}))
    per_class[mid] = pc
    histories[mid] = r.get("history", {})
    rows.append({
        "model": r.get("display_name", mid),
        "model_id": mid,
        "type": r.get("type", "centralized"),
        "params_m": r.get("params_m", None),
        "heldout_miou": r.get("heldout_miou"),
        "pixel_acc": r.get("pixel_acc"),
        "best_val_miou": r.get("best_val_miou"),
        "epochs_trained": r.get("epochs_trained", r.get("rounds", None)),
        "source_ref": r.get("source_ref", ""),
    })

# ── Add Part 2 models (FedAvg / FedProx / Fed-SalGate) ─────────────────────
P2_ORDER = [("DeepLabV3+ / FedAvg", "fedavg"),
            ("DeepLabV3+ / FedProx", "fedprox"),
            ("DeepLabV3+ / Fed-SalGate", "fedsalgate"),
            ("LiteUNet / FedAvg", "lite_fedavg"),
            ("LiteUNet / Fed-SalGate", "lite_fedsalgate")]
for display, key in P2_ORDER:
    m = part2.get("models", {}).get(key)
    if not m:
        continue
    pc = normalize_per_class(m.get("per_class_iou", {}))
    per_class[key] = pc
    rows.append({
        "model": display,
        "model_id": key,
        "type": "federated (Part 2)",
        "params_m": None,
        "heldout_miou": m.get("heldout_miou"),
        "pixel_acc": m.get("pixel_acc"),
        "best_val_miou": m.get("best_round_miou"),
        "epochs_trained": None,
        "source_ref": "Part 2 (this work)",
    })

df = pd.DataFrame(rows)
df["heldout_miou"] = pd.to_numeric(df["heldout_miou"], errors="coerce")
df["pixel_acc"]    = pd.to_numeric(df["pixel_acc"], errors="coerce")
df["best_val_miou"]= pd.to_numeric(df["best_val_miou"], errors="coerce")
df = df.sort_values("heldout_miou", ascending=False).reset_index(drop=True)
print("\\n================= COMPARISON TABLE =================")
print(df.to_string(index=False))
df.to_csv(os.path.join(OUTPUT_DIR, "comparison_table.csv"), index=False)
print("\\nSaved: comparison_table.csv")"""

PLOT_BAR = """def _save(fig, name):
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {name}")

# 1) mIoU bar chart -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))
colors = ["#2ca02c" if (r["type"] == "centralized") else
          ("#ff7f0e" if r["type"] == "federated" else "#1f77b4")
          for _, r in df.iterrows()]
bars = ax.barh(df["model"], df["heldout_miou"], color=colors)
ax.invert_yaxis()
for bar, v in zip(bars, df["heldout_miou"]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{v:.4f}", va="center", fontsize=9)
ax.set_xlabel("Held-out real-GT mIoU")
ax.set_title("Part 3 — Literature Models vs Fed-SalGate (held-out real GT mIoU)")
legend = [mpatches.Patch(color="#2ca02c", label="Centralized (literature)"),
          mpatches.Patch(color="#ff7f0e", label="Federated"),
          mpatches.Patch(color="#1f77b4", label="Federated (ours)")]
ax.legend(handles=legend, loc="lower right")
_save(fig, "40_miou_bar.png")

# 2) pixel accuracy -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))
colors = ["#2ca02c" if (r["type"] == "centralized") else
          ("#ff7f0e" if r["type"] == "federated" else "#1f77b4")
          for _, r in df.iterrows()]
ax.barh(df["model"], df["pixel_acc"], color=colors)
ax.invert_yaxis()
ax.set_xlabel("Pixel accuracy (held-out real GT)")
ax.set_title("Pixel Accuracy Comparison")
_save(fig, "41_pixel_acc_bar.png")"""

PLOT_PERCLASS = """# 3) per-class IoU heatmap ---------------------------------------------------
class_ids = sorted({ci for pc in per_class.values() for ci in pc.keys()})
cls_labels = [f"{c} {CLASS_NAMES[c]}" for c in class_ids]
mat = np.full((len(df), len(class_ids)), np.nan)
for ri, (_, r) in enumerate(df.iterrows()):
    pc = per_class.get(r["model_id"], {})
    for ci, v in pc.items():
        if ci in class_ids:
            mat[ri, class_ids.index(ci)] = v
fig, ax = plt.subplots(figsize=(14, max(6, len(df) * 0.55)))
im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
ax.set_xticks(range(len(class_ids))); ax.set_xticklabels(cls_labels, rotation=45, ha="right")
ax.set_yticks(range(len(df))); ax.set_yticklabels(df["model"])
for i in range(len(df)):
    for j in range(len(class_ids)):
        v = mat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if v < 0.6 else "white")
fig.colorbar(im, ax=ax, label="IoU")
ax.set_title("Per-Class IoU Heatmap (held-out real GT)")
_save(fig, "42_per_class_iou_heatmap.png")"""

PLOT_RADAR = """# 4) radar / spider chart -----------------------------------------------------
metrics = ["heldout_miou", "pixel_acc", "best_val_miou"]
short = {"heldout_miou": "Held-out mIoU", "pixel_acc": "Pixel Acc", "best_val_miou": "Best val mIoU"}
sub = df.dropna(subset=metrics).head(8)
if len(sub) >= 3:
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for _, r in sub.iterrows():
        vals = [r[m] for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", label=r["model"])
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels([short[m] for m in metrics])
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.05), fontsize=8)
    ax.set_title("Multi-Metric Radar", pad=30)
    _save(fig, "43_radar.png")"""

PLOT_CURVES = """# 5) training curves overlay ------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))
for mid, h in histories.items():
    if not h: continue
    key = "epoch" if "epoch" in h else ("round" if "round" in h else None)
    if key is None: continue
    x  = h[key]
    ym = h.get("val_miou")
    if ym is None: continue
    disp = df.loc[df["model_id"] == mid, "model"].values
    label = disp[0] if len(disp) else mid
    ax.plot(x, ym, "o-", markersize=3, label=label)
ax.set_xlabel("Epoch / Round")
ax.set_ylabel("val mIoU (pseudo-masked subset)")
ax.set_title("Training Curves Overlay (val mIoU)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
_save(fig, "44_training_curves.png")"""

EXCEL = """def write_excel():
    wb = openpyxl.Workbook()
    ws = wb.active; ws.title = "Summary"
    ws.append(["Model", "Type", "Params (M)", "Held-out mIoU", "Pixel Acc", "Best val mIoU", "Epochs/Rounds", "Source"])
    for _, r in df.iterrows():
        ws.append([r["model"], r["type"], r["params_m"], r["heldout_miou"],
                   r["pixel_acc"], r["best_val_miou"], r["epochs_trained"], r["source_ref"]])
    ws2 = wb.create_sheet("PerClassIoU")
    ws2.append(["Model"] + cls_labels)
    for _, r in df.iterrows():
        row = [r["model"]]
        pc = per_class.get(r["model_id"], {})
        row += [round(pc.get(c, float("nan")), 4) for c in class_ids]
        ws2.append(row)
    ws3 = wb.create_sheet("Config")
    ws3.append(["field", "value", "note"])
    ws3.append(["Dataset", "FloodNet Track 1", "10 classes"])
    ws3.append(["IMG_SIZE", "512", "identical to Part 2"])
    ws3.append(["Held-out GT photos", "real", "never trained on"])
    ws3.append(["Metric", "mIoU / pixel acc / per-class IoU", "identical to Part 2"])
    for k, v in part2.items():
        if k in ("models",): continue
        ws3.append([k, v, "Part 2"])
    path = os.path.join(EXCEL_DIR, "literature_comparison.xlsx")
    wb.save(path)
    print("Saved:", path)
    return path
excel_path = write_excel()"""

PACKAGE = """# ── Final package -----------------------------------------------------------
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
zip_name = f"Literature_Comparison_{timestamp}.zip"
zip_path = os.path.join(WORKING_DIR, zip_name)
print("\\n" + "=" * 60)
print("PACKAGING FINAL COMPARISON OUTPUTS")
print("=" * 60)
for desc, path in [("Plots", PLOT_DIR), ("Excel", EXCEL_DIR), ("CSV/JSON", OUTPUT_DIR)]:
    n = len([f for f in os.listdir(path) if not f.startswith(".")])
    print(f"  [OK] {desc}: {n} files")
shutil.make_archive(zip_path.replace(".zip", ""), "zip", OUTPUT_DIR)
size_mb = os.path.getsize(zip_path) / 1e6
print(f"\\nCreated: {zip_name} ({size_mb:.1f} MB)")
print(f"Location: {zip_path}")
if os.path.isdir("/kaggle/working"):
    print("Kaggle: /kaggle/working/ is auto-saved. Download from the Output tab.")
print("\\n" + "=" * 60)
print("PART 3 MERGE COMPLETE")
print("=" * 60)"""

cells = [
    cell_md(HEADER),
    cell_md("## 📚 Imports"),
    cell_code(IMPORTS),
    cell_md("## ⚙️ Config + discovery"),
    cell_code(CONFIG),
    cell_md("## 📥 Load results"),
    cell_code(LOAD),
    cell_md("## 📊 Bar charts"),
    cell_code(PLOT_BAR),
    cell_md("## 🌡️ Per-class heatmap"),
    cell_code(PLOT_PERCLASS),
    cell_md("## 🕸️ Radar"),
    cell_code(PLOT_RADAR),
    cell_md("## 📈 Training curves"),
    cell_code(PLOT_CURVES),
    cell_md("## 📗 Excel export"),
    cell_code(EXCEL),
    cell_md("## 📦 Final zip"),
    cell_code(PACKAGE),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python", "version": "3.10.12"}},
      "nbformat": 4, "nbformat_minor": 2}
with open(os.path.join(OUT_DIR, "part3_99_merge_results.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote part3_99_merge_results.ipynb ({len(cells)} cells)")
