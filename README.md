# Fed-SalGate

**Fed-SalGate: Federated Saliency-Consensus Gating for Robust and Rare-Class-Aware UAV Flood Damage Segmentation**

Federated semantic segmentation of post-flood UAV imagery. Instead of plain FedAvg-style
uniform averaging, the server weights each client by how much its class-conditional saliency
(where the model attends for flood-related classes) agrees with the other clients' saliency,
via soft-IoU consensus. This makes the shared model more robust to a faulty/corrupted client
and better at preserving rare, safety-critical damage classes (flooded roads, vehicles, pools)
under non-IID data.

## Data

This code expects the FloodNet Track 1 dataset plus a fixed-mask mirror used to sharpen a
subset of class boundaries. Download both from Kaggle:

- **FloodNet Track 1**: https://www.kaggle.com/datasets/aletbm/aerial-imagery-dataset-floodnet-challenge
- **Fixed-mask mirror**: https://www.kaggle.com/datasets/catashiro31/floodnet-dataset-for-segmentation-fixed-mask

The notebooks auto-discover the dataset location rather than using a hardcoded path: they
search `/kaggle/input` (when running on Kaggle) and your home directory for a folder whose
name matches expected keywords (e.g. containing `FloodNet` and `Track 1`, or `class_mapping.csv`).
To run locally, extract the downloaded datasets into your home directory (or anywhere under
it) and the discovery logic will find them; on Kaggle, just attach both datasets to the
notebook as inputs.

## Install

```bash
pip install -r requirements.txt
```

Developed with Python 3.11+, CUDA-capable GPU recommended (trained on Kaggle T4 / RunPod RTX 4090).

## Which notebook runs what

| Notebook | Role |
|---|---|
| `part1/part1_image_processing.ipynb` | Data processing: class-map construction, oversampling, Dirichlet(α=0.5) non-IID partition across 3 clients, held-out ground-truth split. Run this first — later parts load its outputs. |
| `part2/step2.ipynb` | Centralized training of the full FSCG-UNet architecture (ConvNeXt-Tiny + ViT + ASPP + attention-gated decoder + SSCG saliency head). Produces the centralized accuracy-ceiling result. |
| `part3/step3-3_final.ipynb` | **Main federated experiment.** Trains and compares FedAvg, FedProx, and Fed-SalGate (on Lite-UNet and FSCG-UNet backbones), including the corrupted-client robustness test and the multi-seed study. This is the reference run cited throughout the paper. |
| `graph_attention_addition/step3_graphattn_REAL.ipynb`, `step3_graphattn_run2_REAL.ipynb` | Two independent runs isolating the effect of adding a graph-attention branch (GAT + Chebyshev convolution) at the encoder bottleneck, centralized and federated. |
| `part3_literature_comparison/generated/part3_01..07_*.ipynb` | Seven reproduced literature baselines (UNet/SegNet/DeepLabV3+ × ResNet-34/EfficientNet-B0, FloodNet-Lite, and the federated FedSeg baseline), each run standalone and emitting its own `result_*.json`. |
| `part3_literature_comparison/generated/part3_99_merge_results.ipynb` | Collects all `result_*.json` files into the final comparison table. |
| `part3_literature_comparison/build_*.py` | Scripts used to generate the `part3_01..07,99` notebooks programmatically from a shared template. |

## Notes

- Notebooks were developed and run on Kaggle (T4 GPU) and RunPod (RTX 4090); some auto-detect
  the Kaggle environment via `IN_KAGGLE = os.path.isdir("/kaggle/input")` and adjust paths
  accordingly.
- `FSCG_*` environment variables (e.g. `FSCG_GATE_BETA`, `FSCG_GATE_TAU`, `FSCG_FL_ROUNDS`)
  override the default hyperparameters in Part 3 without editing the notebook.
- Trained model weights and the raw FloodNet imagery are not included in this repository.
