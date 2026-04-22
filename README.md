# ReSUNet Segmentation — Self-supervised Learning via Max-tree Representation

> Reproduction and extension of  
> **Tang et al., "Self-supervised Learning Based on Max-tree Representation for Medical Image Segmentation"**, IJCNN 2022.

---

## Overview

Medical image segmentation requires large amounts of expensive annotated data.  
This project reproduces the SSL strategy from the paper above: instead of learning from pixel-level labels, the network is first trained on a **self-supervised pretext task** — reconstructing a morphological transformation of each image called the **Max-tree area-ratio representation**.

The structural features learned during the pretext task transfer directly to the segmentation task, significantly reducing the annotation burden.

---

## Method

### What is the Max-tree?

A **component max-tree** is a hierarchical representation of an image's level sets.  
Each pixel is mapped to a node in a tree. The **area-ratio attribute** of each node encodes how much of the image is brighter than or equal to that pixel, relative to its parent.  
This captures structural information — organ boundaries, tissue contrast — without any labels.

```
Original image  →  Build max-tree  →  Compute area-ratio  →  Restitute as image
     X                  Tree T                Tree T'                   X'
```

The CNN learns to reconstruct X' from X (pretext task), then its weights are transferred to the segmentation network.

### Two-stage pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1 — SSL Pretext Task  (unlabeled LIDC-IDRI CT scans)         │
│                                                                     │
│   Input X ──► ReSUNet ──► X̂  ◄── MSE loss ──► Target X' (max-tree) │
│                                                                     │
└────────────────────────┬────────────────────────────────────────────┘
                         │  Transfer encoder weights
┌────────────────────────▼────────────────────────────────────────────┐
│  Stage 2 — Segmentation Fine-tuning  (labeled LiTS 2017)            │
│                                                                     │
│   CT scan ──► ReSUNet ──► Segmentation mask  ◄── BCE + Dice loss    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Results

Results on **LiTS 2017** liver segmentation (validation set, 15 cases):

| Method | Dice (%) ↑ | IoU (%) ↑ |
|---|---|---|
| ResUNet — from scratch (baseline) | 90.26 | 77.54 |
| ResUNet + Autoencoder SSL | 91.35 | 79.25 |
| ResUNet + Context Restoration | 93.29 | 81.83 |
| ResUNet + Models Genesis | 94.21 | 82.62 |
| **ResUNet + Max-tree SSL (ours)** | **94.27** | **82.92** |

### Limited data regime

| Training data | Scratch Dice | Max-tree SSL Dice | Improvement |
|---|---|---|---|
| 10% | 82.47 | 92.75 | **+10.28** |
| 20% | 88.87 | 93.35 | +4.48 |
| 50% | 90.18 | 94.02 | +3.84 |
| 100% | 90.26 | 94.27 | +4.01 |

> The SSL strategy is most impactful when labeled data is scarce — which is the typical clinical scenario.

---

## Repository structure

```
ReSUNet_Segmentation/
│
├── src/
│   ├── __init__.py
│   ├── model_resunet.py     # ReSUNet architecture (residual encoder-decoder)
│   ├── max_tree.py          # Max-tree transformation (pretext task target)
│   ├── data_loader.py       # tf.data pipelines for NIfTI / DICOM data
│   ├── metrics.py           # Dice, IoU, BCE+Dice loss (Keras + NumPy)
│   ├── train.py             # Two-stage training script
│   ├── inference.py         # Inference on single volume or folder
│   └── utils.py             # Visualisation utilities
│
├── notebooks/
│   └── ReSUnet_segmentation.ipynb   # End-to-end Google Colab notebook
│
├── images/examples/         # Sample inputs and predicted masks
├── reports/                 # Training logs and metrics
├── data/                    # Text file lists (paths to NIfTI files)
├── requirements.txt
└── README.md
```

---

## Installation
git clone https://github.com/HABS638/ReSUNet_Segmentation.git
cd ReSUNet_Segmentation

python -m venv venv

# Activation de l'environnement virtuel

# Linux / Mac
source venv/bin/activate

# Windows (cmd)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

pip install -r requirements.txt

**Key dependencies:** TensorFlow ≥ 2.12, `higra` (max-tree computation), `nibabel` (NIfTI I/O).

---

## Data preparation

### Datasets

| Dataset | Role | Download |
|---|---|---|
| [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) | Unlabeled pretext training (DICOM CT) | TCIA |
| [LiTS 2017](https://competitions.codalab.org/competitions/17094) | Labeled segmentation fine-tuning | Codalab |

Alternatively, download via Kaggle:
```bash
kaggle datasets download -d andrewmvd/liver-tumor-segmentation   # LiTS
kaggle datasets download -d ayu055/lidcidri                       # LIDC-IDRI
```

### File list format

The training scripts expect plain text files listing one file path per line:

```
data/
├── pretrain_images.txt   # Unlabeled LIDC-IDRI paths (Stage 1)
├── train_images.txt      # LiTS training image paths (Stage 2)
├── train_masks.txt       # LiTS training mask paths
├── val_images.txt
└── val_masks.txt
```

Generate them with:
```bash
find /path/to/lidc   -name '*.nii.gz'  > data/pretrain_images.txt
find /path/to/lits/images -name '*.nii' | sort > data/train_images.txt
find /path/to/lits/masks  -name '*.nii' | sort > data/train_masks.txt
```

---

## Usage

### Full two-stage pipeline

```bash
python src/train.py \
  --mode full \
  --data_dir data \
  --epochs_pretrain 30 \
  --epochs_seg 50 \
  --batch_size 8 \
  --criterion area_ratio \
  --save_dir checkpoints
```

### Segmentation only (from existing pretext checkpoint)

```bash
python src/train.py \
  --mode seg \
  --data_dir data \
  --pretrain_ckpt checkpoints/pretext_best.h5 \
  --epochs_seg 50
```

### Limited-data experiment (e.g., 10 % of labels)

```bash
python src/train.py --mode full --data_pct 0.1 --save_dir checkpoints/10pct
```

### Inference

```bash
# Single volume
python src/inference.py \
  --model checkpoints/seg_best.h5 \
  --input data/test_volume.nii.gz \
  --mask  data/test_mask.nii.gz \
  --output results/

# Full folder
python src/inference.py \
  --model checkpoints/seg_best.h5 \
  --input data/test_folder/ \
  --output results/
```

---

## Metrics

All metrics are computed with proper accumulation over the full validation set.

| Metric | Implementation | Description |
|---|---|---|
| **Dice** | `DiceCoefficient` (Keras) + `dice_numpy` | 2×\|X∩Y\| / (\|X\|+\|Y\|) |
| **IoU** | `IoUScore` (Keras) + `iou_numpy` | \|X∩Y\| / \|X∪Y\| |
| **Precision** | `precision_recall_numpy` | TP / (TP + FP) |
| **Recall** | `precision_recall_numpy` | TP / (TP + FN) |
| **Loss** | `bce_dice_loss` | Binary cross-entropy + soft Dice |

Results are saved to `checkpoints/results.json` after each training run.

---

## Architecture

**ReSUNet** — Residual U-Net with 3 encoder levels, bottleneck, and 3 decoder levels.

```
Input (224×224×1)
│
├─ Encoder block 1 → skip₁  (224×224×32)
│       └─ MaxPool → (112×112×32)
├─ Encoder block 2 → skip₂  (112×112×64)
│       └─ MaxPool → (56×56×64)
├─ Encoder block 3 → skip₃  (56×56×128)
│       └─ MaxPool → (28×28×128)
│
├─ Bottleneck residual block  (28×28×256)
│
├─ Decoder block 3 ← skip₃  (56×56×128)
├─ Decoder block 2 ← skip₂  (112×112×64)
└─ Decoder block 1 ← skip₁  (224×224×32)
        └─ Conv 1×1 + sigmoid → Mask (224×224×1)
```

Each **residual block** = Conv-BN-ReLU × 2 + identity skip.  
**Decoder block** = UpSampling2D → Concatenate(skip) → Residual block.

---

## Notebook (Google Colab)

The full pipeline (data download → pretext → segmentation → visualisation) is available in:

```
notebooks/ReSUnet_segmentation.ipynb
```

Open directly in Colab:  
[![Open In Colab]: https://colab.research.google.com/drive/1PFOtzEb1-qDZmfbm-EYEd20wsh7tkBWM?usp=sharing

---

## Reference

```bibtex
@inproceedings{tang2022maxtree,
  title     = {Self-supervised Learning Based on Max-tree Representation for Medical Image Segmentation},
  author    = {Tang, Qian and Du, Bo and Xu, Yongchao},
  booktitle = {2022 International Joint Conference on Neural Networks (IJCNN)},
  year      = {2022},
  doi       = {10.1109/IJCNN55064.2022.9892853}
}
```

---

