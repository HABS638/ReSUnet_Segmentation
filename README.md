# ReSUNet — Medical Image Segmentation with Self-supervised Learning

> **94.27% Dice** on liver segmentation · **+10 points** with only 10% labeled data · Colab-ready

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PFOtzEb1-qDZmfbm-EYEd20wsh7tkBWM?usp=sharing)

---

## The problem this solves

Annotating medical images requires expensive expert time. This project shows how to train a segmentation model that performs at **94%+ Dice** using **far less labeled data** — by first learning image structure from unlabeled CT scans.

The key idea: before training on labeled data, the model learns to reconstruct a structural representation of each image (Max-tree). This pre-training gives it a head start that matters most when labels are scarce.

---

## Results

| Labels available | Without pre-training | **With Max-tree SSL** | Gain |
|---|---|---|---|
| 10% | 82.5% | **92.8%** | **+10.3 pts** |
| 50% | 90.2% | **94.0%** | +3.8 pts |
| 100% | 90.3% | **94.3%** | +4.0 pts |

> Evaluated on [LiTS 2017](https://competitions.codalab.org/competitions/17094) liver segmentation · Dice coefficient

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/ReSUNet_Segmentation.git
cd ReSUNet_Segmentation
pip install -r requirements.txt
```

**Or run everything in one click →** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/ReSUNet_Segmentation/blob/main/notebooks/ReSUnet_segmentation.ipynb)

The notebook handles data download, training, evaluation, and visualizations automatically.

---

## How it works in 2 steps

**Step 1 — Pre-train on unlabeled data** (no annotations needed)
The model learns to reconstruct a structural map of each CT scan. This forces it to understand anatomy before seeing a single label.

**Step 2 — Fine-tune on labeled data**
Pre-trained weights are transferred. The model reaches higher accuracy with fewer labeled examples.

---

## Run training

```bash
# Full pipeline (pre-training + segmentation)
python src/train.py --mode full --data_dir data --epochs_pretrain 30 --epochs_seg 50

# Segmentation only, from an existing checkpoint
python src/train.py --mode seg --pretrain_ckpt checkpoints/pretext_best.h5

# Run inference on a CT volume
python src/inference.py --model checkpoints/seg_best.h5 --input scan.nii.gz
```

---

## Data

Download via Kaggle:
```bash
kaggle datasets download -d andrewmvd/liver-tumor-segmentation  # LiTS 2017 (labeled)
kaggle datasets download -d ayu055/lidcidri                      # LIDC-IDRI (unlabeled)
```

---

## Based on

Tang et al., *Self-supervised Learning Based on Max-tree Representation for Medical Image Segmentation*, IJCNN 2022. [`doi`](https://doi.org/10.1109/IJCNN55064.2022.9892853)

