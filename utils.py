"""
utils.py — Visualisation and evaluation utilities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.io import imread


# ---------------------------------------------------------------------------
# Qualitative visualisation
# ---------------------------------------------------------------------------

def show_pair(image_path: str, mask_path: str, pred_path: str = None, save_path: str = None):
    """
    Display (and optionally save) an image / GT / prediction triplet.
    Accepts PNG, JPEG, or any skimage-compatible format.
    """
    img = imread(image_path, as_gray=True)
    msk = imread(mask_path, as_gray=True)

    n = 3 if pred_path else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("CT input", fontsize=11)

    axes[1].imshow(msk, cmap="hot")
    axes[1].set_title("Ground truth", fontsize=11)

    if pred_path:
        pred = imread(pred_path, as_gray=True)
        axes[2].imshow(pred, cmap="hot")
        axes[2].set_title("Prediction", fontsize=11)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved → {save_path}")
        plt.close()
    else:
        plt.show()


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4,
                 color: tuple = (1.0, 0.3, 0.0)) -> np.ndarray:
    """
    Return an RGB image with a coloured overlay for the mask.

    Args:
        image:  2-D float array in [0, 1].
        mask:   Binary 2-D array.
        alpha:  Overlay opacity.
        color:  (R, G, B) in [0,1].

    Returns:
        (H, W, 3) RGB uint8 array.
    """
    rgb = np.stack([image, image, image], axis=-1)   # greyscale → RGB
    for c, v in enumerate(color):
        rgb[:, :, c] = np.where(mask > 0.5, (1 - alpha) * image + alpha * v, image)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def plot_training_history(csv_path: str, out_path: str = None):
    """
    Plot training curves from a Keras CSVLogger file.
    Columns expected: epoch, loss, val_loss, dice, val_dice, iou, val_iou
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    _plot_metric(axes[0], df, "loss",  "Loss (BCE + Dice)")
    _plot_metric(axes[1], df, "dice",  "Dice coefficient")
    _plot_metric(axes[2], df, "iou",   "IoU")

    plt.suptitle("Training history", fontsize=13)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved → {out_path}")
        plt.close()
    else:
        plt.show()


def _plot_metric(ax, df, col, title):
    if col in df.columns:
        ax.plot(df["epoch"], df[col], label="train", linewidth=1.5)
    val_col = f"val_{col}"
    if val_col in df.columns:
        ax.plot(df["epoch"], df[val_col], label="val", linewidth=1.5, linestyle="--")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# Quantitative summary table
# ---------------------------------------------------------------------------

def print_results_table(results: dict):
    """Pretty-print a metrics dict."""
    print("\n┌─────────────────────────────────────────────┐")
    print("│          Segmentation Results                │")
    print("├──────────────┬──────────────────────────────┤")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"│  {k:<12}│  {v:.4f}                       │")
        else:
            print(f"│  {k:<12}│  {v!s:<29} │")
    print("└──────────────┴──────────────────────────────┘\n")
