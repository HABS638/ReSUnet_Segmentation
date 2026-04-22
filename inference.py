"""
inference.py — Run ReSUNet on a single image or a folder.

Usage:
  python src/inference.py --model checkpoints/seg_best.h5 --input data/test_vol.nii.gz
  python src/inference.py --model checkpoints/seg_best.h5 --input data/test_folder/ --output results/
"""

import argparse
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import resize as sk_resize

from src.metrics import dice_numpy, iou_numpy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ReSUNet inference")
    p.add_argument("--model",    required=True, help="Path to .h5 checkpoint")
    p.add_argument("--input",    required=True, help="NIfTI file or folder of files")
    p.add_argument("--mask",     default=None,  help="Optional ground-truth mask for metrics")
    p.add_argument("--output",   default="results", help="Output directory")
    p.add_argument("--img_size", type=int, nargs=2, default=[224, 224])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--hu_clip",  type=float, nargs=2, default=[-200, 200])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def predict_nifti(model, nifti_path: str, img_size: tuple, hu_clip: tuple, threshold: float):
    """
    Load a NIfTI volume and predict mask for every axial slice.

    Returns:
        pred_vol: (H, W, D) uint8 binary prediction
        prob_vol: (H, W, D) float32 probability map
    """
    vol = nib.load(nifti_path).get_fdata().astype(np.float32)
    is_3d = vol.ndim == 3

    if not is_3d:
        vol = vol[..., np.newaxis]

    D = vol.shape[2]
    pred_vol = np.zeros_like(vol, dtype=np.uint8)
    prob_vol = np.zeros_like(vol, dtype=np.float32)

    for z in range(D):
        slc = vol[:, :, z]
        slc = np.clip(slc, hu_clip[0], hu_clip[1])
        slc = (slc - hu_clip[0]) / (hu_clip[1] - hu_clip[0])
        slc_r = sk_resize(slc, img_size, preserve_range=True).astype(np.float32)

        inp = slc_r[np.newaxis, ..., np.newaxis]   # (1, H, W, 1)
        prob = model.predict(inp, verbose=0)[0, ..., 0]  # (H, W)
        prob_orig = sk_resize(prob, slc.shape, preserve_range=True).astype(np.float32)

        prob_vol[:, :, z] = prob_orig
        pred_vol[:, :, z] = (prob_orig > threshold).astype(np.uint8)

    return pred_vol, prob_vol


def visualise_result(image_slc, pred_slc, gt_slc=None, out_path: str = None):
    """Save a side-by-side comparison figure."""
    n_cols = 3 if gt_slc is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(image_slc, cmap="gray")
    axes[0].set_title("CT slice")

    axes[1].imshow(pred_slc, cmap="hot")
    axes[1].set_title("Prediction")

    if gt_slc is not None:
        axes[2].imshow(gt_slc, cmap="gray")
        axes[2].set_title("Ground truth")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    img_size = tuple(args.img_size)

    print(f"Loading model from {args.model}")
    model = tf.keras.models.load_model(args.model, compile=False)
    model.summary(line_length=80)

    # Collect input files
    if os.path.isdir(args.input):
        files = [
            os.path.join(args.input, f)
            for f in sorted(os.listdir(args.input))
            if f.lower().endswith((".nii", ".nii.gz"))
        ]
    else:
        files = [args.input]

    print(f"Running inference on {len(files)} file(s)...")

    for fpath in files:
        stem = os.path.basename(fpath).split(".")[0]
        print(f"  Processing {stem} …")

        pred_vol, prob_vol = predict_nifti(
            model, fpath, img_size=img_size,
            hu_clip=tuple(args.hu_clip),
            threshold=args.threshold,
        )

        # Save prediction as NIfTI (preserves affine)
        ref_nii = nib.load(fpath)
        pred_nii = nib.Nifti1Image(pred_vol, ref_nii.affine, ref_nii.header)
        out_nii = os.path.join(args.output, f"{stem}_pred.nii.gz")
        nib.save(pred_nii, out_nii)
        print(f"    Saved NIfTI mask → {out_nii}")

        # Visualise middle slice
        z_mid = pred_vol.shape[2] // 2
        vol = nib.load(fpath).get_fdata().astype(np.float32)
        fig_path = os.path.join(args.output, f"{stem}_vis.png")

        # Load GT mask if provided
        gt_slc = None
        if args.mask:
            gt_vol = nib.load(args.mask).get_fdata()
            gt_slc = gt_vol[:, :, z_mid]
            dice = dice_numpy(gt_vol > 0, pred_vol)
            iou  = iou_numpy(gt_vol > 0, pred_vol)
            print(f"    Dice = {dice:.4f}  |  IoU = {iou:.4f}")

        visualise_result(
            vol[:, :, z_mid],
            pred_vol[:, :, z_mid],
            gt_slc,
            out_path=fig_path,
        )
        print(f"    Saved visualisation → {fig_path}")

    print("\nDone.")
