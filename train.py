"""
train.py — Two-stage training pipeline.

Stage 1 — SSL Pretext Task (max-tree image reconstruction)
  - Input  X  : original CT slice (normalised)
  - Target X' : max-tree area-ratio image of X
  - Loss       : MSE (L2)
  - Architecture: ReSUNet with linear output head

Stage 2 — Segmentation Fine-tuning
  - Transfers encoder weights from Stage 1
  - Trains with BCE + Dice loss on annotated LiTS data
  - Metrics: Dice coefficient, IoU

Usage:
  # Full two-stage pipeline
  python src/train.py --mode full --data_dir data --epochs_pretrain 30 --epochs_seg 50

  # Only segmentation (from scratch or from existing checkpoint)
  python src/train.py --mode seg --data_dir data --epochs_seg 50

  # Only pretext
  python src/train.py --mode pretrain --data_dir data --epochs_pretrain 30
"""

import argparse
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.model_resunet import build_resunet
from src.data_loader import prepare_tf_dataset, prepare_pretext_dataset
from src.metrics import DiceCoefficient, IoUScore, bce_dice_loss, evaluate_batch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ReSUNet training — SSL + Segmentation")
    p.add_argument("--mode",            choices=["pretrain", "seg", "full"], default="full")
    p.add_argument("--data_dir",        default="data")
    p.add_argument("--save_dir",        default="checkpoints")
    p.add_argument("--img_size",        type=int, nargs=2, default=[224, 224])
    p.add_argument("--batch_size",      type=int, default=8)
    p.add_argument("--epochs_pretrain", type=int, default=30)
    p.add_argument("--epochs_seg",      type=int, default=50)
    p.add_argument("--lr_pretrain",     type=float, default=1e-4)
    p.add_argument("--lr_seg",          type=float, default=1e-4)
    p.add_argument("--criterion",       default="area_ratio",
                   choices=["area_ratio", "area", "volume", "contrast"],
                   help="Max-tree attribute for pretext task")
    p.add_argument("--pretrain_ckpt",   default=None,
                   help="Path to existing pretext checkpoint (skip Stage 1)")
    p.add_argument("--data_pct",        type=float, default=1.0,
                   help="Fraction of labelled data to use (e.g. 0.1 for 10%%)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_file_lists(data_dir, split="train"):
    """Load image / mask path lists from text files."""
    img_file = os.path.join(data_dir, f"{split}_images.txt")
    msk_file = os.path.join(data_dir, f"{split}_masks.txt")
    if not (os.path.exists(img_file) and os.path.exists(msk_file)):
        raise FileNotFoundError(
            f"Expected {img_file} and {msk_file}. "
            "Create them with one file path per line."
        )
    with open(img_file) as f:
        images = [l.strip() for l in f if l.strip()]
    with open(msk_file) as f:
        masks = [l.strip() for l in f if l.strip()]
    return images, masks


def load_unlabeled_list(data_dir):
    """Load unlabeled image paths for pretext task (LIDC-IDRI)."""
    ul_file = os.path.join(data_dir, "pretrain_images.txt")
    if not os.path.exists(ul_file):
        raise FileNotFoundError(
            f"Expected {ul_file} with one unlabeled CT file path per line."
        )
    with open(ul_file) as f:
        return [l.strip() for l in f if l.strip()]


def save_training_curves(history, out_path: str, title: str = "Training"):
    """Save a loss + metrics plot to PNG."""
    keys = [k for k in history.history if not k.startswith("val_")]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        ax.plot(history.history[k], label="train")
        if f"val_{k}" in history.history:
            ax.plot(history.history[f"val_{k}"], label="val")
        ax.set_title(k)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved curve → {out_path}")


# ---------------------------------------------------------------------------
# Stage 1 — SSL Pretext Task
# ---------------------------------------------------------------------------

def run_pretext_stage(args, save_dir: str) -> str:
    """
    Train the max-tree image-reconstruction pretext task.
    Returns path to the saved pretext checkpoint.
    """
    print("\n" + "=" * 60)
    print("  STAGE 1 — SSL Pretext Task (max-tree reconstruction)")
    print("=" * 60)

    unlabeled_images = load_unlabeled_list(args.data_dir)
    print(f"  Unlabeled images: {len(unlabeled_images)}")

    img_size = tuple(args.img_size)
    ds = prepare_pretext_dataset(
        unlabeled_images,
        batch_size=args.batch_size,
        img_size=img_size,
        criterion=args.criterion,
    )

    # ReSUNet with linear (no sigmoid) output for regression
    model = build_resunet(
        input_shape=(img_size[0], img_size[1], 1), num_classes=1
    )
    # Swap sigmoid head → linear
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Conv2D
    inp = model.input
    x = model.layers[-2].output  # layer before sigmoid Conv2D
    out = Conv2D(1, 1, padding="same", activation="linear", name="output_pretext")(x)
    pretext_model = Model(inp, out, name="ReSUNet_Pretext")

    pretext_model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr_pretrain),
        loss="mse",
        metrics=["mae"],
    )
    pretext_model.summary(line_length=80)

    ckpt_path = os.path.join(save_dir, "pretext_best.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    ]

    history = pretext_model.fit(
        ds,
        epochs=args.epochs_pretrain,
        callbacks=callbacks,
    )
    save_training_curves(
        history,
        os.path.join(save_dir, "pretrain_curves.png"),
        title="Pretext Task — MSE Loss",
    )
    print(f"  Pretext model saved → {ckpt_path}")
    return ckpt_path


# ---------------------------------------------------------------------------
# Stage 2 — Segmentation Fine-tuning
# ---------------------------------------------------------------------------

def run_segmentation_stage(args, save_dir: str, pretrain_ckpt: str = None) -> dict:
    """
    Fine-tune ReSUNet for liver segmentation.
    Optionally initialise encoder from pretext checkpoint.
    Returns dict of final evaluation metrics.
    """
    print("\n" + "=" * 60)
    print("  STAGE 2 — Segmentation Fine-tuning")
    print("=" * 60)

    train_images, train_masks = load_file_lists(args.data_dir, "train")
    val_images,   val_masks   = load_file_lists(args.data_dir, "val")

    # Optionally subsample for limited-data experiments
    if args.data_pct < 1.0:
        n = max(1, int(len(train_images) * args.data_pct))
        train_images, train_masks = train_images[:n], train_masks[:n]
    print(f"  Train slices: {len(train_images)}, Val slices: {len(val_images)}")

    img_size = tuple(args.img_size)
    train_ds = prepare_tf_dataset(
        train_images, train_masks, batch_size=args.batch_size, img_size=img_size
    )
    val_ds = prepare_tf_dataset(
        val_images, val_masks, batch_size=args.batch_size, img_size=img_size
    )

    # Build segmentation model
    model = build_resunet(input_shape=(img_size[0], img_size[1], 1), num_classes=1)

    # Transfer encoder weights from pretext task
    if pretrain_ckpt and os.path.exists(pretrain_ckpt):
        print(f"  Loading pretext weights from {pretrain_ckpt}")
        pretrain_model = tf.keras.models.load_model(pretrain_ckpt, compile=False)
        # Copy weights layer by layer where names match
        transferred = 0
        for layer in pretrain_model.layers:
            try:
                seg_layer = model.get_layer(layer.name)
                seg_layer.set_weights(layer.get_weights())
                transferred += 1
            except (ValueError, AttributeError):
                pass
        print(f"  Transferred weights for {transferred} layers.")
    else:
        print("  Training from scratch (no pretext weights).")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr_seg),
        loss=bce_dice_loss,
        metrics=[DiceCoefficient(name="dice"), IoUScore(name="iou")],
    )
    model.summary(line_length=80)

    seg_ckpt = os.path.join(save_dir, "seg_best.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            seg_ckpt, monitor="val_dice", mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice", mode="max", factor=0.5, patience=7, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice", mode="max", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(save_dir, "seg_log.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_seg,
        callbacks=callbacks,
    )
    save_training_curves(
        history,
        os.path.join(save_dir, "seg_curves.png"),
        title="Segmentation — BCE+Dice Loss | Dice | IoU",
    )

    # --- Final evaluation on val set ---
    print("\n  Final evaluation on validation set...")
    model = tf.keras.models.load_model(
        seg_ckpt,
        custom_objects={"bce_dice_loss": bce_dice_loss,
                        "DiceCoefficient": DiceCoefficient,
                        "IoUScore": IoUScore},
        compile=False,
    )

    all_preds, all_masks_np = [], []
    for x_batch, y_batch in val_ds:
        preds = model.predict_on_batch(x_batch)
        all_preds.append(preds.numpy())
        all_masks_np.append(y_batch.numpy())

    y_pred_all = np.concatenate(all_preds, axis=0)
    y_true_all = np.concatenate(all_masks_np, axis=0)
    metrics_result = evaluate_batch(y_true_all, y_pred_all)

    print(f"\n  ╔══════════════════════════════╗")
    print(f"  ║  Dice      : {metrics_result['dice']:.4f}            ║")
    print(f"  ║  IoU       : {metrics_result['iou']:.4f}            ║")
    print(f"  ║  Precision : {metrics_result['precision']:.4f}            ║")
    print(f"  ║  Recall    : {metrics_result['recall']:.4f}            ║")
    print(f"  ╚══════════════════════════════╝")

    # Save metrics to JSON
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as fp:
        json.dump({
            **metrics_result,
            "data_pct": args.data_pct,
            "criterion": args.criterion,
            "pretrain_used": pretrain_ckpt is not None,
        }, fp, indent=2)
    print(f"  Results saved → {results_path}")
    return metrics_result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    pretrain_ckpt = args.pretrain_ckpt

    if args.mode in ("pretrain", "full"):
        pretrain_ckpt = run_pretext_stage(args, args.save_dir)

    if args.mode in ("seg", "full"):
        run_segmentation_stage(args, args.save_dir, pretrain_ckpt)
