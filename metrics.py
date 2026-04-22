"""
metrics.py — Segmentation evaluation metrics.

Provides both:
  - TensorFlow/Keras metric classes (for model.compile / model.fit)
  - NumPy functions (for post-hoc evaluation on saved predictions)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


# ---------------------------------------------------------------------------
# Keras metric classes
# ---------------------------------------------------------------------------

class DiceCoefficient(tf.keras.metrics.Metric):
    """
    Mean Dice coefficient over a batch.
    Accumulates intersection and union across batches,
    then computes the global Dice at the end of each epoch.
    """

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6, name: str = "dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.smooth = smooth
        self.intersection = self.add_weight("intersection", initializer="zeros")
        self.sum_pred_true = self.add_weight("sum_pred_true", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        self.intersection.assign_add(tf.reduce_sum(y_true * y_pred))
        self.sum_pred_true.assign_add(tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))

    def result(self):
        return (2.0 * self.intersection + self.smooth) / (self.sum_pred_true + self.smooth)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.sum_pred_true.assign(0.0)


class IoUScore(tf.keras.metrics.Metric):
    """Intersection over Union (Jaccard index)."""

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6, name: str = "iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.smooth = smooth
        self.intersection = self.add_weight("intersection", initializer="zeros")
        self.union = self.add_weight("union", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        inter = tf.reduce_sum(y_true * y_pred)
        self.intersection.assign_add(inter)
        self.union.assign_add(tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter)

    def result(self):
        return (self.intersection + self.smooth) / (self.union + self.smooth)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


# ---------------------------------------------------------------------------
# Keras loss functions
# ---------------------------------------------------------------------------

def dice_loss(y_true, y_pred, smooth: float = 1e-6):
    """Soft Dice loss (differentiable, no thresholding)."""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * inter + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def bce_dice_loss(y_true, y_pred):
    """Combined binary cross-entropy + Dice loss (common in medical imaging)."""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


# ---------------------------------------------------------------------------
# NumPy evaluation helpers
# ---------------------------------------------------------------------------

def dice_numpy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Dice coefficient on numpy arrays."""
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    y_true = y_true.astype(np.float32)
    inter = np.sum(y_true * y_pred_bin)
    return float((2.0 * inter + smooth) / (np.sum(y_true) + np.sum(y_pred_bin) + smooth))


def iou_numpy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """IoU on numpy arrays."""
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    y_true = y_true.astype(np.float32)
    inter = np.sum(y_true * y_pred_bin)
    union = np.sum(y_true) + np.sum(y_pred_bin) - inter
    return float((inter + smooth) / (union + smooth))


def precision_recall_numpy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
    """Returns (precision, recall) tuple."""
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    y_true = y_true.astype(np.float32)
    tp = np.sum(y_true * y_pred_bin)
    fp = np.sum((1 - y_true) * y_pred_bin)
    fn = np.sum(y_true * (1 - y_pred_bin))
    precision = float(tp / (tp + fp + 1e-6))
    recall = float(tp / (tp + fn + 1e-6))
    return precision, recall


def evaluate_batch(y_true_batch: np.ndarray, y_pred_batch: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Evaluate a whole batch; return mean metrics dict.

    Args:
        y_true_batch: (N, H, W) or (N, H, W, 1)
        y_pred_batch: same shape as y_true_batch

    Returns:
        {"dice": float, "iou": float, "precision": float, "recall": float}
    """
    if y_true_batch.ndim == 4:
        y_true_batch = y_true_batch[..., 0]
        y_pred_batch = y_pred_batch[..., 0]

    dices, ious, precs, recs = [], [], [], []
    for i in range(len(y_true_batch)):
        dices.append(dice_numpy(y_true_batch[i], y_pred_batch[i], threshold))
        ious.append(iou_numpy(y_true_batch[i], y_pred_batch[i], threshold))
        p, r = precision_recall_numpy(y_true_batch[i], y_pred_batch[i], threshold)
        precs.append(p)
        recs.append(r)

    return {
        "dice":      float(np.mean(dices)),
        "iou":       float(np.mean(ious)),
        "precision": float(np.mean(precs)),
        "recall":    float(np.mean(recs)),
    }
