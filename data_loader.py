"""
data_loader.py — tf.data pipelines for NIfTI CT volumes.

Two datasets:
  1. prepare_pretext_dataset   — unlabeled LIDC-IDRI slices → (image, max_tree_image)
  2. prepare_tf_dataset        — labeled LiTS slices        → (image, mask)

Both handle:
  - Middle-slice extraction from 3-D NIfTI volumes
  - HU clipping and [0,1] normalisation
  - Resize to target resolution
  - Shuffle + batch + prefetch

Tip: generate the txt file lists with:
  find /path/to/lidc -name '*.nii.gz' > data/pretrain_images.txt
  find /path/to/lits/images -name '*.nii' > data/train_images.txt
  find /path/to/lits/masks  -name '*.nii' > data/train_masks.txt
"""

import os
import numpy as np
import tensorflow as tf
import nibabel as nib
from skimage.transform import resize as sk_resize

from src.max_tree import compute_max_tree_image


# ---------------------------------------------------------------------------
# NIfTI helpers
# ---------------------------------------------------------------------------

def _load_nifti_slice(path: bytes, hu_clip: tuple = (-200, 200)) -> np.ndarray:
    """
    Load the middle axial slice of a NIfTI file.
    Apply HU clipping then normalise to [0, 1].
    Returns a (H, W, 1) float32 array.
    """
    vol = nib.load(path.decode()).get_fdata().astype(np.float32)

    if vol.ndim == 3:
        z = vol.shape[2] // 2
        img = vol[:, :, z]
    else:
        img = vol  # already 2-D

    # HU clipping
    img = np.clip(img, hu_clip[0], hu_clip[1])
    # Normalise
    rng = hu_clip[1] - hu_clip[0]
    img = (img - hu_clip[0]) / rng  # → [0, 1]
    return img[..., np.newaxis].astype(np.float32)


def _load_nifti_mask(path: bytes) -> np.ndarray:
    """Load binary mask (middle slice), return (H, W, 1) float32."""
    vol = nib.load(path.decode()).get_fdata()
    if vol.ndim == 3:
        z = vol.shape[2] // 2
        msk = vol[:, :, z]
    else:
        msk = vol
    msk = (msk > 0).astype(np.float32)
    return msk[..., np.newaxis]


# ---------------------------------------------------------------------------
# Pretext dataset (max-tree self-supervised)
# ---------------------------------------------------------------------------

def _pretext_load_fn(
    img_path: tf.Tensor,
    img_size: tuple,
    criterion: str,
    hu_clip: tuple,
) -> tuple:
    """tf.numpy_function body: loads image, computes max-tree target."""
    img = _load_nifti_slice(img_path.numpy(), hu_clip=hu_clip)   # (H, W, 1)
    img_2d = img[..., 0]

    # Resize to target
    img_r = sk_resize(img_2d, img_size, preserve_range=True).astype(np.float32)

    # Max-tree transformation as self-supervised target
    target = compute_max_tree_image(img_r, criterion=criterion)

    return (
        img_r[..., np.newaxis].astype(np.float32),
        target[..., np.newaxis].astype(np.float32),
    )


def prepare_pretext_dataset(
    image_files: list,
    batch_size: int = 8,
    img_size: tuple = (224, 224),
    criterion: str = "area_ratio",
    hu_clip: tuple = (-1000, 1000),
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Build a tf.data Dataset for the SSL pretext task.

    Yields: (image, max_tree_image) pairs, both (H, W, 1) float32 in [0,1].
    """
    def _map(path):
        img, tgt = tf.numpy_function(
            lambda p: _pretext_load_fn(p, img_size, criterion, hu_clip),
            [path],
            (tf.float32, tf.float32),
        )
        img.set_shape([img_size[0], img_size[1], 1])
        tgt.set_shape([img_size[0], img_size[1], 1])
        return img, tgt

    ds = tf.data.Dataset.from_tensor_slices(image_files)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_files), seed=seed)
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Segmentation dataset (labeled LiTS)
# ---------------------------------------------------------------------------

def _seg_load_fn(
    img_path: tf.Tensor,
    msk_path: tf.Tensor,
    img_size: tuple,
    hu_clip: tuple,
) -> tuple:
    """tf.numpy_function body: loads image + mask."""
    img = _load_nifti_slice(img_path.numpy(), hu_clip=hu_clip)[..., 0]
    msk = _load_nifti_mask(msk_path.numpy())[..., 0]

    img_r = sk_resize(img, img_size, preserve_range=True).astype(np.float32)
    msk_r = sk_resize(msk, img_size, order=0, preserve_range=True).astype(np.float32)

    return (
        img_r[..., np.newaxis],
        (msk_r > 0.5)[..., np.newaxis].astype(np.float32),
    )


def prepare_tf_dataset(
    image_files: list,
    mask_files: list,
    batch_size: int = 8,
    img_size: tuple = (224, 224),
    hu_clip: tuple = (-200, 200),
    shuffle: bool = True,
    seed: int = 42,
    augment: bool = False,
) -> tf.data.Dataset:
    """
    Build a tf.data Dataset for supervised segmentation.

    Yields: (image, mask) pairs, both (H, W, 1) float32.
    """
    if not image_files or not mask_files:
        raise ValueError("image_files and mask_files must not be empty.")
    if len(image_files) != len(mask_files):
        raise ValueError(
            f"Mismatch: {len(image_files)} images vs {len(mask_files)} masks."
        )

    def _map(ip, mp):
        img, msk = tf.numpy_function(
            lambda i, m: _seg_load_fn(i, m, img_size, hu_clip),
            [ip, mp],
            (tf.float32, tf.float32),
        )
        img.set_shape([img_size[0], img_size[1], 1])
        msk.set_shape([img_size[0], img_size[1], 1])
        return img, msk

    ds = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_files), seed=seed)
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Data augmentation (optional)
# ---------------------------------------------------------------------------

def _augment(image: tf.Tensor, mask: tf.Tensor) -> tuple:
    """Random flips and 90° rotations applied consistently to image + mask."""
    stacked = tf.concat([image, mask], axis=-1)  # (H, W, 2)

    # Random horizontal flip
    stacked = tf.image.random_flip_left_right(stacked)
    # Random vertical flip
    stacked = tf.image.random_flip_up_down(stacked)
    # Random 90° rotation (k = 0..3)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    stacked = tf.image.rot90(stacked, k=k)

    image, mask = stacked[..., :1], stacked[..., 1:]
    return image, mask
