"""
ReSUNet — Residual U-Net for medical image segmentation.

Architecture:
  - 3-level encoder with residual blocks + MaxPool
  - Bottleneck residual block
  - 3-level decoder with UpSampling + skip connections
  - Final 1×1 sigmoid head

Reference:
  Tang et al., "Self-supervised Learning Based on Max-tree Representation
  for Medical Image Segmentation", IJCNN 2022.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def conv_bn_relu(x, filters: int, kernel_size: int = 3) -> tf.Tensor:
    x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def residual_block(x, filters: int) -> tf.Tensor:
    """Two-conv residual block with identity shortcut (projection-free)."""
    shortcut = x
    x = conv_bn_relu(x, filters)
    x = conv_bn_relu(x, filters)
    # Projection shortcut if channel dims differ
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.add([shortcut, x])
    x = layers.Activation("relu")(x)
    return x


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks
# ---------------------------------------------------------------------------

def encoder_block(x, filters: int):
    """Residual block → skip, then MaxPool → pooled."""
    skip = residual_block(x, filters)
    pooled = layers.MaxPool2D((2, 2))(skip)
    return skip, pooled


def decoder_block(x, skip, filters: int) -> tf.Tensor:
    """UpSampling → concatenate skip → residual block."""
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip])
    x = residual_block(x, filters)
    return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def build_resunet(
    input_shape: tuple = (256, 256, 1),
    num_classes: int = 1,
    filters: tuple = (32, 64, 128, 256),
) -> Model:
    """
    Build the ReSUNet model.

    Args:
        input_shape: (H, W, C) — use C=1 for grayscale CT/MRI.
        num_classes:  1 for binary segmentation (liver, tumour …).
        filters:      Feature-map sizes at each encoder level + bottleneck.

    Returns:
        Compiled-ready Keras Model named 'ReSUNet'.
    """
    f1, f2, f3, f_bottle = filters

    inputs = layers.Input(shape=input_shape, name="input")

    # --- Encoder ---
    s1, p1 = encoder_block(inputs, f1)   # 256 → 128
    s2, p2 = encoder_block(p1, f2)       # 128 →  64
    s3, p3 = encoder_block(p2, f3)       #  64 →  32

    # --- Bottleneck ---
    b = residual_block(p3, f_bottle)     #  32 ×  32 × 256

    # --- Decoder ---
    d3 = decoder_block(b, s3, f3)        #  32 →  64
    d2 = decoder_block(d3, s2, f2)       #  64 → 128
    d1 = decoder_block(d2, s1, f1)       # 128 → 256

    # --- Output head ---
    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Conv2D(
        num_classes, 1, padding="same", activation=activation, name="output"
    )(d1)

    model = Model(inputs, outputs, name="ReSUNet")
    return model


if __name__ == "__main__":
    m = build_resunet()
    m.summary()
