"""
max_tree.py — Max-tree representation for self-supervised pretext task.

Implements the image transformation described in:
  Tang et al., IJCNN 2022.

Pipeline per image:
  1. Build max-tree via union-find + canonization (Algorithms 1 & 2 in paper).
  2. Compute area attribute for each node.
  3. Compute area-ratio attribute (node_area / parent_area).
  4. Restitute to image: pixel ← area_ratio × 255.

Requires: higra  (pip install higra)
"""

import numpy as np

try:
    import higra as hg
    _HIGRA_AVAILABLE = True
except ImportError:
    _HIGRA_AVAILABLE = False
    print("[max_tree] WARNING: higra not installed — max_tree functions unavailable.")


# ---------------------------------------------------------------------------
# Core transformation
# ---------------------------------------------------------------------------

def compute_max_tree_image(
    image: np.ndarray,
    criterion: str = "area_ratio",
) -> np.ndarray:
    """
    Transform a 2-D grayscale image using max-tree area-ratio attribute.

    Args:
        image:     2-D float array, any range (will be normalised to uint8).
        criterion: One of {'area_ratio', 'area', 'volume', 'contrast'}.

    Returns:
        Transformed image as float32 in [0, 1].
    """
    if not _HIGRA_AVAILABLE:
        raise RuntimeError("higra is required. Install via: pip install higra")

    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {image.shape}")

    # --- Normalise to uint8 ---
    mn, mx = image.min(), image.max()
    if mx == mn:
        return np.zeros_like(image, dtype=np.float32)
    img8 = ((image - mn) / (mx - mn) * 255).astype(np.uint8)

    # --- Build max-tree ---
    graph = hg.get_4_adjacency_graph(img8.shape)
    tree, altitudes = hg.component_tree_max_tree(graph, img8.ravel())

    # --- Compute attribute ---
    if criterion == "area_ratio":
        area = hg.attribute_area(tree)
        # area_ratio[node] = area[node] / area[parent[node]]
        parents = tree.parents()
        parent_area = area[parents]
        parent_area = np.where(parent_area == 0, 1, parent_area)  # avoid /0 at root
        attribute = area / parent_area
    elif criterion == "area":
        attribute = hg.attribute_area(tree).astype(np.float64)
        attribute = attribute / (attribute.max() + 1e-8)
    elif criterion == "volume":
        attribute = hg.attribute_volume(tree, altitudes).astype(np.float64)
        attribute = attribute / (attribute.max() + 1e-8)
    elif criterion == "contrast":
        attribute = hg.attribute_extinction_value(tree, altitudes, hg.attribute_area(tree))
        attribute = attribute.astype(np.float64)
        attribute = attribute / (attribute.max() + 1e-8)
    else:
        raise ValueError(f"Unknown criterion '{criterion}'. "
                         "Choose from: area_ratio, area, volume, contrast.")

    # --- Restitute image ---
    # Each leaf pixel gets its node's attribute value
    leaf_attr = hg.reconstruct_leaf_data(tree, attribute)
    transformed = leaf_attr.reshape(img8.shape).astype(np.float32)

    # Clip to [0,1]
    transformed = np.clip(transformed, 0.0, 1.0)
    return transformed


# ---------------------------------------------------------------------------
# Batch helper (numpy)
# ---------------------------------------------------------------------------

def batch_max_tree_transform(
    images: np.ndarray,
    criterion: str = "area_ratio",
) -> np.ndarray:
    """
    Apply max-tree transformation to a batch of images.

    Args:
        images:    (N, H, W) or (N, H, W, 1) float array.
        criterion: Passed to compute_max_tree_image.

    Returns:
        Transformed batch, same shape as input, dtype float32.
    """
    squeeze = images.ndim == 4
    if squeeze:
        images = images[..., 0]
    transformed = np.stack(
        [compute_max_tree_image(images[i], criterion) for i in range(len(images))],
        axis=0,
    )
    if squeeze:
        transformed = transformed[..., np.newaxis]
    return transformed.astype(np.float32)


# ---------------------------------------------------------------------------
# Quick visual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Synthetic test image (gradient + circles)
    H, W = 128, 128
    y, x = np.mgrid[:H, :W]
    img = np.sin(x / 20.0) * np.cos(y / 20.0)
    img = (img - img.min()) / (img.max() - img.min())

    transformed = compute_max_tree_image(img, criterion="area_ratio")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img, cmap="gray");         axes[0].set_title("Original")
    axes[1].imshow(transformed, cmap="gray"); axes[1].set_title("Max-tree (area ratio)")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig("max_tree_test.png", dpi=120)
    print("Saved max_tree_test.png")
