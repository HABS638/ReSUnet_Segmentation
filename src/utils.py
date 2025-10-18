"""Small utility functions"""
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imsave

def show_pair(image_path, mask_path, pred_path=None):
    import matplotlib.pyplot as plt
    img = imread(image_path, as_gray=True)
    msk = imread(mask_path, as_gray=True)
    fig, axes = plt.subplots(1,3 if pred_path else 2, figsize=(12,4))
    axes[0].imshow(img, cmap='gray'); axes[0].set_title('Image')
    axes[1].imshow(msk, cmap='gray'); axes[1].set_title('Ground Truth')
    if pred_path:
        pred = imread(pred_path, as_gray=True)
        axes[2].imshow(pred, cmap='gray'); axes[2].set_title('Prediction')
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.show()
