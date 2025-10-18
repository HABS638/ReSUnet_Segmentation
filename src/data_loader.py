"""Utilities to load NIfTI / DICOM datasets and prepare tf.data datasets."""
import os
import numpy as np
import tensorflow as tf
import nibabel as nib
import pydicom
from skimage.transform import resize

def load_nifti_file(path):
    img = nib.load(path).get_fdata()
    return img.astype(np.float32)

def load_dicom_series(folder_path):
    # Load DICOM series from a folder (simple approach)
    slices = []
    for f in sorted(os.listdir(folder_path)):
        if f.lower().endswith('.dcm'):
            ds = pydicom.dcmread(os.path.join(folder_path, f))
            slices.append(ds.pixel_array)
    if len(slices)==0:
        return None
    arr = np.stack(slices, axis=0).astype(np.float32)
    return arr

def normalize_image(img):
    if img.max() == 0:
        return img
    return img / img.max()

def prepare_tf_dataset(image_files, mask_files, batch_size=8, img_size=(256,256)):
    if not image_files or not mask_files:
        raise ValueError('Liste d'images ou masques vide')
    def _load(i_path, m_path):
        img = tf.numpy_function(lambda p: nib.load(p.decode()).get_fdata().astype('float32'), [i_path], tf.float32)
        msk = tf.numpy_function(lambda p: nib.load(p.decode()).get_fdata().astype('float32'), [m_path], tf.float32)
        img = tf.expand_dims(img, axis=-1)
        img = tf.image.resize(img, img_size)
        msk = tf.expand_dims(msk, axis=-1)
        msk = tf.image.resize(msk, img_size)
        img = img / (tf.reduce_max(img) + 1e-8)
        msk = tf.cast(msk>0.5, tf.float32)
        return img, msk

    ds = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
