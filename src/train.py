"""Training script for ReSUnet"""
import argparse, os
import tensorflow as tf
from src.model_resunet import build_resunet
from src.data_loader import prepare_tf_dataset
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data', help='Chemin vers les données')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--img_size', type=int, nargs=2, default=[256,256])
    p.add_argument('--save_dir', type=str, default='checkpoints')
    return p.parse_args()

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    # NOTE: user must supply lists of image/mask paths, here we assume they exist as txt lists
    # This is a lightweight trainer for demonstration
    # For real use, replace by proper dataset parsing
    image_list_file = os.path.join(args.data_dir, 'images.txt')
    mask_list_file = os.path.join(args.data_dir, 'masks.txt')
    if not os.path.exists(image_list_file) or not os.path.exists(mask_list_file):
        print('Fichiers images.txt ou masks.txt non trouvés dans', args.data_dir)
        exit(1)
    with open(image_list_file) as f:
        image_files = [l.strip() for l in f.readlines() if l.strip()]
    with open(mask_list_file) as f:
        mask_files = [l.strip() for l in f.readlines() if l.strip()]
    ds = prepare_tf_dataset(image_files, mask_files, batch_size=args.batch_size, img_size=tuple(args.img_size))
    model = build_resunet(input_shape=(args.img_size[0], args.img_size[1], 1), num_classes=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    model.fit(ds, epochs=args.epochs)
    model.save(os.path.join(args.save_dir, 'model_last.h5'))
