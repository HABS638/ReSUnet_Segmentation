"""Inference script to run ReSUnet on single images or folders."""
import argparse, os
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imsave, imread

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True, help='Path to model h5')
    p.add_argument('--input', type=str, required=True, help='Input image or folder')
    p.add_argument('--output', type=str, default='images/examples/predicted_mask.png')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = load_model(args.model, compile=False)
    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(('.png','.jpg','.nii','.nii.gz'))]
        for f in files:
            img = imread(f, as_gray=True)
            img_r = resize(img, (256,256), preserve_range=True)
            pred = model.predict(img_r[None,...,None])[0,...,0]
            pred_bin = (pred>0.5).astype('uint8')*255
            out_path = os.path.join('images/examples', os.path.basename(f).split('.')[0] + '_pred.png')
            imsave(out_path, pred_bin)
            print('Saved', out_path)
    else:
        img = imread(args.input, as_gray=True)
        img_r = resize(img, (256,256), preserve_range=True)
        pred = model.predict(img_r[None,...,None])[0,...,0]
        pred_bin = (pred>0.5).astype('uint8')*255
        imsave(args.output, pred_bin)
        print('Saved', args.output)
