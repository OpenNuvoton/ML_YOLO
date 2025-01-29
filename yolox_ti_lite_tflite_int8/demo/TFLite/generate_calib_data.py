"""Generate calibration data for INT8 quantization.
This module generates calibration data from a directory of images.
The generated data can be used for INT8 quantization of a model.
It preprocesses images to a specified size and optionally converts from BGR to RGB color space.

Functions:
    parse_args: Parses command line arguments.
    main: Generates and saves calibration data.

Example:
    python generate_calib_data.py --img-dir datasets/coco/train2017/ --n-img 200 -o calib_data.npy
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np

from utils import preprocess


def parse_args():
    """Parse command line arguments.
    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', default='datasets/coco/train2017/')
    parser.add_argument('--img-size', type=int, nargs=2, default=[416, 416])
    parser.add_argument('--n-img', type=int, default=200)
    parser.add_argument('-o', '--out', default='calib_data.npy')
    parser.add_argument("--no_torgb", action="store_true", help="convert from BGR to RGB")
    return parser.parse_args()


def main():
    """Generate and save calibration data.
    This function parses command line arguments, selects a subset of images,
    preprocesses them (resizing and optionally color space conversion),
    and saves the resulting data as a NumPy array.
    """
    # reference:
    # https://github.com/PINTO0309/onnx2tf#7-calibration-data-creation-for-int8-quantization

    args = parse_args()

    img_paths = glob.glob(os.path.join(args.img_dir, '*'))
    img_paths.sort()

    assert len(img_paths) >= args.n_img

    random.seed(0)
    random.shuffle(img_paths)

    if not args.no_torgb:
        print("convert from BGR to RGB format!!")

    calib_data = []
    for i, img_path in enumerate(img_paths):
        if i >= args.n_img:
            break
        img = cv2.imread(img_path)
        if not args.no_torgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, _ = preprocess(img, args.img_size)
        img = img[np.newaxis]
        calib_data.append(img)
    calib_data = np.vstack(calib_data)  # [n_img, img_size[0], img_size[1], 3]

    print(f'calib_datas.shape: {calib_data.shape}')

    np.save(file=args.out, arr=calib_data)


if __name__ == '__main__':
    main()
