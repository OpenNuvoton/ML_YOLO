import argparse
import glob
import os
import random

import cv2
import numpy as np

# Declare as global variables, can be updated based trained model image size
img_width = 320
img_height = 320

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', default=r'D:\ML_train_data\CelebA_Spoof\CelebA_Spoof_crop\data_1.5_80\train')
    parser.add_argument('--img-size', type=int, nargs=2, default=[320, 320])
    parser.add_argument('--n-img', type=int, default=200)
    parser.add_argument('-o', '--out', default='calib_data.npy')
    parser.add_argument("--no_torgb", action="store_true", help="convert from BGR to RGB")
    return parser.parse_args()


def main():
    # reference:
    # https://github.com/PINTO0309/onnx2tf#7-calibration-data-creation-for-int8-quantization

    args = parse_args()

    img_paths = glob.glob(os.path.join(args.img_dir, '*'))
    img_paths.sort()

    assert len(img_paths) >= args.n_img

    random.seed(0)
    random.shuffle(img_paths)

    print(len(img_paths))

    if not args.no_torgb:
        print("convert from BGR to RGB format!!")

    calib_data = []
    images_stack = []
    for i, img_path in enumerate(img_paths):
        if i >= args.n_img:
            break
        img = cv2.imread(img_path)
        #if not args.no_torgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ##img, _ = preprocess(img, args.img_size)
        img = cv2.resize(img, (args.img_size[0], args.img_size[1]),)
        img = img.astype(np.float32) / 255.0
        
        images_stack.append(img)
        img = img[np.newaxis]
        calib_data.append(img)

    stacked_images = np.stack(images_stack, axis=0)
    print(stacked_images.shape)
    mean = np.mean(stacked_images, axis=(0, 1, 2))  # Mean for each channel
    std = np.std(stacked_images, axis=(0, 1, 2))  # STD for each channel
    #print(mean, std)

    calib_data = np.vstack(calib_data)  # [n_img, img_size[0], img_size[1], 3]

    print(f'calib_datas.shape: {calib_data.shape}')

    np.save(file=args.out, arr=calib_data)


if __name__ == '__main__':
    main()
