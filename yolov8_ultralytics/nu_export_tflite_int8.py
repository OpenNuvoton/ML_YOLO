from ultralytics import YOLO
from ultralytics.utils import (
    ARM64,
    DEFAULT_CFG,
    IS_JETSON,
    LINUX,
    LOGGER,
    MACOS,
    PYTHON_VERSION,
    ROOT,
    WINDOWS,
    __version__,
    callbacks,
    colorstr,
    get_default_args,
    yaml_save,
)

import onnx2tf
from pathlib import Path
import argparse
import os
import glob
import random
import cv2
import numpy as np


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', type=str, default='onnx', help='onnx or tflite')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', type=str, default='cpu', help='device to use')

    parser.add_argument('--img-dir', type=str, help='img dataset of calib')
    parser.add_argument('--n-img', type=int, default=200)
    parser.add_argument('-o', '--out', default='calib_data.npy')

    return parser.parse_args()

if __name__ == '__main__':

    args = parser_arguments()
    # Create a dictionary of kwargs
    kwargs = vars(args)
    print(kwargs)
    f = Path(args.weights).parent

    if args.format == "onnx":
        model = YOLO(args.weights)
        # convert to onnx
        success = model.export(format="onnx", imgsz=args.imgsz, simplify=True, separate_outputs=True, export_hw_optimized=True, device=args.device)
    else: # tflite int8
        # load the calibration data
        if not os.path.exists(args.out):
            img_paths = glob.glob(os.path.join(args.img_dir, '*'))
            img_paths.sort()
            assert len(img_paths) >= args.n_img
            random.seed(0)
            random.shuffle(img_paths)
            print(len(img_paths))
        
            calib_data = []
            images_stack = []
            for i, img_path in enumerate(img_paths):
                if i >= args.n_img:
                    break
                img = cv2.imread(img_path)
                #if not args.no_torgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
                ##img, _ = preprocess(img, args.img_size)
                img = cv2.resize(img, (args.imgsz, args.imgsz),)
                img = img.astype(np.float32) / 255.0
                
                images_stack.append(img)
                img = img[np.newaxis]
                calib_data.append(img)
        
            #stacked_images = np.stack(images_stack, axis=0)
            #print(stacked_images.shape)
            #mean = np.mean(stacked_images, axis=(0, 1, 2))  # Mean for each channel
            #std = np.std(stacked_images, axis=(0, 1, 2))  # STD for each channel
        
            calib_data = np.vstack(calib_data)  # [n_img, img_size[0], img_size[1], 3]
            print(f'calib_datas.shape: {calib_data.shape}')
            np.save(file=args.out, arr=calib_data)
        
        tmp_file = args.out  # int8 calibration images file
        np_data = [["images", tmp_file, [[[[0, 0, 0]]]], [[[[1, 1, 1]]]]]]
    
        # onnx2tf
        f_onnx = os.path.join(f, "best.onnx")
        
        io_quant_dtype = "int8" # "uint8"
    
        prefix = colorstr("TensorFlow SavedModel:")
        LOGGER.info(f"{prefix} starting TFLite export with onnx2tf {onnx2tf.__version__}...")
        onnx2tf.convert(
            input_onnx_file_path=f_onnx,
            output_folder_path=str(f),
            not_use_onnxsim=True,
            verbosity="info",
            output_integer_quantized_tflite=True,
            quant_type="per-tensor",  # "per-tensor" (faster) or "per-channel" (slower but more accurate)
            custom_input_op_name_np_data_path=np_data,
            input_output_quant_dtype=io_quant_dtype,
        )