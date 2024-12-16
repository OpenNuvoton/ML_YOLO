import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow.lite as tflite

from utils import COCO_CLASSES, multiclass_nms_class_aware, preprocess, postprocess, vis, easy_preprocess, yolofastest_postprocess, yolofastest_preprocess

import tensorflow as tf
from pycocotools.coco import COCO

def load_cocoformat_labels(anno_path):
    coco = COCO(os.path.join("datasets", anno_path))
    cats = coco.loadCats(coco.getCatIds())
    _classes = tuple([c["name"] for c in cats])

    return _classes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='path to .tflite model')
    parser.add_argument('-i', '--img', required=True, help='path to image file')
    parser.add_argument('-o', '--out-dir', default='tmp/tflite', help='path to output directory')
    parser.add_argument('-s', '--score-thr', type=float, default=0.01, help='threshould to filter by scores')
    parser.add_argument('-mt', '--model-type', default='yolox', const='yolox', nargs='?',
                    choices=['yolox', 'yolof'], help='preprocess-way (default: %(default)s)')
    parser.add_argument('-pp', '--preprocess-way', default='cv2', const='cv2', nargs='?',
                    choices=['yolox', 'cv2'], help='preprocess-way (default: %(default)s)')
    parser.add_argument(
        "-a",
        "--anno_file",
        type=str,
        default='medicine_coco/annotations/medicine_train.json',
        help="Path to annotation file.",
    )
    parser.add_argument('--all_pics', action="store_true", help="use all plots in folder")
    parser.add_argument("--no_torgb", action="store_true", help="convert from BGR to RGB")
    return parser.parse_args()

def main():
    # reference:
    # https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/demo/tflite/yolox_tflite_demo.py


    # h5
    if 0: # for h5 model
        model = tf.keras.models.load_model(r'yolo-fastest-1.1.h5')
        model.summary()

    args = parse_args()

    # prepare model
    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # model info
    input_dtype = input_details[0]['dtype']
    input_scale = input_details[0]['quantization'][0]
    input_zero = input_details[0]['quantization'][1]
    print("Model Shape: {} {} Model Dtype: {}"
    .format(input_details[0]['shape'][1], input_details[0]['shape'][2], input_dtype))
    print("Model input Scale: {} Model input Zero point: {}"
    .format(input_scale, input_zero))

    output_dtype = output_details[0]['dtype']
    output_scale = output_details[0]['quantization'][0]
    output_zero  = output_details[0]['quantization'][1]
    print("Model output Shape: {} {} Model output Dtype: {}"
    .format(output_details[0]['shape'][0], output_details[0]['shape'][1], output_dtype))
    print("Model output Scale: {} Model output Zero point: {}"
    .format(output_details[0]['quantization'][0], output_details[0]['quantization'][1]))

    # preprocess
    input_shape = input_details[0]['shape']
    b, h, w, c = input_shape
    img_size = (h, w)

    # load classes label
    _classes = load_cocoformat_labels(args.anno_file)

    dir_path = Path(args.img).parent
    for filename in os.scandir(dir_path):
        
        if args.all_pics:
            origin_img = cv2.imread(os.path.join(filename))
        else:
            origin_img = cv2.imread(args.img)
        
        if not args.no_torgb:
             origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

        origin_img_size = (origin_img.shape[0], origin_img.shape[1])
    
        if args.model_type == 'yolof':
    
            # preprocess
            if (args.preprocess_way == 'cv2'):
                img, ratio = easy_preprocess(origin_img, img_size)
            else:    
                img, ratio = preprocess(origin_img, img_size)
            img = img[np.newaxis].astype(np.float32)  # add batch dim
            
            
            if input_dtype == np.int8:
                img = img - 128
                img = img.astype(np.int8)
            
            else:
                img = (img)/255
    
            if 0: # for h5 model
                predictions = model.predict(img)
                outputs_1 = np.array(predictions[0])[0]  # remove batch dim
                outputs_2 = np.array(predictions[1])[0]  # remove batch dim
                print(outputs_1.shape)
                print(outputs_2.shape)
                outputs_1.astype(np.float32)
                outputs_2.astype(np.float32)
    
            else:     
                # run inference
                interpreter.set_tensor(input_details[0]['index'], img)
                interpreter.invoke()
        
                outputs_1 = interpreter.get_tensor(output_details[0]['index'])[0]  # remove batch dim
                outputs_2 = interpreter.get_tensor(output_details[1]['index'])[0]  # remove batch dim
                # int8
                if input_dtype == np.int8:
                    outputs_1 = output_scale * (outputs_1.astype(np.float32) - output_zero)
                    outputs_2 = output_details[1]['quantization'][0] * (outputs_2.astype(np.float32) - output_details[1]['quantization'][1])
            
            anchor1 = [12, 18,  37, 49,  52,132]
            anchor2 = [115, 73, 119,199, 242,238]
            num_boxs = len(anchor1)/2
            class_num = int((outputs_1.shape[2] / num_boxs) - 5)
            assert class_num==len(_classes), "The classes doesn't match with yolofastest output"
    
            
            detection_res_list_0 = yolofastest_postprocess(outputs_1, anchor1, class_num, img_size, origin_img_size, args.preprocess_way, args.score_thr)
            detection_res_list_1 = yolofastest_postprocess(outputs_2, anchor2, class_num, img_size, origin_img_size, args.preprocess_way, args.score_thr)
            detection_res_list_0.extend(detection_res_list_1)
    
            print(len(detection_res_list_0))
    
            boxes_xyxy = np.ones((len(detection_res_list_0), 4))
            scores = np.zeros((len(detection_res_list_0), class_num))
            idx = 0
            for det in detection_res_list_0:
                boxes_xyxy[idx, 0] = det['x'] - det['w'] / 2.0
                boxes_xyxy[idx, 1] = det['y'] - det['h'] / 2.0
                boxes_xyxy[idx, 2] = det['x'] + det['w'] / 2.0
                boxes_xyxy[idx, 3] = det['y'] + det['h'] / 2.0
                scores[idx, :] = det['sig']
                idx+=1
    
            dets = multiclass_nms_class_aware(boxes_xyxy, scores, nms_thr=0.45, score_thr=args.score_thr)
    
            print(dets)
    
            # visualize and save
            if dets is None:
                print("no object detected.")
            else:
                det_ori_box = dets[:, :4]
                final_boxes, final_scores, final_cls_inds = det_ori_box, dets[:, 4], dets[:, 5]
                origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                                 conf=args.score_thr, class_names=_classes)
                if not args.no_torgb:
                    origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)

                os.makedirs(args.out_dir, exist_ok=True)    

                if args.all_pics:
                    output_path = os.path.join(args.out_dir, Path(filename).name)
                    cv2.imwrite(output_path, origin_img)
                else:
                    output_path = os.path.join(args.out_dir, Path(args.img).name)
                    cv2.imwrite(output_path, origin_img)
                    break
    
        else:
            #img, ratio = preprocess(origin_img, img_size)
            img, ratio = easy_preprocess(origin_img, img_size)
            img = img[np.newaxis].astype(np.float32)  # add batch dim
        
            if input_dtype == np.int8:
                #print("input int8 converting:")
                img = img / input_scale + input_zero
                img = img.astype(np.int8)
        
            # run inference
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
    
            outputs = interpreter.get_tensor(output_details[0]['index'])
            outputs = outputs[0]  # remove batch dim
    
            if output_dtype == np.int8:
                #print("output int8 converting:")
                outputs = output_scale * (outputs.astype(np.float32) - output_zero)
        
            # postprocess
            preds = postprocess(outputs, (h, w))
            boxes = preds[:, :4]
            scores = preds[:, 4:5] * preds[:, 5:]
            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
            #boxes_xyxy /= ratio
            dets = multiclass_nms_class_aware(boxes_xyxy, scores, nms_thr=0.65, score_thr=args.score_thr)
            print(dets)
        
            # visualize and save
            if dets is None:
                print("no object detected.")
            else:
    
                #dets[:, :4] = dets[:, :4]/ratio # preprocess
                dets[:, 0] = dets[:, 0]/ratio[1] # easy_preprocess
                dets[:, 2] = dets[:, 2]/ratio[1] # easy_preprocess
                dets[:, 1] = dets[:, 1]/ratio[0] # easy_preprocess
                dets[:, 3] = dets[:, 3]/ratio[0] # easy_preprocess
    
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                                 conf=args.score_thr, class_names=_classes)
                if not args.no_torgb:
                    origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)
        
                os.makedirs(args.out_dir, exist_ok=True)
                
                if args.all_pics:
                    output_path = os.path.join(args.out_dir, Path(filename).name)
                    cv2.imwrite(output_path, origin_img)
                else:
                    output_path = os.path.join(args.out_dir, Path(args.img).name)
                    cv2.imwrite(output_path, origin_img)
                    break


if __name__ == '__main__':
    main()
