import argparse
import os
import time
import json
import io
import contextlib
import itertools
from tqdm import tqdm
from pathlib import Path

from tabulate import tabulate

import cv2
import numpy as np
import tensorflow.lite as tflite

from utils import COCO_CLASSES, multiclass_nms_class_aware, preprocess, postprocess, vis, yolofastest_preprocess, easy_preprocess
from pycocotools.coco import COCO

# ToDo, tmp global var
per_class_mAP = True

def load_cocoformat_labels(data_dir, anno_path):
    anno_dir = "annotations"
    coco = COCO(os.path.join(data_dir, anno_dir, anno_path))
    cats = coco.loadCats(coco.getCatIds())
    _classes = tuple([c["name"] for c in cats])

    return _classes

def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--model', required=True, help='path to .tflite model')
        parser.add_argument('-i', '--img', help='path to image file')
        parser.add_argument('-v', '--val', default='datasets\coco', help='path to validation dataset')
        #parser.add_argument('-v', '--val', default='datasets\coco_test_sp', help='path to validation dataset')
        parser.add_argument('-o', '--out-dir', default='tmp/tflite', help='path to output directory')
        parser.add_argument('-s', '--score-thr', type=float, default=0.001, help='threshould to filter by scores')
        parser.add_argument('-pp', '--preprocess-way', default='cv2', const='cv2', nargs='?',
                    choices=['yolox', 'cv2'], help='preprocess-way (default: %(default)s)')
        parser.add_argument("-a", "--anno_file", type=str, 
                            default='medicine_val.json', help="Path to annotation file.",)
        parser.add_argument("--no_torgb", action="store_true", help="convert from BGR to RGB")
        return parser.parse_args()

class coco_format_dataset():
    def __init__(
        self,
        data_dir="datasets\coco_test_sp",
        anno_file = "",
        img_size=(416, 416),
        no_torgb=False
    ):
        self.img_size = img_size # This val isn't used in validation
        self.data_dir = data_dir
        self.img_dir_name = "val2017"
        self.anno_dir = "annotations"
        self.gd_annotation_file = os.path.join(self.data_dir, self.anno_dir, anno_file)
        self.coco = COCO(self.gd_annotation_file)
        self.img_ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.annotations = self._load_coco_annotations()
        self.no_torgb = no_torgb

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.img_ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (img_info, file_name)    

    def _load_image(self, index):
        file_name = self.annotations[index][1]
        img_file = os.path.join(self.data_dir, self.img_dir_name, file_name)
        img = cv2.imread(img_file)
        if not self.no_torgb:
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert img is not None

        return img

    def pull_item(self, index):
        id_ = self.img_ids[index]

        img_info, file_name = self.annotations[index]
        img = self._load_image(index)

        return img, file_name, img_info, np.array([id_])    

    def __getitem__(self, index):
        img, file_name, img_info, img_id = self.pull_item(index)
        #if self.preproc is not None:
        #    img, target = self.preproc(img, target, self.input_dim)
        return img, file_name, img_info, img_id    
    
    def per_class_mAP_table(self, coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
        per_class_mAP = {}
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]
    
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            per_class_mAP[name] = float(ap * 100)
    
        num_cols = min(colums, len(per_class_mAP) * len(headers))
        result_pair = [x for pair in per_class_mAP.items() for x in pair]
        row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
        table_headers = headers * (num_cols // len(headers))
        table = tabulate(
            row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
        )
        return table
    
    def evaluate_prediction(self, data_dict, _class_names):
           
            print("Evaluate in main process...")
    
            annType = ["segm", "bbox", "keypoints"]
    
            info = "\n"
    
            # Evaluate the Dt (detection) json comparing with the ground truth
            if len(data_dict) > 0:
                cocoGt = self.coco
    
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            
                try:
                    from yolox.layers import COCOeval_opt as COCOeval
                except ImportError:
                    from pycocotools.cocoeval import COCOeval
    
                    print("Use standard COCOeval.")
    
                cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
                cocoEval.evaluate()
                cocoEval.accumulate()
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                info += redirect_string.getvalue()
                if per_class_mAP:
                    info += "per class mAP:\n" + self.per_class_mAP_table(cocoEval, class_names=_class_names)
                return cocoEval.stats[0], cocoEval.stats[1], info
            else:
                return 0, 0, info
    
    def xyxy2xywh(self, bboxes):
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        return bboxes
    
    def convert_to_coco_format(self, outputs, model_img_size, info_imgs, ids):
            data_list = []
            for (output, ori_img, img_id) in zip(
                outputs, info_imgs, ids
            ):
                if output is None:
                    continue
                
                bboxes = output[:, 0:4]
    
                bboxes = self.xyxy2xywh(bboxes)
    
                cls = output[:, 5]
                scores = output[:, 4]
                for ind in range(bboxes.shape[0]):
                    #label = COCO_CLASSES[int(cls[ind])] # Update your class
                    label = self.class_ids[int(cls[ind])]
                    pred_data = {
                        "image_id": int(img_id),
                        "category_id": label,
                        "bbox": bboxes[ind].tolist(),
                        "score": scores[ind].item(),
                        "segmentation": [],
                    }  # COCO json format
                    data_list.append(pred_data)

                
            return data_list


def main():
    # reference:
    # https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/demo/tflite/yolox_tflite_demo.py

    args = parse_args()

    # setup dataset
    data_list = []
    my_dataset = coco_format_dataset(data_dir=args.val, anno_file=args.anno_file, no_torgb=args.no_torgb)

    # prepare model
    interpreter = tflite.Interpreter(model_path=(args.model.strip('\'').strip('\\')))
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

    input_shape = input_details[0]['shape']
    b, h, w, c = input_shape
    model_img_size = (h, w)
    

    for cur_iter, (origin_img, file_name, info_imgs, ids) in enumerate(tqdm(my_dataset)):
        
        # preprocess
        if (args.preprocess_way == 'cv2'):
            img, ratio = easy_preprocess(origin_img, model_img_size)
        else:    
            img, ratio = preprocess(origin_img, model_img_size)
        img = img[np.newaxis].astype(np.float32)  # add batch dim
        
        if input_dtype == np.int8:
            img = img / input_scale + input_zero
            img = img.astype(np.int8)
        
        # run inference
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
    
        outputs = interpreter.get_tensor(output_details[0]['index'])
        outputs = outputs[0]  # remove batch dim
    
        if output_dtype == np.int8:
            outputs = output_scale * (outputs.astype(np.float32) - output_zero)
        inference_time = (time.perf_counter() - start) * 1000 
    
        # postprocess
        preds = postprocess(outputs, (h, w))
        boxes = preds[:, :4]
        scores = preds[:, 4:5] * preds[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0

        # resize to original
        if (args.preprocess_way == 'cv2'):
            boxes_xyxy[:, 0] = boxes_xyxy[:, 0]/ratio[1]
            boxes_xyxy[:, 2] = boxes_xyxy[:, 2]/ratio[1]
            boxes_xyxy[:, 1] = boxes_xyxy[:, 1]/ratio[0]
            boxes_xyxy[:, 3] = boxes_xyxy[:, 3]/ratio[0]
        else:    
            boxes_xyxy /= ratio
        
        dets = multiclass_nms_class_aware(boxes_xyxy, scores, nms_thr=0.65, score_thr=args.score_thr)

        # single coco mAP eval
        data_list.extend(my_dataset.convert_to_coco_format([dets], model_img_size, [info_imgs], ids))

    # coco mAP eval
    *_, summary = my_dataset.evaluate_prediction(data_list, load_cocoformat_labels(args.val, args.anno_file))
    print(summary)

if __name__ == '__main__':
    main()
