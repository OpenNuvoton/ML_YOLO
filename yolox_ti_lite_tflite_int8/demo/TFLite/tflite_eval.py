"""
Module: tflite_eval
This module evaluates a TensorFlow Lite YOLOX_nano model on a COCO format dataset.
It handles preprocessing, inference, postprocessing, and COCO mAP evaluation.
"""

import argparse
import os
import time
import json
import io
import contextlib
import itertools
from tqdm import tqdm

from tabulate import tabulate

import cv2
import numpy as np
import tensorflow.lite as tflite

from utils import COCO_CLASSES, multiclass_nms_class_aware, preprocess, postprocess, easy_preprocess
from pycocotools.coco import COCO

try:
    from yolox.layers import COCOeval_opt as COCOeval
except ImportError:
    from pycocotools.cocoeval import COCOeval
    print("Use standard COCOeval.")

PER_CLASS_MAP = True

def load_cocoformat_labels(data_dir, anno_path):
    """Loads class labels from a COCO annotation file.
    Args:
        data_dir: The directory containing the COCO annotation files.
        anno_path: The path to the annotation JSON file (e.g., 'instances_val2017.json').
    Returns:
        A tuple containing the class names.  Returns None if there is an error.
    """
    anno_dir = "annotations"
    coco = COCO(os.path.join(data_dir, anno_dir, anno_path))
    cats = coco.loadCats(coco.getCatIds())
    _classes = tuple([c["name"] for c in cats])

    return _classes


def parse_args():
    """Parses command-line arguments using argparse.

    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="path to .tflite model")
    parser.add_argument("-i", "--img", help="path to image file")
    parser.add_argument("-v", "--val", default="datasets\\coco", help="path to validation dataset")
    parser.add_argument("-o", "--out-dir", default="tmp/tflite", help="path to output directory")
    parser.add_argument("-s", "--score-thr", type=float, default=0.001, help="threshould to filter by scores")
    parser.add_argument("-pp", "--preprocess-way", default="cv2", const="cv2", nargs="?", choices=["yolox", "cv2"], help="preprocess-way (default: %(default)s)")
    parser.add_argument(
        "-a",
        "--anno_file",
        type=str,
        default="medicine_val.json",
        help="Path to annotation file.",
    )
    parser.add_argument("--no_torgb", action="store_true", help="convert from BGR to RGB")
    return parser.parse_args()


class CocoFormatDataset:
    """
    A custom dataset class for loading and preprocessing COCO format data.
    Args:
        data_dir: Path to the directory containing COCO dataset images and annotations.
        anno_file: Filename of the annotation JSON file (e.g., 'instances_val2017.json').
        img_size: Tuple specifying the desired input image size for the model (height, width).  Note: This is not actually used in validation.
    Attributes:
        img_size: The target image size.
        data_dir: The root directory of the COCO dataset.
        img_dir_name: The name of the subdirectory containing images (assumed to be 'val2017').
        anno_dir: Assumed directory name for annotations within data_dir.
        gd_annotation_file: Complete path to the ground truth annotation file.
        coco: A pycocotools.coco.COCO object representing the dataset.
        img_ids: A list of image IDs in the dataset.
        class_ids: A list of sorted category IDs.
        annotations: A list of pre-loaded annotations for efficient access.
    """

    def __init__(self, data_dir="datasets\\coco_test_sp", anno_file="", img_size=(416, 416), no_torgb=False):
        self.img_size = img_size  # This val isn't used in validation
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
        """Loads annotations for a given image ID.
        Args:
            id_: The ID of the image.
        Returns:
            A tuple containing:
                - img_info: A tuple (height, width) representing the original image dimensions.
                - file_name: The filename of the image.
        """
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
        # resized_info = (int(height * r), int(width * r))

        file_name = im_ann["file_name"] if "file_name" in im_ann else f"{id_:012}" + ".jpg"

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
        """Retrieves image, filename, info, and IDs for a given index.
        Args:
            index: The index of the item to retrieve.
        Returns:
            A tuple containing:
                - img: The loaded image as a NumPy array.
                - file_name: The filename of the image.
                - img_info: A tuple (height, width) of the original image dimensions.
                - img_id: A NumPy array containing the image ID.
        """
        id_ = self.img_ids[index]

        img_info, file_name = self.annotations[index]
        img = self._load_image(index)

        return img, file_name, img_info, np.array([id_])

    def __getitem__(self, index):
        img, file_name, img_info, img_id = self.pull_item(index)
        # if self.preproc is not None:
        #    img, target = self.preproc(img, target, self.input_dim)
        return img, file_name, img_info, img_id

    def per_class_map_table(self, coco_eval, class_names=COCO_CLASSES, colums=6):
        """Generates a formatted table showing per-class mean Average Precision (mAP) scores.
        Args:
            coco_eval: A pycocotools.cocoeval.COCOeval object containing evaluation results.
            class_names: A list of class names. Defaults to COCO_CLASSES.
            headers: A list of column headers for the table. Defaults to ["class", "AP"].
            colums: The number of columns in the table. Defaults to 6.
        Returns:
            A string containing the formatted table.
        """
        headers = ["class", "AP"]
        per_class_map_dict = {}
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            per_class_map_dict[name] = float(ap * 100)

        num_cols = min(colums, len(per_class_map_dict) * len(headers))
        result_pair = [x for pair in per_class_map_dict.items() for x in pair]
        row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
        table_headers = headers * (num_cols // len(headers))
        table = tabulate(
            row_pair,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=table_headers,
            numalign="left",
        )
        return table

    def evaluate_prediction(self, data_dict, _class_names):
        """Evaluates model predictions against ground truth using COCOeval.
        Args:
            data_dict: A dictionary containing model predictions in COCO format.
            _class_names: A list of class names.
        Returns:
            A tuple containing:
                - mAP: The mean Average Precision (mAP) score.
                - mAP50: The mAP score at IoU threshold of 0.5.
                - info: A string containing evaluation summary information.
        """

        print("Evaluate in main process...")

        ann_type = ["segm", "bbox", "keypoints"]

        info = "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            coco_gt = self.coco

            json.dump(data_dict, open("./yolox_testdev_2017.json", "w", encoding="utf-8"))
            coco_dt = coco_gt.loadRes("./yolox_testdev_2017.json")

            coco_eval = COCOeval(coco_gt, coco_dt, ann_type[1])
            coco_eval.evaluate()
            coco_eval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                coco_eval.summarize()
            info += redirect_string.getvalue()
            if PER_CLASS_MAP:
                info += "per class mAP:\n" + self.per_class_map_table(coco_eval, class_names=_class_names)
            return coco_eval.stats[0], coco_eval.stats[1], info
        else:
            return 0, 0, info

    def xyxy2xywh(self, bboxes):
        """Converts bounding boxes from xyxy (x_min, y_min, x_max, y_max) to xywh (x_center, y_center, width, height) format.
        Args:
            bboxes: A NumPy array of shape (N, 4) representing bounding boxes in xyxy format.
        Returns:
            A NumPy array of shape (N, 4) representing bounding boxes in xywh format.
        """
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        return bboxes

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        """Converts model output (detections) to the COCO JSON format.
        Args:
            outputs
            info_imgs
            ids
        Returns:
            A list of dictionaries, where each dictionary represents a detection in COCO JSON format.
        """
        data_list = []
        for output, _, img_id in zip(outputs, info_imgs, ids):
            if output is None:
                continue

            bboxes = output[:, 0:4]

            bboxes = self.xyxy2xywh(bboxes)

            cls = output[:, 5]
            scores = output[:, 4]
            for ind in range(bboxes.shape[0]):
                # label = COCO_CLASSES[int(cls[ind])] # Update your class
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
    """
    Main function for evaluating a TensorFlow Lite YOLOv model on a COCO format dataset.
    This function handles argument parsing, dataset loading, model loading, inference, postprocessing, and COCO mAP evaluation.
    """
    # reference:
    # https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/demo/tflite/yolox_tflite_demo.py

    args = parse_args()

    # setup dataset
    data_list = []
    my_dataset = CocoFormatDataset(data_dir=args.val, anno_file=args.anno_file, no_torgb=args.no_torgb)

    # prepare model
    interpreter = tflite.Interpreter(model_path=(args.model.strip("'").strip("\\")))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # model info
    input_dtype = input_details[0]["dtype"]
    input_scale = input_details[0]["quantization"][0]
    input_zero = input_details[0]["quantization"][1]
    print("Model Shape: {} {} Model Dtype: {}".format(input_details[0]["shape"][1], input_details[0]["shape"][2], input_dtype))
    print("Model input Scale: {} Model input Zero point: {}".format(input_scale, input_zero))

    output_dtype = output_details[0]["dtype"]
    output_scale = output_details[0]["quantization"][0]
    output_zero = output_details[0]["quantization"][1]
    print("Model output Shape: {} {} Model output Dtype: {}".format(output_details[0]["shape"][0], output_details[0]["shape"][1], output_dtype))
    print("Model output Scale: {} Model output Zero point: {}".format(output_details[0]["quantization"][0], output_details[0]["quantization"][1]))

    input_shape = input_details[0]["shape"]
    _, h, w, _ = input_shape
    model_img_size = (h, w)

    for _, (origin_img, _, info_imgs, ids) in enumerate(tqdm(my_dataset)):
        # cur_iter, (origin_img, file_name, info_imgs, ids)
        # preprocess
        if args.preprocess_way == "cv2":
            img, ratio = easy_preprocess(origin_img, model_img_size)
        else:
            img, ratio = preprocess(origin_img, model_img_size)
        img = img[np.newaxis].astype(np.float32)  # add batch dim

        if input_dtype == np.int8:
            img = img / input_scale + input_zero
            img = img.astype(np.int8)

        # run inference
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()

        outputs = interpreter.get_tensor(output_details[0]["index"])
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
        if args.preprocess_way == "cv2":
            boxes_xyxy[:, 0] = boxes_xyxy[:, 0] / ratio[1]
            boxes_xyxy[:, 2] = boxes_xyxy[:, 2] / ratio[1]
            boxes_xyxy[:, 1] = boxes_xyxy[:, 1] / ratio[0]
            boxes_xyxy[:, 3] = boxes_xyxy[:, 3] / ratio[0]
        else:
            boxes_xyxy /= ratio

        dets = multiclass_nms_class_aware(boxes_xyxy, scores, nms_thr=0.65, score_thr=args.score_thr)

        # single coco mAP eval
        data_list.extend(my_dataset.convert_to_coco_format([dets], [info_imgs], ids))

    # coco mAP eval
    *_, summary = my_dataset.evaluate_prediction(data_list, load_cocoformat_labels(args.val, args.anno_file))
    print(summary)


if __name__ == "__main__":
    main()
