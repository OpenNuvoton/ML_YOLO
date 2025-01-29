# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Nuvoton update 2024
"""
Utilities for image preprocessing and postprocessing in object detection.

This module provides functions for:

- Preprocessing images for different object detection models (YOLOFastest, easy, generic).
- Postprocessing object detection outputs, including non-maximum suppression (NMS).
- Visualizing detection results on images.
- Defining COCO class names and colors for visualization.

"""

import cv2
import numpy as np
from PIL import Image


def yolofastest_preprocess(ori_img, input_size):
    """Preprocesses an image for YOLOFastest model.

    Args:
        ori_img: The original image as a NumPy array.
        input_size: The desired input size as a tuple (width, height).

    Returns:
        A NumPy array representing the preprocessed image.
    """
    ori_img = Image.fromarray(ori_img)
    ori_img.thumbnail(input_size)

    delta_w = abs(input_size[0] - ori_img.size[0])
    delta_h = abs(input_size[1] - ori_img.size[1])
    resized_image = Image.new("RGB", input_size, (255, 255, 255, 0))

    resized_image.paste(ori_img, (int(delta_w / 2), int(delta_h / 2)))
    # resized_image.show()

    rgb_data = np.array(resized_image, dtype=np.float32)
    # cv2.imwrite(r'test_pre.jpg', rgb_data)

    _ = min(input_size[0] / rgb_data.shape[0], input_size[1] / rgb_data.shape[1])

    return rgb_data


def easy_preprocess(img, input_size):
    """Preprocesses an image using simple resizing.

    Args:
        img: The input image as a NumPy array.  Expected to be in BGR format if from cv2.
        input_size: The desired input size as a tuple (width, height).

    Returns:
        A tuple containing:
            - resized_img: The resized image as a NumPy array.  Type is np.float32.
            - r: A tuple representing the resize ratio (width_ratio, height_ratio).
    """

    resized_img = (
        cv2.resize(
            img,
            (int(input_size[0]), int(input_size[1])),
            interpolation=cv2.INTER_LINEAR,
        )
        .astype(np.uint8)
        .astype(np.float32)
    )

    r = (input_size[0] / img.shape[0], input_size[1] / img.shape[1])

    return resized_img, r


def preprocess(img, input_size):
    """Preprocesses an image with padding to maintain aspect ratio.
    This function resizes the input image to fit within the specified input size while maintaining the original aspect ratio.
    It then pads the resized image with a constant value (114) to create a final image of the exact input size.
    Args:
        img: The input image as a NumPy array.  Should be in a format acceptable by cv2.resize (e.g., BGR from cv2).
        input_size: The desired input size as a tuple (height, width).
    Returns:
        A tuple containing:
    """

    # reference:
    # https://github.com/TexasInstruments/edgeai-yolox/blob/b6a825e1a0105acbbefc0f24770082e3f1f1c320/yolox/data/data_augment.py#L174

    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)

    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.astype(np.float32)
    # cv2.imwrite(r'C:\Users\USER\Desktop\ML\yolox-ti-lite_tflite\tmp\tflite\test_pre_cv2.jpg', padded_img)

    return padded_img, r


def postprocess(outputs, img_size, p6=False):
    """Postprocesses object detection outputs.
    This function transforms raw model outputs into bounding box coordinates and confidence scores.  It handles different model configurations (with or without a P6 layer).
    Args:
        outputs: The raw outputs from the object detection model.  The exact format depends on the model, but it's expected to contain encoded bounding box information and class probabilities.
        img_size: The original image size as a tuple (height, width).  Used to scale the bounding boxes back to the original image coordinates.
        p6: A boolean indicating whether the model uses a P6 layer (higher resolution feature map).  This affects the stride values used for calculation.
    Returns:
        A NumPy array representing the processed detection output.  The array likely contains bounding box coordinates (x1, y1, x2, y2), confidence scores, and class IDs.
    """
    # reference:
    # https://github.com/TexasInstruments/edgeai-yolox/blob/b6a825e1a0105acbbefc0f24770082e3f1f1c320/yolox/utils/demo_utils.py#L99

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2)
        # print("grid before: {}".format(grid.shape))
        grid = grid.reshape(1, -1, 2)
        # print("grid after: {}".format(grid.shape))
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    # print(grids.shape)
    # print(expanded_strides.shape)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    # reference:
    # https://github.com/TexasInstruments/edgeai-yolox/blob/b6a825e1a0105acbbefc0f24770082e3f1f1c320/yolox/utils/demo_utils.py#L17

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    # reference:
    # https://github.com/TexasInstruments/edgeai-yolox/blob/b6a825e1a0105acbbefc0f24770082e3f1f1c320/yolox/utils/demo_utils.py#L56

    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    """Visualizes bounding boxes and labels on an image.
    This function draws bounding boxes around detected objects and displays their class labels and confidence scores on the image.
    Args:
        img: The input image as a NumPy array (presumably BGR format from OpenCV).
        boxes: A NumPy array of bounding box coordinates (x1, y1, x2, y2).
        scores: A NumPy array of detection scores (confidence scores).
        cls_ids: A NumPy array of class IDs.
        conf: The minimum confidence score threshold to display a detection.  Boxes with scores below this threshold are ignored. Defaults to 0.5.
        class_names:  A list of class names. If provided, object labels will use these names.  Otherwise, class IDs will be used.
    Returns:
        The image with bounding boxes and labels drawn on it.
    """
    # reference:
    # https://github.com/TexasInstruments/edgeai-yolox/blob/b6a825e1a0105acbbefc0f24770082e3f1f1c320/yolox/utils/visualize.py#L11

    for i, box in enumerate(boxes):
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score > conf:
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = f"{class_names[cls_id]}:{score * 100:.1f}%"
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), txt_bk_color, -1)
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def yolofastest_postprocess(outputs, anchor, class_num, input_size, ori_img_size, pre_way, threshold) -> list:
    """Postprocesses the output of a YOLO-Fastest object detection model.
    This function takes the raw output from a YOLO-Fastest model and converts it into a list of detection dictionaries.
    Each dictionary represents a detected object and contains its bounding box coordinates, confidence score, and class probabilities.
    """
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    detection_res = []
    num_anchor = int(len(anchor) / 2)
    resolution_height = outputs.shape[0]
    resolution_width = outputs.shape[1]
    per_resolution_info = outputs.shape[2]

    # 4 + 85n
    # objectness = sigmoid(outputs[:,:,[4, 89, 174]])
    objectness = sigmoid(outputs[:, :, [x for x in range(4, int(per_resolution_info), int(per_resolution_info / num_anchor))]])
    bbox_x = outputs[:, :, [x for x in range(0, int(per_resolution_info), int(per_resolution_info / num_anchor))]]
    bbox_y = outputs[:, :, [x for x in range(1, int(per_resolution_info), int(per_resolution_info / num_anchor))]]
    bbox_w = outputs[:, :, [x for x in range(2, int(per_resolution_info), int(per_resolution_info / num_anchor))]]
    bbox_h = outputs[:, :, [x for x in range(3, int(per_resolution_info), int(per_resolution_info / num_anchor))]]

    mask_matrix = objectness > threshold

    for h in range(resolution_height):
        for w in range(resolution_width):
            for anc in range(num_anchor):
                if mask_matrix[h, w, anc]:

                    # print("h: {} w: {} anc: {} obj: {}".format(h, w, anc, objectness[h, w, anc]))
                    det = {}
                    det["objectness"] = objectness[h, w, anc]

                    det["x"] = (sigmoid(bbox_x[h, w, anc]) + w) / resolution_width
                    det["y"] = (sigmoid(bbox_y[h, w, anc]) + h) / resolution_height
                    det["w"] = (np.exp(bbox_w[h, w, anc]) * anchor[anc * 2]) / input_size[1]
                    det["h"] = (np.exp(bbox_h[h, w, anc]) * anchor[anc * 2 + 1]) / input_size[0]

                    # for s in class_num:
                    sig = sigmoid(outputs[h, w, 5 + (class_num + 5) * anc : (class_num + 5) + (class_num + 5) * anc]) * det["objectness"]
                    mask_classes = (sig > threshold).astype(int)
                    det["sig"] = sig * mask_classes
                    # det['sig'] = np.sort(sig)
                    # print("{}, {}, {}: {}".format(h, w, anc, sig))

                    # resize to original
                    if pre_way == "cv2":
                        det["x"] *= ori_img_size[1]
                        det["y"] *= ori_img_size[0]
                        det["w"] *= ori_img_size[1]
                        det["h"] *= ori_img_size[0]
                    else:
                        if (input_size[0] / ori_img_size[0]) < (input_size[1] / ori_img_size[1]):
                            det["x"] *= ori_img_size[0]
                            det["y"] *= ori_img_size[0]
                            det["w"] = det["w"] * ori_img_size[0] * input_size[1] / input_size[0]
                            det["h"] = det["h"] * ori_img_size[0]
                        else:
                            det["x"] *= ori_img_size[1]
                            det["y"] *= ori_img_size[1]
                            det["w"] = det["w"] * ori_img_size[1]
                            det["h"] = det["h"] * ori_img_size[1] * input_size[0] / input_size[1]
                    detection_res.append(det)

    return detection_res


# reference:
# https://github.com/TexasInstruments/edgeai-yolox/blob/b6a825e1a0105acbbefc0f24770082e3f1f1c320/yolox/data/datasets/coco_classes.py#L5
COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


# reference:
# https://github.com/TexasInstruments/edgeai-yolox/blob/b6a825e1a0105acbbefc0f24770082e3f1f1c320/yolox/utils/visualize.py#L45
_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)
