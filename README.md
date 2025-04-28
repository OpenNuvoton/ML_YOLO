# ML_YOLO
- This page guides you in training YOLO-series models using PyTorch or Darknet and converting them into fully quantized TFLite models for easy deployment on MCU/MPU devices. It also includes support for the Vela compiler for ARM NPU devices.

## 1. Choose the models
- There are three models available for deploying your customized object detection models on Nuvoton MCUs.
-
| Model | Training Framwork |Int8 Full Quantization TFLite| Folder |Description |
| :-- | :-- | :-- | :-- |:-- |
| Yolo Fastest v1.1  | Darknet | :heavy_check_mark: | [yolo_fastest_v1.1](https://github.com/OpenNuvoton/ML_YOLO/tree/master/yolo_fastest_v1.1) |([readme](https://github.com/OpenNuvoton/ML_YOLO/tree/master/yolo_fastest_v1.1#readme))|
| YoloX-nano | PyTorch | :heavy_check_mark: | [yolox_ti_lite_tflite_int8](https://github.com/OpenNuvoton/ML_YOLO/tree/master/yolox_ti_lite_tflite_int8) |Some model updates have been made to improve the accuracy of quantized models. Please check the folder link for more details. ([readme](https://github.com/OpenNuvoton/ML_YOLO/tree/master/yolov8_ultralytics#readme))|
| Yolov8-nano | PyTorch | :heavy_check_mark: | [yolov8_ultralytics](https://github.com/OpenNuvoton/ML_YOLO/tree/master/yolov8_ultralytics) |Some model updates have been made to enhance the performance of quantized models. Please check the folder link for more details. ([readme](https://github.com/OpenNuvoton/ML_YOLO/tree/master/yolox_ti_lite_tflite_int8#readme))|

## 2. Model Comparison
- Users can select models based on their application usage scenarios.
- The performance below is generated using the ARM Vela Compiler v3.10 and includes only model inference on the NPU. The mAP values are based on the COCO val2017 dataset and validation scripts for the TFLite INT8 model.
- <img src="assets/Object Detection on Edge SRAM vs mAP50.png" width="600">
- <img src="assets/Object Detection on Edge Flash vs mAP50.png" width="600">
- <img src="assets/Object Detection on Edge FPS(vela) vs mAP50.png" width="600">
   
# Inference code
- Yolo_fastest_v1.1
    - MCU: [M55M1BSP](https://github.com/OpenNuvoton/M55M1BSP/tree/master/SampleCode/MachineLearning)
        - ObjectDetection_FreeRTOS/
    - MPU: [MA35D1](https://github.com/OpenNuvoton/MA35D1_Linux_Applications/tree/master/machine_learning)

- YoloX-nano/ Yolov8-nano
    - The ML_SampleCode repositories are private. Please contact Nuvoton to request access to these sample codes. [Link](https://www.nuvoton.com/ai/contact-us/)
    - MCU: [ML_M55M1_SampleCode (private repo)](https://github.com/OpenNuvoton/ML_M55M1_SampleCode)
        - ObjectDetection_FreeRTOS_yoloxn/
        - ObjectDetection_YOLOv8n/
     
# Q&A
### 1. Where can I download the dataset?
- COCO-2017
   - There are several ways to download it. Here are some links where you can download it manually.
   - [kaggle](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset), [cocodataset.org](https://cocodataset.org/#download)
- Other example dataset
   - [roboflow dataset](https://public.roboflow.com/object-detection/), [kaggle](https://www.kaggle.com/datasets)

### 2. The annotation format
- Model:
    - Yolo Fastest v1.1: YOLO txt format
    - YoloX-nano: COCO format
    - Yolov8-nano: Ultralytics YOLO txt format
- Script:
    - YOLO to COCO: `yolox_ti_lite_tflite_int8/tools/yolo2coco.py`
    - COCO to YOLO: `ML_YOLO/yolo_fastest_v1.1/Yolo-Fastest-darknet/training_demo/json2txt.ipynb`

