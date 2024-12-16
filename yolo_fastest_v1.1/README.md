# ML_YOLO
- This tool assists you in training the YOLO-series models using Darknet and converting them to TFLite models, which are easy to deploy on MCU/MPU devices. It also provides support for the Vela compiler for ARM NPU devices.
## 1. Yolo-Fastest-darknet
- `Yolo-Fastest-darknet/`
- This is the training model step/workfolder.
- Use Darknet to train the YOLO-Fastestv1 model, which is obtained from [dog-qiuqiu/Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest) and comes with a pre-trained model.
- How to use it, please check `Yolo-Fastest-darknet/`.
## 2. darknet_tflite
- `darknet_tflite/`
- This step involves converting the Darknet model to a TensorFlow model and then to a TFLite model.
- To handle this step, Python scripts are used, so it is necessary to have our [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) environment installed.   
## 3. vela
- `vela`
- If the user intends to deploy the model on an ARM NPU device, the Vela compiler assists in converting the TFLite model to Vela TFLite C++ source files.

# Inference code
- MCU: [M55M1](https://github.com/OpenNuvoton/M55M1BSP/tree/master/SampleCode/MachineLearning)
    - ObjectDetection_FreeRTOS/
- MPU: [MA35D1](https://github.com/OpenNuvoton/MA35D1_Linux_Applications/tree/master/machine_learning)
