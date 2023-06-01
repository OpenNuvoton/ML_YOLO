# ML_yolo
- This Tool help you training the yolo-series model with darknet, converting to tflite model which is easy to delpoy on MCU/MPU, 
and final supporting vela compiler for ARM NPU device.
## Yolo-Fastest-darknet
- `Yolo-Fastest-darknet/`
- This is training model step/workfolder.
- Use the darknet to train the yolo-fastestv1. It is from [dog-qiuqiu/Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest) and have pre-train model.
- How to use it, please check `Yolo-Fastest-darknet/`
## darknet_tflite
- `darknet_tflite/`
- This is a step to convert darknet model to tensorflow model and tflite model.
- Use python scripts to handle this step, so it needs our [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) env installed.  
## vela
- `vela`
- If user wants to deploy on ARM NPU device, the vela compiler help you convert tflite to vela tflite C++ source file.

# Inference code
- MCU: [M55+NPU](https://github.com/chchen59/M55A1BSP)  (not yet released)
- MPU: [MA35D1](https://github.com/OpenNuvoton/MA35D1_Linux_Applications/tree/master/machine_learning)
