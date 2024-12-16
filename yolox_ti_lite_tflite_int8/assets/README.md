# Update yolo_head for ARM Vela Compiler
- Need NHWC reshape for tflite format, if not, at converting step, there will be transposes layers to change the input format.

#### Original
- pytorch => onnx:
    - <img src="onnx_ori.png" width="400" >
- onnx => tflite:
    - <img src="ori_tflite_int8.png" width="400" >
- vela:
    - <img src="ori_vela_tflite.png" width="150" >

#### Updated
- pytorch => onnx:
    - <img src="onnx.png" width="400" >
- onnx => tflite:
  - <img src="tflite_int8.png" width="400" >
- vela:
  - <img src="vela_tflite.png" width="150" >
