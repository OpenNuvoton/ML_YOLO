## Yolov8 Edge int8 tflite version for MCU with/wo NPU device
This project is from [YOLOv8 DeGirum Train](https://github.com/DeGirum/ultralytics_yolov8) and from [ultralytics](https://github.com/ultralytics/ultralytics).
DeGirum training uses ReLU6 activation to have improved model performance at edge.
Below are some instructions to train and deploy to MCU with ethous-U device. 

## Installation
 - Create a new python env. If you aren't familiar with python env creating, you can reference here: [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise?tab=readme-ov-file#2-installation--env-create)
 ```bash 
conda create --name yolov8_DG  python=3.10
conda activate yolov8_DG
```
 - upgrade pip
 ```bash 
python -m pip install --upgrade pip setuptools
```
**1.** Installing pytorch, basing on the type of system, CUDA version, PyTorch version [pytorch_locally](https://pytorch.org/get-started/locally/)
- The below example is CUDA needed. If cpu only, please check [pytorch_locally](https://pytorch.org/get-started/locally/). 
```bash 
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```
**2.** Installing the Ultralytics_yolov8
- download this repo and open this directory.
```bash
python -m pip install .[export]
```

 ## How to Use
 ### 1. Train
- example:
```bash
python dg_train.py --model-cfg relu6-yolov8.yaml --data coco.yaml --imgsz 320 --weights yolov8n.pt --batch 64 --epochs 200
```
- **Custome Dataset Train**:
    - dataset format: [Ultralytics YOLO](https://docs.ultralytics.com/zh/datasets/detect/) format
    - dataset download guide: [image dataset download](https://github.com/OpenNuvoton/ML_YOLO?tab=readme-ov-file#1-where-can-i-download-the-dataset)
    - The dataset structure and folders name must same as below:
    ```bash
    <dataset_name>
      |
      |----train.txt (save each path of train imgs)
      |----val.txt   (save each path of validation imgs)
      |
      |----train
      |        |---------train imgs
      |        |---------train labels txt file
      |
      |----val
               |---------validation imgs
               |---------validation labels txt file
    ```
    
    - update dataset path in `coco.yaml`, for example:
    ```bash
    path: <your_dataset_path>\output_yolo_dogcat  # dataset root dir
    train: train.txt
    val: val.txt
    test: val.txt
    
    # Classes
    names:
      0: cat
      1: dog
    ```
    - update number of classes in `ultralytics/cfg/models/v8/relu6-yolov8.yaml`, for example:
    ```bash
    nc: 80  # number of classes
    ```
  
    - (Optional) convert COCO JSON format to YOLO txt format
        - Follow steps in `ML_YOLO\yolo_fastest_v1.1\Yolo-Fastest-darknet\training_demo\json2txt.ipynb` or
        - Follow Ultralytics API [port-or-convert-label-formats](https://docs.ultralytics.com/zh/datasets/detect/#port-or-convert-label-formats)
    

### 2. Evaluate Pytorch Model (Optional)
- example:
```bash
python dg_val.py --weights .\runs\train\exp2\weights\best.pt --data coco.yaml --img 320
```
### 3. Windows Batch Script for Converting Pytorch to Deployment Format (tflite int8/vela) (Optional)
- This script will help you finish point 4 and pint 6, and of course you can manually execute the python & cmds below.
- 1. Update `yolov8n_convert.bat` basing on your model info, for example:
    ```bash
    set IMG_SIZE=320
    set MODEL_FILE_NAME=best
    set OUTPUT_DIR=runs/train/exp2/weights
    set TRAIN_DATASET=datasets/coco/train2017/images
    ```
- 2. Run `yolov8n_convert.bat`
### 4. Pytorch to Tflite int8
- Firstly, convert to ONNX, example:
```bash
python nu_export_tflite_int8.py --format onnx --weights .\runs\train\exp2\weights\best.pt --img 320
```

- Secondly, create calibration data, example:
```bash
python generate_calib_data.py --img-size 320 320 --n-img 200 -o calib_data_320_n200_rgb.npy --img-dir datasets\coco\train2017\images
```

- Thirdly, convert to TFLITE int8, example:
```bash
onnx2tf -i runs\train\exp2\weights\best.onnx -oiqt -cind images calib_data_320_n200_rgb.npy "[[[[0,0,0]]]]" "[[[[1,1,1]]]]"
```

### 5. Evaluate TFlite int8/float Model
- example:
```bash
python dg_val.py --weights .\runs\train\exp2\weights\best_full_integer_quant.tflite --data coco.yaml --img 320
```
- <img src="https://github.com/user-attachments/assets/e233fbf3-8dc1-4d83-9f4f-8326964aa1b9" width="40%">

### 6. Use Vela Compiler and Convert to Deplyment Format
- move the int8 tflite model to `vela\generated\` 
- in `vela` and update `variables.bat`
    ```bash
    set MODEL_SRC_FILE=<your tflite model>
    set MODEL_OPTIMISE_FILE=<output vela model>
    ```
    - example:
    ```bash
    set MODEL_SRC_FILE=yolov8n_full_integer_quant.tflite
    set MODEL_OPTIMISE_FILE=yolov8n_full_integer_quant_vela.tflite
    ```
- The output file for deplyment is `vela\generated\yolov8n_full_integer_quant_vela.tflite` and `vela\generated\yolov8n_full_integer_quant_vela.tflite.cc`

## Inference code
- The output file for deplyment is for example `vela\generated\yolov8n_full_integer_quant_vela.tflite` and move it to SD card root directory to update new model. (Please refer the below M55M1 BSP firmware.)

- The ML_SampleCode repositories are private. Please contact Nuvoton to request access to these sample codes. [Link](https://www.nuvoton.com/ai/contact-us/)
    - [ML_M55M1_SampleCode (private repo)](https://github.com/OpenNuvoton/ML_M55M1_SampleCode)




