# YOLOX-ti-lite-nu tflite version for MCU with/wo NPU device
This repository focuses on YOLOX Nano with slight updates, which can be deployed on Nuvoton devices.   

 - Add `exps\default\yolox_nano_ti_lite_nu.py` for 320X320 depthwise YOLOX-ti-lite version.
 - Add int8/f32 tflite mAP evaluating script: `demo\TFLite\tflite_eval`. Support models: tflite_yolox_nano_ti, tflite_yolofastest_v1(mAP is lower)
 - Update yolox nano model to fully support ARM vela compiler. Detail: [Remove Transpose](https://github.com/MaxCYCHEN/yolox-ti-lite_tflite_int8/blob/main/assets/README.md)  
 - This repository is a fork of [TexasInstruments/edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox) and [motokimura/yolox-ti-lite_tflite](https://github.com/motokimura/yolox-ti-lite_tflite).
 - You can also reference the original [readme](https://github.com/MaxCYCHEN/yolox-ti-lite_tflite_int8/blob/main/README_motokimura.md) or [edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)

 ## Installation
 - Create a new python env. If you aren't familiar with python env creating, you can reference here: [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise?tab=readme-ov-file#2-installation--env-create)
 ```bash 
conda create --name yolox_nu  python=3.10
conda activate yolox_nu
```
 - upgrade pip
 ```bash 
python -m pip install --upgrade pip
```
 - setuptools should <= 75.8.0
   
**1.** Installing pytorch, basing on the type of system, CUDA version, PyTorch version, and MMCV version [pytorch_locally](https://pytorch.org/get-started/locally/)
- The below example is CUDA needed. If cpu only, please check [pytorch_locally](https://pytorch.org/get-started/locally/). 
```bash 
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
```
**2.** Installing mmcv, basing on your hardware config [Install with pip](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip)
```bash 
python -m pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```
**3.** Installing other python requirements
- download this repo and open this directory.
```bash
python -m pip install --no-input -r requirements.txt
```
**4.** Installing the YOLOX
```bash
python setup.py develop
```
- If there is an error msg says about `Microsoft Visual C++ is required`, please install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

 ## How to Use
 ### 0. UIRun notebook
 - `UIRun.ipynb` offers an easy UI to run train, convert to tflite int8/vela and test flite int8.
 - About the dataset for training, please follow Custome Train below.
 ### 1. Train
 ```bash
python tools/train.py -f <MODEL_CONFIG_FILE> -d 1 -b <BATCH_SIZE> --fp16 -o -c <PRETRAIN MODEL PATH>
```
- example:
```bash
python tools/train.py -f exps/default/yolox_nano_ti_lite_nu.py -d 1 -b 64 --fp16 -o -c pretrain/tflite_yolox_nano_ti/320_DW/yolox_nano_320_DW_ti_lite.pth
```
- Please use `exps/default/yolox_nano_ti_lite_nu.py` for yolox-nano-ti-nu model which is able running on Nuvoton devices.
- **Custome Dataset Train**:
    - dataset format: COCO json format
    - dataset download guide: [image dataset download](https://github.com/OpenNuvoton/ML_YOLO?tab=readme-ov-file#1-where-can-i-download-the-dataset)
    - The dataset structure and folders name must same as below:
    ```bash
    datasets/<dataset_name>
               |
               |----annotations
               |        |----------<train_annotation_json_file>
               |        |----------<val_annotation_json_file>
               |
               |----train2017
               |        |---------train imgs
               |
               |----val2017
                        |---------validation imgs
    ```
    - update dataset path in `exps/default/yolox_nano_ti_lite_nu.py`, for example:
    ```bash
    self.data_dir = "datasets/hagrid_coco"
    self.train_ann = "hagrid_train.json"
    self.val_ann = "hagrid_val.json" 
    ```
    - update resolution, number of classes and training epoch in `exps/default/yolox_nano_ti_lite_nu.py`, for example:
    ```bash
    self.input_size = (320, 320) # resolution
    self.test_size = (320, 320) # resolution
    self.num_classes = 11 # number of classes
    self.max_epoch = 150 # training epoch
    ```
    - (Optional) convert YOLO annotaion txt format to COCO JSON format
    ```bash
    python tools/yolo2coco.py --path <YOLO_FORMAT_DATASET>\train --output annotation_train.json
     ```
### 2. Windows Batch Script for Converting Pytorch to Deployment Format
- This script will help you finish point 3 to pint 6, and of course you can manually execute the python & cmds below.
- Update `yolox_convert`, for example:
    ```bash
    set MODEL_FILE_NAME=yolox_nano_ti_lite_nu_hg_150
    set YOLOX_M_CONFIG=exps/default/yolox_nano_ti_lite_nu.py
    set OUTPUT_DIR=YOLOX_outputs/yolox_nano_ti_lite_nu_hg_150

    set TRAIN_DATASET=datasets/hagrid_coco/train2017
    ```
- Run `yolox_convert`
### 3. Evaluate Pytorch Model (Optional)
```bash
python tools/eval.py -f <MODEL_CONFIG_FILE> -c <TRAINED_PYTORCH_MODEL> --conf 0.001
```
- example:
```bash
python tools/eval.py -f exps/default/yolox_nano_ti_lite_nu.py -c YOLOX_outputs/yolox_nano_ti_lite_nu/latest_ckpt.pth --conf 0.001
```

### 4. Pytorch to ONNX
```bash
python tools/export_onnx.py -f <MODEL_CONFIG_FILE> -c <TRAINED_PYTORCH_MODEL> --output-name <ONNX_MODEL_PATH>
```
- example:
```bash
python tools/export_onnx.py -f exps/default/yolox_nano_ti_lite_nu.py -c YOLOX_outputs/yolox_nano_ti_lite_nu/latest_ckpt.pth --output-name YOLOX_outputs/yolox_nano_ti_lite_nu/yolox_nano_nu_medicine.onnx
```

### 5. ONNX to TFlite
- Create calibration data
```bash
python demo/TFLite/generate_calib_data.py --img-size <IMG_SIZE> --n-img <NUMBER_IMG_FOR_CALI> -o <CALI_DATA_NPY_FILE> --img-dir <PATH_OF_TRAIN_IMAGE_DIR>
```
- example:
```bash
python demo/TFLite/generate_calib_data.py --img-size 320 320 --n-img 200 -o YOLOX_outputs\yolox_nano_ti_lite_nu\calib_data_320x320_n200.npy --img-dir datasets\medicine_coco\train2017
```
- Convert ONNX to TFlite
```bash
onnx2tf -i <ONNX_MODEL_PATH> -oiqt -qcind images <CALI_DATA_NPY_FILE> "[[[[0,0,0]]]]" "[[[[1,1,1]]]]"
```
- example:
```bash
onnx2tf -i YOLOX_outputs/yolox_nano_ti_lite_nu/yolox_nano_nu_medicine.onnx -oiqt -qcind images YOLOX_outputs\yolox_nano_ti_lite_nu\calib_data_320x320_n200.npy "[[[[0,0,0]]]]" "[[[[1,1,1]]]]"
```
### 6. Use Vela Compiler and Convert to Deplyment Format
- move the int8 tflite model to `vela\generated\` 
- in `vela` and update `variables.bat`
    ```bash
    set MODEL_SRC_FILE=<your tflite model>
    set MODEL_OPTIMISE_FILE=<output vela model>
    ```
    - example:
    ```bash
    set MODEL_SRC_FILE=yolox_nano_ti_lite_nu_hg_150_full_integer_quant.tflite
    set MODEL_OPTIMISE_FILE=yolox_nano_ti_lite_nu_hg_150_full_integer_quant_vela.tflite
    ```
- The output file for deplyment is `vela\generated\yolox_nano_ti_lite_nu_full_integer_quant_vela.tflite.cc`
### 7. Evaluate TFlite int8/float Model (Optional)
```bash
python demo\TFLite\tflite_eval.py -m <FULL_INTEGER_QUANT_TFLITE> -v <DATASET_PATH> -a <VAL_ANNOTATION_FILE>
```
- example:
```bash
python demo\TFLite\tflite_eval.py -m YOLOX_outputs/yolox_nano_ti_lite_nu/yolox_nano_nu_hg_full_integer_quant.tflite -v datasets\hagrid_coco -a hagrid_val.json
```
- <img src="https://github.com/MaxCYCHEN/yolox-ti-lite_tflite_int8/assets/105192502/eb0f2f47-a001-4bad-a50b-c0e5afdad87f" width="40%">

### 8. Test Singl/All Validation Images
- Provide more realistic bounding box results for validation data using the int8 TensorFlow Lite (TFLite) model.
```bash
python demo\TFLite\tflite_inference.py -m <FULL_INTEGER_QUANT_TFLITE> -s <SCORE_THR> -i <PATH_OF_IMAGE> -a <PATH_OF_VAL_ANNOTATION_FILE>
```
- `--all_pics` for test all images in  <PATH_OF_IMAGE>
- example:
```bash
python demo\TFLite\tflite_inference.py -m YOLOX_outputs\yolox_nano_ti_lite_nu\yolox_nano_ti_lite_nu_hg_full_integer_quant.tflite -s 0.6 -i datasets/hagrid_coco/val2017/0001.jpg -a hagrid_coco/annotations/hagrid_val.json
```
- The result images with bounding boxes will save in `tmp\tflite`

## Inference code
- The output file for deplyment is for example `yolox_nano_ti_lite_nu_full_integer_quant_vela.tflite.cc` and move it to `ObjectDetection_FreeRTOS_yoloxn/Model/` in bsp sample code to update new model.
- The ML_SampleCode repositories are private. Please contact Nuvoton to request access to these sample codes. [Link](https://www.nuvoton.com/ai/contact-us/)
    - [ML_M55M1_SampleCode (private repo)](https://github.com/OpenNuvoton/ML_M55M1_SampleCode)
