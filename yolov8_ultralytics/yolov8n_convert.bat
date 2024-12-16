@ ECHO off
::set below vals to yours
set MODEL_SRC_DIR=C:\ProgramData\miniforge3
set CONDA_ENV=yolov8_DG

set YOLO_DIR=%~dp0

::set below vals to yours 
set IMG_SIZE=320
set NUM_CLS=80
set MODEL_FILE_NAME=best
set OUTPUT_DIR=runs/train/exp2/weights
set TRAIN_DATASET=../yolox-ti-lite_tflite_int8/datasets/coco/train2017/images
set DATA_YAML=coco.yaml

set YOLO_PYTORCH=%OUTPUT_DIR%/%MODEL_FILE_NAME%.pt
set YOLO_ONNX=%OUTPUT_DIR%/%MODEL_FILE_NAME%.onnx
set CALIB_DATA=%OUTPUT_DIR%/calib_data_%IMG_SIZE%x%IMG_SIZE%_n200_hg.npy

call %MODEL_SRC_DIR%\Scripts\activate.bat
call conda activate %CONDA_ENV%
::call cd %~dp0
call cd %YOLO_DIR%

::pytorch => onnx => tflite
@echo on
call python nu_export_tflite_int8.py --format onnx --weights %YOLO_PYTORCH% --img %IMG_SIZE%
call python generate_calib_data.py --img-size %IMG_SIZE% %IMG_SIZE% --n-img 200 -o %CALIB_DATA% --img-dir %TRAIN_DATASET%
call onnx2tf -i %YOLO_ONNX% -oiqt -cind images %CALIB_DATA% "[[[[0,0,0]]]]" "[[[[1,1,1]]]]"
call robocopy saved_model %OUTPUT_DIR% /move

::vela
@echo off
set IMAGE_SIZE_EXPR="extern const int originalImageSize = %IMG_SIZE%"
set CHANNELS_EXPR="extern const int channelsImageDisplayed = 3"
set CLASS_EXPR="extern const int numClasses = %NUM_CLS%"
set MODEL_TFLITE_FILE=%MODEL_FILE_NAME%_full_integer_quant.tflite
set MODEL_OPTIMISE_FILE=%MODEL_FILE_NAME%_full_integer_quant_vela.tflite
set VELA_OUTPUT_DIR=%OUTPUT_DIR%\vela
set TEMPLATES_DIR=vela\Tool\tflite2cpp\templates
::accelerator config. ethos-u55-32, ethos-u55-64, ethos-u55-128, ethos-u55-256, ethos-u65-256, ethos-u65-512
set VELA_ACCEL_CONFIG=ethos-u55-256
::optimise option. Size, Performance
set VELA_OPTIMISE_OPTION=Size
::configuration file
set VELA_CONFIG_FILE=vela\Tool\vela\default_vela.ini
::memory mode. Selects the memory mode to use as specified in the vela configuration file
set VELA_MEM_MODE=Shared_Sram
::system config. Selects the system configuration to use as specified in the vela configuration file
set VELA_SYS_CONFIG=Ethos_U55_High_End_Embedded

set vela_argu= %OUTPUT_DIR%\%MODEL_TFLITE_FILE% --accelerator-config=%VELA_ACCEL_CONFIG% --optimise %VELA_OPTIMISE_OPTION% --config %VELA_CONFIG_FILE% --memory-mode=%VELA_MEM_MODE% --system-config=%VELA_SYS_CONFIG% --output-dir=%VELA_OUTPUT_DIR%
set model_argu= --tflite_path %VELA_OUTPUT_DIR%\%MODEL_OPTIMISE_FILE% --output_dir %VELA_OUTPUT_DIR% --template_dir %TEMPLATES_DIR% -ns arm -ns app -ns yoloxnanonu -e %IMAGE_SIZE_EXPR% -e %CHANNELS_EXPR% -e %CLASS_EXPR%

if not exist "%VELA_OUTPUT_DIR%" (
    echo Folder does not exist. Creating folder...
    mkdir "%VELA_OUTPUT_DIR%"
    echo Folder created successfully.
) else (
    echo Folder already exists.
)

@echo on
vela\Tool\vela\vela-3_10_0.exe %vela_argu%
vela\Tool\tflite2cpp\gen_model_cpp.exe %model_argu%

pause