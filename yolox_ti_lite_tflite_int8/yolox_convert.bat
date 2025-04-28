@ ECHO off
set "POSSIBLE_LOCATIONS=C:\ProgramData\miniforge3;C:\Users\%USERNAME%\miniforge3;C:\Users\%USERNAME%\AppData\Local\miniforge3;C:\ProgramData\Miniconda3;C:\Users\%USERNAME%\Miniconda3;C:\Users\%USERNAME%\AppData\Local\Miniconda3"

for %%d in (%POSSIBLE_LOCATIONS%) do (
    if exist "%%d\Scripts\activate.bat" (
        set "MODEL_SRC_DIR=%%d"
        goto found_conda
    )
)

:found_conda
set CONDA_ENV=yolox_nu

set YOLOX_DIR=%~dp0

set IMG_SIZE=192
set NUM_CLS=80
set MODEL_FILE_NAME=yolox_n_nu_192_coco
set YOLOX_M_CONFIG=exps/default/yolox_nano_ti_lite_nu.py
set OUTPUT_DIR=YOLOX_outputs/yolox_nano_ti_lite_nu_192
set TRAIN_DATASET=datasets/coco/train2017

set YOLOX_PYTORCH=%OUTPUT_DIR%/latest_ckpt.pth
set YOLOX_ONNX=%OUTPUT_DIR%/%MODEL_FILE_NAME%.onnx
set CALIB_DATA=%OUTPUT_DIR%/calib_data_%IMG_SIZE%x%IMG_SIZE%_n200_hg.npy

call %MODEL_SRC_DIR%\Scripts\activate.bat
call conda activate %CONDA_ENV%
::call cd %~dp0
call cd %YOLOX_DIR%

::pytorch => onnx => tflite
@echo on
call python tools/export_onnx.py -f %YOLOX_M_CONFIG% -c %YOLOX_PYTORCH% --output-name %YOLOX_ONNX% -im %IMG_SIZE% -nc %NUM_CLS%
call python demo/TFLite/generate_calib_data.py --img-size %IMG_SIZE% %IMG_SIZE% --n-img 200 -o %CALIB_DATA% --img-dir %TRAIN_DATASET%
call onnx2tf -i %YOLOX_ONNX% -oiqt -qcind images %CALIB_DATA% "[[[[0,0,0]]]]" "[[[[1,1,1]]]]"
call robocopy saved_model %OUTPUT_DIR% /move

::vela
@echo off
set IMAGE_SIZE_EXPR="extern const int originalImageSize = 320"
set CHANNELS_EXPR="extern const int channelsImageDisplayed = 3"
set CLASS_EXPR="extern const int numClasses = 6"
set MODEL_TFLITE_FILE=%MODEL_FILE_NAME%_full_integer_quant.tflite
set MODEL_OPTIMISE_FILE=%MODEL_FILE_NAME%_full_integer_quant_vela.tflite
set VELA_OUTPUT_DIR=%OUTPUT_DIR%\vela
set TEMPLATES_DIR=vela\Tool\tflite2cpp\templates
::accelerator config. ethos-u55-32, ethos-u55-64, ethos-u55-128, ethos-u55-256, ethos-u65-256, ethos-u65-512
set VELA_ACCEL_CONFIG=ethos-u55-256
::optimise option. Size, Performance
set VELA_OPTIMISE_OPTION=Performance
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