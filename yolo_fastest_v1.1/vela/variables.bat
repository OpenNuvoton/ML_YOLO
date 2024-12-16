set IMAGE_SRC_DIR=..\samples
set IMAGE_SRC_WIDTH=320
set IMAGE_SRC_HEIGHT=320

::set LABEL_SRC_FILE=..\labels\labels_mobilenet_v2_1.0_224.txt
::set GEN_LABEL_FILE_NAME=Labels

set MODEL_SRC_DIR=..\workspace\MyTask_1
set MODEL_SRC_FILE=yolo-fastest-1.1-int8.tflite
set MODEL_OPTIMISE_FILE=yolo-fastest-1.1-int8_vela.tflite
::The vela OPTIMISE_FILE should be SRC_FILE_NAME + _vela

set TEMPLATES_DIR=Tool\tflite2cpp\templates

set GEN_SRC_DIR=..\workspace\MyTask_1\vela
set GEN_INC_DIR=generated\include

::vela configure variable section

::accelerator config. ethos-u55-32, ethos-u55-64, ethos-u55-128, ethos-u55-256, ethos-u65-256, ethos-u65-512
set VELA_ACCEL_CONFIG=ethos-u55-256

::optimise option. Size, Performance
set VELA_OPTIMISE_OPTION=Performance

::configuration file
set VELA_CONFIG_FILE=Tool\vela\default_vela.ini

::system config. Selects the system configuration to use as specified in the vela configuration file
set VELA_SYS_CONFIG=Ethos_U55_High_End_Embedded

::memory mode. Selects the memory mode to use as specified in the vela configuration file
set VELA_MEM_MODE=Shared_Sram










