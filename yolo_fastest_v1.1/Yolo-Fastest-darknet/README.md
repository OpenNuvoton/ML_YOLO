## Yolo-Fastest-darknet
- This is training model step/workfolder.
- Use the darknet to train the yolo-fastestv1. It is from [dog-qiuqiu/Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest) and have pre-train model.
-  We offer an easy UI notebook `yolo-fastest-usage.ipynb` for training, testing and deployment of tflite.

## Build
- The built exe `darknet.exe` or `darknet_no_gpu.exe` are in the folders already and we offer some maybe missing dlls. However, if it doesn't work or using different GPU(RTX 3070~3090/ RTX 40 series), the rebuild is needed.
- The build steps is from here [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet). Some issue you may encounter is here `Yolo-Fastest-darknet/training_demo/darknet_cpu_install_win10.txt` or `darknet_gpu_install_win10.txt`. 

## Usage
#### Prepare the dataset
- The dataset format aligns with Darknet Yolo, where each image corresponds to a .txt label file. The label file format is also based on Darknet Yolo's data set label format: "category cx cy w h".
```
0 0.468977 0.514602 0.470129 0.714609
0 0.477576 0.572844 0.427842 0.750625
```
- *(important)* User should create the dataset following the directory as below:
```
  Your Dataset Path
  ├── coco.names            # category label file
  ├── train                 # train dataset
  │   ├── 000001.jpg
  │   ├── 000001.txt
  │   ├── 000002.jpg
  │   ├── 000002.txt
  │   ├── 000003.jpg
  │   └── 000003.txt
  ├── train.txt              # save each train data(image) path
  ├── val                    # validation dataset
  │   ├── 000057.jpg
  │   ├── 000057.txt
  │   ├── 000087.jpg
  │   ├── 000087.txt
  │   ├── 000098.jpg
  │   └── 000098.txt
  └── val.txt                # save each validation data(image) path

  ```
- The image and its corresponding *.txt label file have the same name and are store in `train/` or `val/`. 
- Examples of the dataset path files, such as `train.txt` and `val.txt`, are as follows:
- We offer a script to generate `train.txt` and `val.txt` basing on path of saving train and validation data. `ML_YOLO/Yolo-Fastest-darknet/training_demo/json2txt.ipynb`.
train.txt
  ```
  C:\Users\USER\Desktop\Yolo-Fastest-darknet\training_demo\output_yolo_person\train\000000000036.jpg
  C:\Users\USER\Desktop\Yolo-Fastest-darknet\training_demo\output_yolo_person\train\000000000049.jpg
  C:\Users\USER\Desktop\Yolo-Fastest-darknet\training_demo\output_yolo_person\train\000000000061.jpg
  ```
  val.txt
  ```
  C:\Users\USER\Desktop\Yolo-Fastest-darknet\training_demo\output_yolo_person\val\000000000139.jpg
  C:\Users\USER\Desktop\Yolo-Fastest-darknet\training_demo\output_yolo_person\val\000000000785.jpg
  C:\Users\USER\Desktop\Yolo-Fastest-darknet\training_demo\output_yolo_person\val\000000000872.jpg
  ```
- Example of the category label file, such as `coco.names`, is as follows:
coco.names
  ```
  person
  bicycle
  car
  motorbike
  ...
  
  ```
- This tool also provides an example notebook for converting the COCO dataset format to YOLO format. You can find it at `ML_YOLO/Yolo-Fastest-darknet/training_demo/json2txt.ipynb`.

#### Train
- Please open `ML_YOLO/yolo-fastest-usage.ipynb`. Begin by entering your project name and selecting the path of the dataset you created in the previous step. Next, click on 'Prepare for Training.' This UI notebook assists in finalizing various settings, such as calculating anchors for the dataset, preparing the pretrain model, and updating the YOLO training configuration.
- (Alternatively way:) Users can reference and directly employ the Darknet YOLO commands found in `ML_YOLO/cmd_ML_YOLO.txt`.

#### Convert of tflite
- `ML_YOLO/yolo-fastest-usage.ipynb` also help you finish this converting.
- (Alternatively way:) Users can refer to and directly utilize the Python script commands located in `ML_YOLO/cmd_ML_YOLO.txt`.

#### Convert from tflite to tflite vela
- `ML_YOLO/yolo-fastest-usage.ipynb` also help you finish this converting.
- (Alternatively way:) Users have the ability to update `vela/variables.bat` based on their individual files and preferences

## Cite as
dog-qiuqiu. (2021, July 24). dog-qiuqiu/Yolo-Fastest: 
yolo-fastest-v1.1.0 (Version v.1.1.0). Zenodo. 
http://doi.org/10.5281/zenodo.5131532
