#!/bin/bash
# reference:
# https://github.com/DeNA/PyTorch_YOLOv3/blob/1a3cd6e465db0b67ab783d92076c6ebfae9359a2/requirements/getcoco.sh

mkdir -p datasets/COCO
cd datasets/COCO

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

rm -f train2017.zip
rm -f val2017.zip
rm -f annotations_trainval2017.zip
