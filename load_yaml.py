# -*- coding:utf-8 -*-
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    yaml_path = 'yolov8_lcnet_075_musimam.yaml'
    model = YOLO(yaml_path,)
    # print(model)

if __name__ == '__main__':
    main()
