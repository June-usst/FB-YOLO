# -*- coding:utf-8 -*-
from ultralytics import YOLO
import os
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Training and Evaluation Script")
    parser.add_argument('--weights', type=str, default=r'D:\Jun\yolov8lj\yaml\yolov8_lcnet_sedccs.yaml',
                        help="Path to the model")
    parser.add_argument('--eneiou', type=str, default=True, help='Whether to use eneiou or not (True/False)')
    parser.add_argument('--mode', type=str, choices=['train', 'val'], default='train', help="Mode: train or val")
    parser.add_argument('--data', type=str, default='data.yaml', help="Data configuration file")
    parser.add_argument('--epoch', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch', type=int, default=32, help="Batch size")
    parser.add_argument('--workers', type=int, default=8, help="Number of data loading workers")
    parser.add_argument('--device', type=str, default='0', help="Device to run on, e.g., '0' for GPU 0")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 'train':
        model = YOLO(args.weights)
        model.train(data=args.data, epochs=args.epoch, batch=args.batch, workers=args.workers,
                    device=args.device,eneiou=args.eneiou,name='AC')  # 训练模型


    else:
        batch = args.batch * 2
        model = YOLO(args.weights)
        model.val(data=args.data, batch=batch, workers=args.workers, device=args.device)


if __name__ == '__main__':
    main()
