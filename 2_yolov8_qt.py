import sys
import argparse
import cv2
import os
import time
from ultralytics import YOLO
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, \
    QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
# # 设置 Qt 平台插件的路径
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms'

# 解析检测参数
parser = argparse.ArgumentParser()
parser.add_argument('--weights', default=r"weights/7_lcnet_sedc_eneiou/weights/best.pt", type=str, help='模型权重路径')
parser.add_argument('--conf_thre', type=float, default=0.2, help='置信度阈值')
parser.add_argument('--iou_thre', type=float, default=0.5, help='IoU阈值')
opt = parser.parse_args()

# 使用GPU还是CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 获取颜色
def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


# 检测器类
class Detector(object):
    def __init__(self, weight_path, conf_threshold=0.5, iou_threshold=0.5):
        self.device = device
        self.model = YOLO(weight_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.names = self.model.names

    def detect_image(self, img_bgr):
        results = self.model(img_bgr, verbose=True, conf=self.conf_threshold, iou=self.iou_threshold,
                             device=self.device)
        bboxes_cls = results[0].boxes.cls
        bboxes_conf = results[0].boxes.conf
        bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')

        for idx in range(len(bboxes_cls)):
            box_conf = f"{bboxes_conf[idx]:.2f}"
            box_cls = int(bboxes_cls[idx])
            bbox_xyxy = bboxes_xyxy[idx]
            bbox_label = self.names[box_cls]
            xmin, ymin, xmax, ymax = bbox_xyxy
            img_bgr = cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), get_color(box_cls + 2), 2)
            cv2.putText(img_bgr, f'{str(bbox_label)}/{str(box_conf)}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        get_color(box_cls + 2), 2)
        return img_bgr


# 主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("YOLO检测器")
        self.setFixedSize(900, 700)  # 固定窗口大小
        self.center_window()

        # 初始化检测器
        self.detector = Detector(weight_path=opt.weights, conf_threshold=opt.conf_thre, iou_threshold=opt.iou_thre)

        # 设置主界面布局
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 3px solid #4A90E2;
                border-radius: 20px;
                padding: 15px;
            }
        """)

        # 创建按钮
        self.select_image_button = QPushButton("图片检测", self)
        self.select_image_button.clicked.connect(self.open_image)

        self.video_button = QPushButton("视频检测", self)
        self.video_button.clicked.connect(self.toggle_video_detection)

        self.camera_button = QPushButton("摄像头检测", self)
        self.camera_button.clicked.connect(self.toggle_camera_detection)

        self.video_active = False
        self.camera_active = False

        # 优化按钮样式
        self.set_button_style(self.select_image_button, "#4A90E2")
        self.set_button_style(self.video_button, "#8E44AD")
        self.set_button_style(self.camera_button, "#E67E22")

        # 布局管理
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_image_button)
        button_layout.addWidget(self.video_button)
        button_layout.addWidget(self.camera_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # 优化按钮样式
    def set_button_style(self, button, color):
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 15px;
                padding: 12px;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: #2980b9;
            }}
        """)

    # 居中窗口
    def center_window(self):
        screen = QApplication.desktop().screenGeometry()
        size = self.geometry()
        self.move(int((screen.width() - size.width()) / 2), int((screen.height() - size.height()) / 2))

    # 打开图片并进行检测
    def open_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "图片检测", "", "Images (*.png *.xpm *.jpg)", options=options)
        if file_name:
            img_bgr = cv2.imread(file_name)
            img_bgr = self.detector.detect_image(img_bgr)
            self.display_image(img_bgr)

    # 切换视频检测
    def toggle_video_detection(self):
        if self.video_active:
            self.video_active = False
            self.video_button.setText("视频检测")
        else:
            self.video_active = True
            self.video_button.setText("停止检测")
            self.detect_video()

    # 切换摄像头检测
    def toggle_camera_detection(self):
        if self.camera_active:
            self.camera_active = False
            self.camera_button.setText("摄像头检测")
        else:
            self.camera_active = True
            self.camera_button.setText("停止摄像头")
            self.detect_camera()

    # 视频检测
    def detect_video(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mov)", options=options)
        if file_name:
            cap = cv2.VideoCapture(file_name)
            while self.video_active and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.detector.detect_image(frame)
                self.display_image(frame)
                QApplication.processEvents()
                time.sleep(0.03)
            cap.release()

    # 摄像头检测
    def detect_camera(self):
        cap = cv2.VideoCapture(0)  # 默认摄像头
        while self.camera_active:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.detector.detect_image(frame)
            self.display_image(frame)
            QApplication.processEvents()
            time.sleep(0.03)
        cap.release()

    # 显示图片
    def display_image(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
