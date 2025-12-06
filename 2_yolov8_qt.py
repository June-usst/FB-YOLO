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

# Parse command line arguments for detection parameters
parser = argparse.ArgumentParser()
parser.add_argument('--weights', default=r"weights/7_lcnet_sedc_eneiou/weights/best.pt", type=str,
                    help='Path to model weights')
parser.add_argument('--conf_thre', type=float, default=0.2, help='Confidence threshold')
parser.add_argument('--iou_thre', type=float, default=0.5, help='IoU threshold')
opt = parser.parse_args()

# Use GPU if available, otherwise use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_color(idx):
    """
    Generate a distinct color based on index

    Args:
        idx: Integer index for color generation

    Returns:
        Tuple of (B, G, R) color values
    """
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


class Detector(object):
    """
    YOLO-based object detector class
    """

    def __init__(self, weight_path, conf_threshold=0.5, iou_threshold=0.5):
        """
        Initialize detector with model and parameters

        Args:
            weight_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
        """
        self.device = device
        self.model = YOLO(weight_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.names = self.model.names  # Class names

    def detect_image(self, img_bgr):
        """
        Perform object detection on input image

        Args:
            img_bgr: Input image in BGR format

        Returns:
            Image with bounding boxes and labels drawn
        """
        # Run YOLO inference
        results = self.model(img_bgr, verbose=True, conf=self.conf_threshold,
                             iou=self.iou_threshold, device=self.device)

        # Extract detection results
        bboxes_cls = results[0].boxes.cls
        bboxes_conf = results[0].boxes.conf
        bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')

        # Draw bounding boxes and labels
        for idx in range(len(bboxes_cls)):
            box_conf = f"{bboxes_conf[idx]:.2f}"
            box_cls = int(bboxes_cls[idx])
            bbox_xyxy = bboxes_xyxy[idx]
            bbox_label = self.names[box_cls]
            xmin, ymin, xmax, ymax = bbox_xyxy

            # Draw rectangle
            img_bgr = cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax),
                                    get_color(box_cls + 2), 2)

            # Draw label with confidence
            cv2.putText(img_bgr, f'{str(bbox_label)}/{str(box_conf)}',
                        (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        get_color(box_cls + 2), 2)

        return img_bgr


class MainWindow(QMainWindow):
    """
    Main GUI window for YOLO detector application
    """

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("YOLO Detector")
        self.setFixedSize(900, 700)  # Fixed window size
        self.center_window()

        # Initialize detector with command line arguments
        self.detector = Detector(weight_path=opt.weights,
                                 conf_threshold=opt.conf_thre,
                                 iou_threshold=opt.iou_thre)

        # Setup image display label
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 3px solid #4A90E2;
                border-radius: 20px;
                padding: 15px;
            }
        """)

        # Create control buttons
        self.select_image_button = QPushButton("Image Detection", self)
        self.select_image_button.clicked.connect(self.open_image)

        self.video_button = QPushButton("Video Detection", self)
        self.video_button.clicked.connect(self.toggle_video_detection)

        self.camera_button = QPushButton("Camera Detection", self)
        self.camera_button.clicked.connect(self.toggle_camera_detection)

        # State variables for video and camera detection
        self.video_active = False
        self.camera_active = False

        # Apply button styling
        self.set_button_style(self.select_image_button, "#4A90E2")
        self.set_button_style(self.video_button, "#8E44AD")
        self.set_button_style(self.camera_button, "#E67E22")

        # Setup layout
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

    def set_button_style(self, button, color):
        """
        Apply custom styling to buttons

        Args:
            button: QPushButton object to style
            color: Background color in hex format
        """
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

    def center_window(self):
        """Center the window on the screen"""
        screen = QApplication.desktop().screenGeometry()
        size = self.geometry()
        self.move(int((screen.width() - size.width()) / 2),
                  int((screen.height() - size.height()) / 2))

    def open_image(self):
        """Open and process an image file"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                   "Images (*.png *.xpm *.jpg)", options=options)
        if file_name:
            # Read and process image
            img_bgr = cv2.imread(file_name)
            img_bgr = self.detector.detect_image(img_bgr)
            self.display_image(img_bgr)

    def toggle_video_detection(self):
        """Toggle video detection on/off"""
        if self.video_active:
            self.video_active = False
            self.video_button.setText("Video Detection")
        else:
            self.video_active = True
            self.video_button.setText("Stop Detection")
            self.detect_video()

    def toggle_camera_detection(self):
        """Toggle camera detection on/off"""
        if self.camera_active:
            self.camera_active = False
            self.camera_button.setText("Camera Detection")
        else:
            self.camera_active = True
            self.camera_button.setText("Stop Camera")
            self.detect_camera()

    def detect_video(self):
        """Process video file for object detection"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video", "",
                                                   "Videos (*.mp4 *.avi *.mov)", options=options)
        if file_name:
            cap = cv2.VideoCapture(file_name)
            while self.video_active and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process each frame
                frame = self.detector.detect_image(frame)
                self.display_image(frame)

                # Allow GUI to update
                QApplication.processEvents()
                time.sleep(0.03)  # Control playback speed

            cap.release()

    def detect_camera(self):
        """Process live camera feed for object detection"""
        cap = cv2.VideoCapture(0)  # Default camera

        while self.camera_active:
            ret, frame = cap.read()
            if not ret:
                break

            # Process each frame
            frame = self.detector.detect_image(frame)
            self.display_image(frame)

            # Allow GUI to update
            QApplication.processEvents()
            time.sleep(0.03)  # Control frame rate

        cap.release()

    def display_image(self, img_bgr):
        """
        Display image in the GUI

        Args:
            img_bgr: Image in BGR format to display
        """
        # Convert BGR to RGB for Qt display
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w

        # Create QImage and display
        qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Scale and display image while maintaining aspect ratio
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())