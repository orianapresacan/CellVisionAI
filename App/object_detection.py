from ultralytics import YOLO
import os
import shutil
from utils import resource_path


def read_bounding_boxes(file_path, image_width, image_height):
    boxes = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            _, xc, yc, w, h = map(float, line.split())
            x = int((xc - w / 2) * image_width)
            y = int((yc - h / 2) * image_height)
            w = int(w * image_width)
            h = int(h * image_height)
            boxes.append((x, y, w, h))
    return boxes


def get_detections(filePath):
    model = YOLO(resource_path('checkpoints/best_yolo8x.pt'))
    results = model(source=filePath, conf=0.65, save=True, imgsz=2048, save_txt=True, save_crop=False, show_labels=False, show_conf=True)

    source_folder = 'runs/detect/predict/labels'
    if os.path.exists(source_folder) and os.listdir(source_folder):
        for filename in os.listdir(source_folder):
            src_file = os.path.join(source_folder, filename)
            dst_file = os.path.join('.', filename)  
            shutil.move(src_file, dst_file)

    shutil.rmtree('runs/detect/predict/labels', ignore_errors=True)
    shutil.rmtree('runs/detect/predict', ignore_errors=True)
    shutil.rmtree('runs/detect', ignore_errors=True)
    shutil.rmtree('runs', ignore_errors=True)