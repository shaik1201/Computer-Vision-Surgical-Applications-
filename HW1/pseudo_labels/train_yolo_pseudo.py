import torch
from ultralytics import YOLO
import os

def train_initial_model():
    print('Training the initial model...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    model = YOLO("/home/student/HW - CV_operating_room/HW1/yolov8n.pt")

    data_yaml_path = os.path.abspath("data.yaml")

    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"'{data_yaml_path}' does not exist")

    model.train(task='detect', data=data_yaml_path, epochs=25, imgsz=640, device=device, fliplr=0)

def train_pseudo_model():
    print('Training the pseudo model...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    model = YOLO("/home/student/HW - CV_operating_room/HW1/pseudo_labels/runs/detect/train5/weights/best.pt")

    data_yaml_path = os.path.abspath("data.yaml")

    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"'{data_yaml_path}' does not exist")

    model.train(task='detect', data=data_yaml_path, epochs=50, imgsz=640, device=device, fliplr=0)
