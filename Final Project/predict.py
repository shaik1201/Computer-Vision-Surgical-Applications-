import cv2
import numpy as np
import os
from ultralytics import YOLO

MODEL_PATH = '/home/student/project/mlflow/finetuning/exp7/results/surgical_instruments_lr_0.0014/weights/best.pt'
IMAGE_PATH = '/home/student/project/mlflow/synth_data/main/data/images/train/000000.png'
SEGMENTED_OUTPUT_PATH = 'tagged_images/segmented_image.png'

def load_model(model_path):
    """
    Load the model using YOLO-specific loader.
    """
    model = YOLO(model_path)
    return model

def predict_and_save_tagged_image(model, image_path, output_path):
    """
    Use the model to predict on a single image and save the tagged image with bounding boxes and labels.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return
    
    results = model.predict(image)

    tagged_image = results[0].plot()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, tagged_image)
    print(f"Tagged image saved at: {output_path}")

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    
    predict_and_save_tagged_image(model, IMAGE_PATH, SEGMENTED_OUTPUT_PATH)

