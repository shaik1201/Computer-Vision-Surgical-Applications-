import os
import shutil
from gen_pseudo_labels import gen_pseudo_labels
from train_yolo_pseudo import train_initial_model, train_pseudo_model
from augment import augment_data
from preprocess_data import gamma_correction

def make_dirs():
    source_train_images = '/home/student/HW - CV_operating_room/HW1/images/train'
    source_train_labels = '/home/student/HW - CV_operating_room/HW1/labels/train'
    source_val_images = '/home/student/HW - CV_operating_room/HW1/images/val'
    source_val_labels = '/home/student/HW - CV_operating_room/HW1/labels/val'
    
    dest_train_images = 'images/train'
    dest_train_labels = 'labels/train'
    dest_val_images = 'images/val'
    dest_val_labels = 'labels/val'
    
    try:
        shutil.copytree(source_train_images, dest_train_images)
        shutil.copytree(source_train_labels, dest_train_labels)
        shutil.copytree(source_val_images, dest_val_images)
        shutil.copytree(source_val_labels, dest_val_labels)
        print("Directories copied successfully.")
    except FileExistsError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def run_pipeline():
    make_dirs()
    gamma_correction()
    train_initial_model()
    gen_pseudo_labels()
    augment_data()
    train_pseudo_model()

run_pipeline()
