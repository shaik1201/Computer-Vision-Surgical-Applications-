# configs.py
# Paths
MODEL_PATH = "/home/student/project/mlflow/finetuning/exp7/results/surgical_instruments_lr_0.0014/weights/best.pt"
TRAIN_IMAGES_PATH = "/home/student/project/mlflow/synth_data/main/data/images/train"
TRAIN_LABELS_PATH = "/home/student/project/mlflow/synth_data/main/data/labels/train"
VAL_IMAGES_PATH = "/home/student/project/mlflow/synth_data/main/data/images/val"
VAL_LABELS_PATH = "/home/student/project/mlflow/synth_data/main/data/labels/val"
REAL_VIDEO_PATH = "/datashare/project/vids_tune/4_2_24_B_2.mp4"
REAL_FRAMES_PATH = "./Extracted_Frames"

# Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 2
LAMBDA_FACTOR = 0.1
