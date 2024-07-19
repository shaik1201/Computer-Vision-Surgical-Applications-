import cv2
from ultralytics import YOLO
import os
import numpy as np

def gamma_correction(img, gamma=0.6):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def gen_pseudo_labels():
    print('Generating pseudo labels...')
    # Initialize the YOLO model
    model = YOLO("/home/student/HW - CV_operating_room/HW1/pseudo_labels/runs/detect/train/weights/best.pt")

    # Path to the video file
    video_path = '/datashare/HW1/id_video_data/4_2_24_B_2.mp4'
    cap = cv2.VideoCapture(video_path)
    CONF_THRESHOLD = 0.3
    FRAME_INTERVAL = 50

    # Output directories
    output_dir = "./"
    frames_dir = os.path.join(output_dir, "images", "train")
    labels_dir = os.path.join(output_dir, "labels", "train")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    frame_count = 0
    sampled_frame_count = 0

    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = gamma_correction(frame)

        height, width = frame.shape[:2]

        if frame_count % FRAME_INTERVAL == 0:
            # Process sampled frame
            results = model(frame)

            # Flag to check if any detection meets the confidence threshold
            frame_saved = False

            for result in results:
                xywh = result.boxes.xywh
                confs = result.boxes.conf
                classes = result.boxes.cls

                # Save frames and labels above threshold
                for box, conf, cls in zip(xywh, confs, classes):
                    if conf >= CONF_THRESHOLD:
                        if not frame_saved:
                            frame_path = os.path.join(frames_dir, f"pseudo_frame_{sampled_frame_count}.jpg")
                            cv2.imwrite(frame_path, frame)
                            frame_saved = True
                            
                        x_center, y_center, box_width, box_height = box

                        # Normalize the coordinates
                        x_center /= width
                        y_center /= height
                        box_width /= width
                        box_height /= height

                        label_path = os.path.join(labels_dir, f"pseudo_frame_{sampled_frame_count}.txt")
                        with open(label_path, 'a') as f:
                            f.write(f"{int(cls.item())} {x_center} {y_center} {box_width} {box_height}\n")
                            
                        sampled_frame_count += 1
                        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
