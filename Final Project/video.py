import torch
import cv2
import os
from ultralytics import YOLO

MODEL_PATH = '/home/student/project/mlflow/finetuning/exp3/results/try4/weights/best.pt'

VIDEO_PATH = '/datashare/project/vids_test/4_2_24_A_1.mp4'
OUTPUT_PATH = 'tagged_videos/4_2_24_A_1.mp4'

def load_model(model_path):
    """
    Load the model using PyTorch or a YOLO-specific loader.
    Adjust this based on the specific model format used in your project.
    """
    model = YOLO(model_path)
    return model

def process_video(model, video_path, output_path, time_limit=None):
    """
    Process a video using the loaded model for segmentation.
    If time_limit is None, the whole video will be processed.
    
    Args:
        model: The loaded YOLO model.
        video_path: Path to the input video.
        output_path: Path to save the output video.
        time_limit: The time limit in seconds for processing the video (default is None, which means process the entire video).
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    frame_limit = None
    if time_limit is not None:
        frame_limit = int(time_limit * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break 

        results = model(frame)

        for result in results:
            annotated_frame = result.plot() 

            out.write(annotated_frame)

        frame_count += 1

        if frame_limit is not None and frame_count >= frame_limit:
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames. Video saved to {output_path}.")


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    
    time_limit = None

    process_video(model, VIDEO_PATH, OUTPUT_PATH, time_limit)

    print(f"Video processing completed. Segmented video saved to {OUTPUT_PATH}")