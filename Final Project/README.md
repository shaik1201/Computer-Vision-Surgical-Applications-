# How to start:

1. Clone the repository:
   ```sh
   git clone https://github.com/shaik1201/Computer-Vision-Surgical-Applications-.git

2. cd into Final Project
    ```sh
    cd Computer-Vision-Surgical-Applications-
    cd Final Project

3. install the required packages
    ```sh
    pip install -r requirements.txt

4. To run prediction on the image uploaded to this repo, run:
    ```sh
    python predict.py

This will open a window with the tagged image. You can check it on other images by changing the variable `IMAGE_PATH` to your image path.

5. To run prediction on the short ood video uploaded to this repo, run:

    ```sh
    python video.py

This will create a file named "tagged_video.mp4". You can check it on other video by changing the variable `VIDEO_PATH` to your video path.

## generate_data
This folder contains the code for generating the synthetic data for training.

## The YOLO_DANN
This folder contains the code for creating and training the YOLO_DANN model.

## The FCN_DANN
This folder contains the code for creating and training the FCN_DANN model.

## Download the Model Weights

You can download the trained YOLO model weights from the link below:

[Download model's weights](https://github.com/shaik1201/Computer-Vision-Surgical-Applications-/blob/main/Final%20Project/models_weights/exp3_try4.pt)