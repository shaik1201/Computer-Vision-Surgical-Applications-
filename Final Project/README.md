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

4. To run prediction on the image, run:
    ```sh
    python predict.py

You can check it on other images by changing the variable `IMAGE_PATH` to your image path.

5. To run prediction on a video, run:

    ```sh
    python video.py

You can check it on other video by changing the variable `VIDEO_PATH` to your video path.

## The generate_data folder
This folder contains the code for generating the synthetic data for training.

## The YOLO_DANN folder
This folder contains the code for creating and training the YOLO_DANN model.

## The FCN_DANN folder
This folder contains the code for creating and training the FCN_DANN model.

## Download the Model Weights

You can download the trained YOLO model weights from the link below:

[Download model's weights](https://github.com/shaik1201/Computer-Vision-Surgical-Applications-/blob/main/Final%20Project/models_weights/exp3_try4.pt)

NOTE: the model weights are the ones before the domain adaptation since the results for the domain adaptation were bad, we decided to submit the pre-adaptation weights.