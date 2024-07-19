# How to start:

1. Clone the repository:
   ```sh
   git clone https://github.com/shaik1201/Computer-Vision-Surgical-Applications-.git

2. cd into HW1
    ```sh
    `cd Computer-Vision-Surgical-Applications/HW1`

3. install the requoired packages
    ```sh
    `pip install -r requirements.txt`

4. To run prediction on the image uploaded to this repo, run::
    ```sh
    `python predict.py`

This will open a window with the tagged image. You can check it on other images by changing the variable `img_path` to your image path.

5. To run prediction on the short ood video uploaded to this repo, run:

    ```sh
    `python video.py`

This will create a file named "tagged_video.mp4". You can check it on other video by changing the variable `video_path` to your video path.

## Download the Model Weights

You can download the trained YOLO model weights from the link below:

[Download model's weights](https://github.com/shaik1201/Computer-Vision-Surgical-Applications-/raw/main/HW1/best.pt)
