import os
import cv2
import numpy as np

def gamma_correction(gamma=0.6):
    print('Applying Gamma Correction...')
    img_dir = '/home/student/HW - CV_operating_room/HW1/pseudo_labels/images/train'
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    for img_file in os.listdir(img_dir):
        if img_file.endswith(".jpg") or img_file.endswith(".png"):  # Add other extensions if needed
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img_corrected = cv2.LUT(img, table)
                cv2.imwrite(img_path, img_corrected)
                print(f"Processed and saved: {img_file}")