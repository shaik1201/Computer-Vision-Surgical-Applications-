import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt


def display_image(lines, img, title="Image", cmap=None):

    for line in lines:
        print(line)
        parts = line.strip().split()
        label = int(parts[0])
        if label == 0:
            label = 'Empty'
        elif label == 1:
            label = 'Tweezers'
        elif label == 2:
            label = 'Needle_driver'
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        img_height, img_width, _ = img.shape
        x1 = int((x_center - width/2) * img_width)
        y1 = int((y_center - height/2) * img_height)
        x2 = int((x_center + width/2) * img_width)
        y2 = int((y_center + height/2) * img_height)
        
        color = (0, 255, 0)  # Green for label 0, adjust as needed
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)

    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    plt.close() 
    

model = YOLO("best.pt")
img_path = 'images/train/db41f653-output_0103.png'
labels_path = 'labels/train/db41f653-output_0103.txt'
with open(labels_path, 'r') as f:
    lines = f.readlines()
    
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
display_image(lines, img)
