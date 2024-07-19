import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

def display_image(lines, img, title="Image", cmap=None):
    for line in lines:
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

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    plt.close() 
    

model = YOLO("best.pt")
img_path = '1c0b1584-frame_1789.jpg'
    
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

lines = []
results = model(img)

for result in results:
    xywh = result.boxes.xywh
    confs = result.boxes.conf
    classes = result.boxes.cls
    
    for box, conf, cls in zip(xywh, confs, classes):
        x_center = box[0].item()
        y_center = box[1].item()
        box_width = box[2].item()
        box_height = box[3].item()
        
        x_center /= img.shape[1]
        y_center /= img.shape[0]
        box_width /= img.shape[1]
        box_height /= img.shape[0]
        
        lines.append(f"{int(cls.item())} {x_center} {y_center} {box_width} {box_height}")

display_image(lines, img)
