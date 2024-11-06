import os
import json
import shutil

# Paths
dataset_dir = 'data'
images_dir = os.path.join(dataset_dir, 'images')
annotations_dir = os.path.join(dataset_dir, 'annotations')
labels_dir = os.path.join(dataset_dir, 'labels')

os.makedirs(os.path.join(labels_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(labels_dir, 'val'), exist_ok=True)

# Function to convert segmentation to YOLO format
def convert_segmentation_to_yolo(anns, img_info, label_path):
    img_width = img_info['width']
    img_height = img_info['height']
    with open(label_path, 'w') as label_file:
        for ann in anns:
            class_id = ann['category_id']  # Adjust if necessary
            segmentation = ann['segmentation']
            if isinstance(segmentation, list):
                # Handle multiple segmentations (polygons)
                for seg in segmentation:
                    normalized_seg = []
                    for i in range(0, len(seg), 2):
                        x = seg[i] / img_width
                        y = seg[i+1] / img_height
                        normalized_seg.extend([x, y])
                    # Write to label file
                    label_line = f"{class_id} " + " ".join(map(str, normalized_seg))
                    label_file.write(label_line + '\n')
            else:
                # Handle RLE format if necessary
                print(f"Skipping RLE segmentation for image ID {img_info['id']}.")
                continue

# Load train and val annotations
for split in ['train', 'val']:
    annotation_file = os.path.join(annotations_dir, f'instances_{split}.json')
    if not os.path.exists(annotation_file):
        continue  # Skip if the file doesn't exist
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    images = {img['id']: img for img in data['images']}
    annotations = data['annotations']
    img_id_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    # Process each image
    for img_id, anns in img_id_to_anns.items():
        img_info = images[img_id]
        img_filename = os.path.basename(img_info['file_name'])
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, split, label_filename)
        
        # Convert and save segmentation annotations to YOLO format
        convert_segmentation_to_yolo(anns, img_info, label_path)
