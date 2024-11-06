import os
import shutil
import json
import random

# Paths
dataset_dirs = [d for d in os.listdir('/home/student/project/mlflow/synth_data/main')
                if os.path.isdir(os.path.join('/home/student/project/mlflow/synth_data/main', d)) and d.endswith('_output') and 'T' not in d]
output_dir = 'data'
images_dir = os.path.join(output_dir, 'images')
annotations_dir = os.path.join(output_dir, 'annotations')

os.makedirs(os.path.join(images_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(images_dir, 'val'), exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# Initialize counters and data holders
image_id = 0
annotation_id = 0
train_ratio = 0.8  # Using all data for training in this case

train_images = []
train_annotations = []
val_images = []
val_annotations = []

# Define the categories explicitly
categories = [
    {'id': 0, 'name': 'NH'},
    {'id': 1, 'name': 'T'}
]

# Initialize a cumulative id_mapping outside the loop
id_mapping = {}

for dataset_dir in dataset_dirs:
    # Determine the category_id based on the directory name
    # if dataset_dir.startswith('NH'):
    #     category_id = 0
    # elif dataset_dir.startswith('T'):
    #     category_id = 1
    # else:
    #     raise ValueError(f"Unexpected dataset directory name: {dataset_dir}")

    # Load annotations/home/student/project/dataset/images/train
    with open(os.path.join('/home/student/project/mlflow/synth_data/main', dataset_dir, 'coco_data/coco_annotations.json'), 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    # Shuffle images
    random.shuffle(images)
    num_train = int(len(images) * train_ratio)
    train_imgs = images[:num_train]
    val_imgs = images[num_train:]

    # Process training images
    for img in train_imgs:
        old_id = img['id']
        original_file_name = img['file_name']  # Store the original file name

        new_filename = f'{image_id:06d}.png'
        img['file_name'] = os.path.join('train', new_filename)  # Update for new dataset
        img['id'] = image_id
        id_mapping[(dataset_dir, old_id)] = image_id  # Use dataset_dir to make keys unique
        image_id += 1

        # Copy image file
        src = os.path.join('/home/student/project/mlflow/synth_data/main',
                           dataset_dir, 'coco_data/images', os.path.basename(original_file_name))
        dst = os.path.join(images_dir, 'train', new_filename)
        shutil.copy(src, dst)

        train_images.append(img)

    # Process validation images (will be empty if train_ratio == 1)
    for img in val_imgs:
        old_id = img['id']
        original_file_name = img['file_name']

        new_filename = f'{image_id:06d}.png'
        img['file_name'] = os.path.join('val', new_filename)
        img['id'] = image_id
        id_mapping[(dataset_dir, old_id)] = image_id
        image_id += 1

        # Copy image file
        src = os.path.join('/home/student/project/mlflow/synth_data/main',
                           dataset_dir, 'coco_data/images', os.path.basename(original_file_name))
        dst = os.path.join(images_dir, 'val', new_filename)
        shutil.copy(src, dst)

        val_images.append(img)

    # Update annotations
    for ann in annotations:
        ann['id'] = annotation_id
        annotation_id += 1

        # Use the composite key to get the correct mapping
        ann['image_id'] = id_mapping[(dataset_dir, ann['image_id'])]

        # Update the category_id based on the directory
        # if ann['category_id'] == 1:
        #     ann['category_id'] = 0
        # elif ann['category_id'] == 2:
        #     ann['category_id'] = 1

        if ann['image_id'] in [img['id'] for img in train_images]:
            train_annotations.append(ann)
        else:
            val_annotations.append(ann)

# Save merged annotations
train_data = {
    'images': train_images,
    'annotations': train_annotations,
    'categories': categories
}
val_data = {
    'images': val_images,
    'annotations': val_annotations,
    'categories': categories
}

with open(os.path.join(annotations_dir, 'instances_train.json'), 'w') as f:
    json.dump(train_data, f)
with open(os.path.join(annotations_dir, 'instances_val.json'), 'w') as f:
    json.dump(val_data, f)
