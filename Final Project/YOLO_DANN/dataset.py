import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from PIL import ImageDraw
from configs import TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, VAL_IMAGES_PATH, VAL_LABELS_PATH, REAL_FRAMES_PATH

import logging
logging.basicConfig(
    filename='dataset.log',
    level=logging.INFO, 
    format='%(asctime)s - %(message)s', 
    filemode='w'
)

class YoloSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, img_size=(640, 640), real=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_size = img_size
        self.image_filenames = sorted(os.listdir(image_dir))
        self.real = real

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_filename)[0] + '.txt')

        image = Image.open(img_path).convert("RGB")
        
        mask = self.load_yolo_polygon_mask(label_path, self.img_size)

        if self.transform:
            image = self.transform(image)
            
        if self.real:
            mask[mask == 0] = 1

        return image, mask

    def load_yolo_polygon_mask(self, label_path, img_size):
        """Create a binary mask from polygon YOLO label format"""
        mask = Image.new('L', img_size, 0) 
        draw = ImageDraw.Draw(mask)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                labels = file.readlines()
                
                for label in labels:
                    data = list(map(float, label.strip().split()))
                    class_id = int(data[0])  
                    polygon_points = data[1:]
                    
                    polygon_pixel_coords = [
                        (int(x * img_size[0]), int(y * img_size[1])) 
                        for x, y in zip(polygon_points[::2], polygon_points[1::2])
                    ]
                    
                    draw.polygon(polygon_pixel_coords, outline=1, fill=1)
        
        return torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0) 


def load_synthetic_data(train_images_path, train_labels_path, val_images_path, val_labels_path, batch_size=8, real=False):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    
    train_dataset = YoloSegmentationDataset(train_images_path, train_labels_path, transform=transform, real=real)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = YoloSegmentationDataset(val_images_path, val_labels_path, transform=transform, real=real)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def extract_frames(video_path, output_folder):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(output_folder, f"frame_{count}.jpg"), image)
        success, image = vidcap.read()
        count += 1


import os
import matplotlib.pyplot as plt

def visualize_images_with_masks(images, masks, num_samples=5, save_dir='visualizations'):
    """
    Visualizes a few images with their corresponding masks and saves them as files.
    
    :param images: Tensor of images (shape: [batch_size, channels, height, width])
    :param masks: Tensor of masks (shape: [batch_size, 1, height, width])
    :param num_samples: Number of samples to visualize
    :param save_dir: Directory to save the visualizations
    """
    num_samples = min(num_samples, images.size(0))
    
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).numpy() 
        mask = masks[i][0].numpy() 
        
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Image')

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.title('Mask')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'image_mask_{i}.png'))
        plt.close()  

if __name__ == '__main__':
    train_loader, val_loader = load_synthetic_data(
        TRAIN_IMAGES_PATH, 
        TRAIN_LABELS_PATH, 
        VAL_IMAGES_PATH, 
        VAL_LABELS_PATH
    )

    first_batch = next(iter(train_loader))  # Get the first batch
    images, masks = first_batch

    logging.info(f"Images shape: {images.shape}")
    logging.info(f"Masks shape: {masks.shape}")

    logging.info(f"masks[0][0] sum: {masks[0][0].sum(1)}")

    visualize_images_with_masks(images, masks, num_samples=5, save_dir='visualizations')

    
    # train_loader = load_synthetic_data(REAL_FRAMES_PATH, REAL_FRAMES_PATH, REAL_FRAMES_PATH, REAL_FRAMES_PATH, real=True)[0]

    # # Log or print the first batch for inspection
    # first_batch = next(iter(train_loader))  # Get the first batch
    # images, masks = first_batch

    # logging.info(f"Images shape: {images.shape}")
    # logging.info(f"Masks shape: {masks.shape}")

    # logging.info(f"masks[0][0] sum: {masks[0][0].sum(1)}")
    # # masks[masks == 0] = 1
    # # logging.info(f"masks[0][0] sum: {masks[0][0].sum(1)}")
    
    
    