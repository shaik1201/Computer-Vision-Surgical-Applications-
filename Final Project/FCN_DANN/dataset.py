import os
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import logging
torch.set_printoptions(threshold=224*224*2)

log_file_path = "dataset.log"
logging.basicConfig(
    filename=log_file_path,
    filemode='w',
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)

class CombinedDataset(Dataset):
    def __init__(self, image_dir, mask_dir, real_frames_dir, transform=None, target_transform=None, image_size=(640, 640)):
        """
        Args:
            image_dir (str): Directory with all the images.
            mask_dir (str): Directory with all the mask labels in YOLO polygon format.
            transform (callable, optional): Transform to apply to the images.
            target_transform (callable, optional): Transform to apply to the masks.
            image_size (tuple): The size (width, height) to resize the mask to.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.real_frames_dir = real_frames_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.real_frames_files = sorted(os.listdir(real_frames_dir))
        self.image_size = image_size

    def __len__(self):
        return len(self.image_files) + len(self.real_frames_files)

    def __getitem__(self, idx):
        logging.info(f'len(self.image_files): {len(self.image_files)}')
        logging.info(f'len(self.real_frames_files): {len(self.real_frames_files)}')
        logging.info(f'idx: {idx}')

        if idx < len(self.image_files):  # synth
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            image = Image.open(img_path).convert("RGB")

            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
            mask = self.create_mask_from_yolo(mask_path)

            if self.transform:
                image = self.transform(image)

            mask = np.array(mask, dtype=np.int64) 
            mask = torch.from_numpy(mask)  
            mask = mask.unsqueeze(0) 

            domain_labels = torch.zeros(1)

            return image, mask, domain_labels, 'synth'

        else:  # real
            real_idx = idx - len(self.image_files)
            img_path = os.path.join(self.real_frames_dir, self.real_frames_files[real_idx])
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            mask = torch.zeros((1, *self.image_size), dtype=torch.float32) 

            domain_labels = torch.ones(1)

            return image, mask, domain_labels, 'real'


    def create_mask_from_yolo(self, mask_path):
        mask = Image.new('L', self.image_size, 0) 
        draw = ImageDraw.Draw(mask)

        with open(mask_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0]) 
                
                polygon = [(float(parts[i]) * self.image_size[0], float(parts[i + 1]) * self.image_size[1])
                            for i in range(1, len(parts), 2)]

                fill_value = class_id + 1 

                draw.polygon(polygon, outline=fill_value, fill=fill_value)

        return mask


    
def plot_mask(data):
    output_dir = 'output_samples'
    os.makedirs(output_dir, exist_ok=True)

    images, masks, domain_labels, synth_or_real = data

    for i in range(len(images)):
        image_path = os.path.join(output_dir, f"image_{i}.png")
        save_image(images[i], image_path)

        mask_path = os.path.join(output_dir, f"mask_{i}.png")
        save_image(masks[i].float(), mask_path) 
        
        mask_np = masks[i].cpu().numpy()

        unique_values, counts = np.unique(mask_np, return_counts=True)
        pixel_info = dict(zip(unique_values, counts))

        logging.info(f"Mask {i} - Unique pixel values and their counts: {pixel_info}")

        print(f"Mask {i} - Unique pixel values and their counts: {pixel_info}")

    logging.info(f"Saved images and masks to '{output_dir}'")


if __name__ == '__main__':
    image_dir = '/home/student/project/mlflow/synth_data/main/data/images/train'
    mask_dir = '/home/student/project/mlflow/synth_data/main/data/labels/train'

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    dataset = CombinedDataset(image_dir=image_dir, mask_dir=mask_dir, real_frames_dir='/home/student/project/mlflow/DANN/Extracted_Frames',
                              transform=transform, target_transform=transform)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for data in dataloader:
        images, labels, domain_labels, synth_or_real = data
        logging.info(f'images shape: {images.shape}')
        logging.info(f'labels shape: {labels.shape}')
        # logging.info(f'labels: {labels[0][0]}')
        logging.info(f'domain_labels shape: {domain_labels.shape}')
        logging.info(f'domain_labels: {domain_labels}')
        logging.info(f'synth_or_real: {synth_or_real}')
        plot_mask(data)
        
        break
        

logging.info(f"Logging complete! Output saved to {log_file_path}")
