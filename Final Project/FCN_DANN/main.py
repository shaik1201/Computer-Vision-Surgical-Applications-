from dataset import CombinedDataset
from config import TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, VAL_IMAGES_DIR, VAL_MASKS_DIR, EPOCHS
from model import DANN_FCN_Segmentation
import torch.nn as nn
from train_one_epoch import train_one_epoch, validate_one_epoch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

train_dataset = CombinedDataset(
    image_dir=TRAIN_IMAGES_DIR, 
    mask_dir=TRAIN_MASKS_DIR, 
    real_frames_dir='/home/student/project/mlflow/DANN/Extracted_Frames',
    transform=transform, 
    target_transform=transform, 
    image_size=(640, 640)
)

val_dataset = CombinedDataset(
    image_dir=VAL_IMAGES_DIR, 
    mask_dir=VAL_MASKS_DIR, 
    transform=transform, 
    target_transform=transform, 
    image_size=(640, 640)
)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)


model = DANN_FCN_Segmentation(n_classes=3).to(device)


weights = torch.tensor([0.05, 0.4, 0.55]).to(device)
seg_loss_fn = nn.CrossEntropyLoss(weight=weights)
domain_loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


best_vloss = float('inf')
timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    
    
    train_seg_loss, train_domain_loss, train_total_loss = train_one_epoch(
        epoch, train_dataloader, seg_loss_fn, domain_loss_fn, optimizer, model, device
    )

    
    val_total_loss, val_seg_loss, val_domain_loss = validate_one_epoch(
        val_dataloader, seg_loss_fn, domain_loss_fn, model, device
    )

   
    if val_total_loss < best_vloss:
        best_vloss = val_total_loss
        best_model_path = f"best_model_{timestamp}.pth"
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved new best model with validation loss: {best_vloss:.4f}")

    print(
        f"End of Epoch {epoch + 1}: "
        f"Train Seg Loss {train_seg_loss:.4f}, Train Domain Loss {train_domain_loss:.4f}, "
        f"Train Total Loss {train_total_loss:.4f}, "
        f"Validation Seg Loss {val_seg_loss:.4f}, Validation Domain Loss {val_domain_loss:.4f}, "
        f"Validation Total Loss {val_total_loss:.4f}"
    )

print("Training complete!")
