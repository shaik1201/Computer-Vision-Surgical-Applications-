import torch
from model import YOLOv8_DANN, load_pretrained_model
from dataset import load_synthetic_data, extract_frames
from utils import initialize_optimizer, initialize_loss_functions, train_one_epoch
from configs import MODEL_PATH, TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH,\
    VAL_IMAGES_PATH, VAL_LABELS_PATH, REAL_VIDEO_PATH, REAL_FRAMES_PATH, BATCH_SIZE, LEARNING_RATE, EPOCHS

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    yolo_model = load_pretrained_model(MODEL_PATH)
    model = YOLOv8_DANN(yolo_model, in_features=80)  # Adjust in_features based on YOLOv8 output size
    model.to(device)
    train_loader, val_loader = load_synthetic_data(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, VAL_IMAGES_PATH, VAL_LABELS_PATH, real=False)

    real_loader = load_synthetic_data(REAL_FRAMES_PATH, REAL_FRAMES_PATH, REAL_FRAMES_PATH, REAL_FRAMES_PATH, real=True)[0]
    
    optimizer = initialize_optimizer(model, LEARNING_RATE)
    criterion_segmentation, criterion_domain = initialize_loss_functions()

    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, real_loader, criterion_segmentation, criterion_domain, optimizer, device, epoch)

if __name__ == "__main__":
    main()
