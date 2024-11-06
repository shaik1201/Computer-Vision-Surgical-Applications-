import torch
import torch.optim as optim
import logging

logging.basicConfig(
    filename='utils.log',  
    level=logging.INFO,  
    format='%(asctime)s - %(message)s', 
    filemode='w'  
)

def initialize_optimizer(model, lr=1e-4):
    return optim.Adam(model.parameters(), lr=lr)

def initialize_loss_functions():
    criterion_segmentation = torch.nn.BCELoss() 
    criterion_domain = torch.nn.BCELoss()  
    return criterion_segmentation, criterion_domain


def train_one_epoch(model, train_loader, real_loader, criterion_segmentation, criterion_domain, optimizer, device, epoch, lambda_factor=0.01, save_path="best_model.pt"):
    
    log_model_info(model)
    
    model.domain_classifier.train()
    for layer in model.yolo_model.modules():
        layer.training = True
    
    epoch_seg_loss = 0.0
    epoch_domain_loss_syn = 0.0
    epoch_domain_loss_real = 0.0
    best_total_loss = float('inf')  
    
    for batch_idx, syn_batch in enumerate(train_loader):
        syn_images, syn_labels = syn_batch
        syn_images = syn_images.to(device)
        syn_labels = syn_labels.to(device)
        
        logging.info(f'Batch {batch_idx + 1}/{len(train_loader)} - syn_labels shape: {syn_labels.shape}')

        seg_output, domain_output = model(syn_images)
        
        logging.info(f'domain_output shape: {domain_output.shape}')
        
        all_masks = []
        for i in seg_output:
            sum_tensor = torch.zeros((1, 640, 640), device=i.masks.data.device)
            for j in i.masks.data:
                sum_tensor += j
            combined_mask = (sum_tensor > 0).float()
            all_masks.append(combined_mask)
        
        all_masks_tensor = torch.stack(all_masks)
        logging.info(f'Generated combined masks with shape: {all_masks_tensor.shape}')
        
        seg_loss = criterion_segmentation(all_masks_tensor, syn_labels)
        epoch_seg_loss += seg_loss.item()
        
        domain_labels = torch.zeros(len(syn_images), 1).to(device)
        domain_loss_syn = criterion_domain(domain_output, domain_labels)
        epoch_domain_loss_syn += domain_loss_syn.item()
        
        total_loss = seg_loss + lambda_factor * domain_loss_syn
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        logging.info(f'Batch {batch_idx + 1}/{len(train_loader)} - seg_loss: {seg_loss.item():.4f}, domain_loss_syn: {domain_loss_syn.item():.4f}, total_loss: {total_loss.item():.4f}')
    
    avg_seg_loss = epoch_seg_loss / len(train_loader)
    avg_domain_loss_syn = epoch_domain_loss_syn / len(train_loader)
    logging.info(f'Average seg_loss for synthetic data: {avg_seg_loss:.4f}')
    logging.info(f'Average domain_loss_syn for synthetic data: {avg_domain_loss_syn:.4f}')
    
    if batch_idx % 10 == 0:
        logging.info(f'Batch {batch_idx}/{len(train_loader)} - '
                    f'seg_loss: {seg_loss.item():.4f}, '
                    f'domain_loss_syn: {domain_loss_syn.item():.4f}')
    
    for batch_idx, real_batch in enumerate(real_loader):
        real_images, _ = real_batch
        real_images = real_images.to(device)

        _, domain_output = model(real_images)
        
        domain_labels = torch.ones(len(real_images), 1).to(device)
        domain_loss_real = criterion_domain(domain_output, domain_labels)
        epoch_domain_loss_real += domain_loss_real.item()
        
        optimizer.zero_grad()
        domain_loss_real.backward()
        optimizer.step()
        
        logging.info(f'Real Batch {batch_idx + 1}/{len(real_loader)} - domain_loss_real: {domain_loss_real.item():.4f}')
        
    avg_domain_loss_real = epoch_domain_loss_real / len(real_loader)
    logging.info(f'Average domain_loss_real for real data: {avg_domain_loss_real:.4f}')
    
    total_epoch_loss = avg_seg_loss + lambda_factor * (avg_domain_loss_syn + avg_domain_loss_real)
    logging.info(f'Total epoch loss: {total_epoch_loss:.4f}')
    
    if total_epoch_loss < best_total_loss:
        best_total_loss = total_epoch_loss
        torch.save(model, save_path)
        logging.info(f"New best model saved with loss {best_total_loss:.4f} at {save_path}")
    else:
        logging.info(f"No improvement in total loss: {total_epoch_loss:.4f} (Best: {best_total_loss:.4f})")


def log_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info("Model Structure:")
    logging.info(model)
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Trainable parameters: {trainable_params}")
