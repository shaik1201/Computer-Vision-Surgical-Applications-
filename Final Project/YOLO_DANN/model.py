import torch
import torch.nn as nn
from ultralytics import YOLO
from torch.autograd import Function
from copy import deepcopy
import traceback


import logging
logging.basicConfig(
    filename='model.log', 
    level=logging.INFO,
    format='%(asctime)s - %(message)s', 
    filemode='w'
)

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super(GRL, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)

class DomainClassifier(nn.Module):
    def __init__(self, in_features=None):
        super(DomainClassifier, self).__init__()
        self.fc1 = None 
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1) 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 256).to(x.device) 
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class YOLOv8_DANN(nn.Module):
    def __init__(self, yolo_segmentation_model, in_features):
        super(YOLOv8_DANN, self).__init__()
        self.yolo_model = yolo_segmentation_model
        self.grl = GRL(alpha=1.0)
        self.domain_classifier = DomainClassifier(in_features)
        self._init_weights()
        
    

    def extract_features(self, model, img, layer_index=21):
        self.domain_classifier.eval() 
        for layer in self.yolo_model.modules():
            layer.training = False
        def hook_fn(module, input, output):
            global features
            features = output
            
        hook = model.model.model[layer_index].register_forward_hook(hook_fn)
        print(hook)
        with torch.no_grad():
            model(img)
        hook.remove()
        return features
        
    def _init_weights(self):
        """Ensure all weights are trainable"""
        for param in self.yolo_model.parameters():
            param.requires_grad = True



    def forward(self, x):
        logging.info(f'Input x.shape: {x.shape}')
        
        features = self.extract_features(self.yolo_model, x, 15)
        logging.info(f'Extracted features shape: {features.shape}')
        
        seg_output = self.yolo_model(x)
        logging.info(f'seg_output type: {type(seg_output)}')
        logging.info(f'seg_output len: {len(seg_output)}')
        
        try:
            reversed_features = self.grl(features.clone())
            logging.info(f'Reversed features shape before domain classification: {reversed_features.shape}')
            
            batch_size = reversed_features.shape[0]
            flattened_features = reversed_features.view(batch_size, -1) 
            logging.info(f'Flattened features shape: {flattened_features.shape}')
            
            domain_output = self.domain_classifier(flattened_features)
            return seg_output, domain_output
        
        except Exception as e:
            logging.error(f'Error occurred during forward pass: {str(e)}')
            logging.error(f'Input x shape: {x.shape}')
            logging.error(f'Features shape: {features.shape}')
            if 'reversed_features' in locals():
                logging.error(f'Reversed features shape: {reversed_features.shape}')
            
            logging.error(traceback.format_exc())
            
            raise 



def load_pretrained_model(model_path):
    model = YOLO(model_path)
    return model
