import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function


class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, lambda_grl=1.0):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_grl, None

class DANN_FCN_Segmentation(nn.Module):
    def __init__(self, n_classes, lambda_grl=1.0):
        super(DANN_FCN_Segmentation, self).__init__()
        
        self.encoder = models.segmentation.fcn_resnet50(pretrained=True)
        self.encoder.classifier[4] = nn.Conv2d(512, n_classes, kernel_size=1)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.domain_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.lambda_grl = lambda_grl
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

    def forward(self, x, alpha=1.0):
        
        features = self.encoder.backbone(x)['out'] 
        seg_output = self.encoder.classifier(features)
        seg_output = self.upsample(seg_output)
        pooled_features = self.adaptive_pool(features) 
        reversed_features = GradientReversalLayer.apply(pooled_features, alpha * self.lambda_grl)
        domain_output = self.domain_classifier(reversed_features)
        
        return seg_output, domain_output 