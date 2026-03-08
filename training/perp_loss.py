"""
High Receptive Field Perceptual Loss (HRFPL) & Inpainting Loss
Pipeline Stage: 4 (Core Experiments)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG19PerceptualLoss(nn.Module):
    """
    Calculates the Perceptual Loss using a pre-trained VGG19 feature extractor.
    Forces the model to learn high-frequency textures rather than blurry averages.
    """
    def __init__(self):
        super().__init__()
        # Load the pre-trained VGG19 network
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # We extract features from these specific slices of the VGG network
        self.blocks = nn.ModuleList([
            vgg[:4],    # relu1_2
            vgg[4:9],   # relu2_2
            vgg[9:18],  # relu3_4
            vgg[18:27], # relu4_4
            vgg[27:36]  # relu5_4
        ])
        
        # Freeze the VGG network so it doesn't get updated during training
        for param in self.parameters():
            param.requires_grad = False
            
        # VGG expects images normalized to ImageNet statistics [0, 1] bounds
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # 1. Un-normalize from our dataset's [-1, 1] range to [0, 1]
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0
        
        # 2. Normalize to ImageNet statistics for VGG
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        loss = 0.0
        x, y = pred, target
        
        # 3. Pass through VGG blocks and calculate L1 distance at each stage
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
            
        return loss

class InpaintingLoss(nn.Module):
    """
    The master loss function combining pixel-perfect L1 loss with HRFPL.
    """
    def __init__(self, perceptual_weight=0.1):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGG19PerceptualLoss()
        self.perceptual_weight = perceptual_weight

    def forward(self, pred, target):
        # Base pixel-level L1 Loss
        l1 = self.l1_loss(pred, target)
        # High Receptive Field Perceptual Loss
        perceptual = self.perceptual_loss(pred, target)
        
        # Combined objective
        return l1 + (self.perceptual_weight * perceptual)
