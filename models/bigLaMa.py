"""
BigLaMa Architecture (ResNet-18 Backbone Pivot)
Pipeline Stage: Architecture Setup & Transfer Learning
"""
import os
import logging
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def fetch_pretrained_lama(target_dir="models/weights"):
    """
    Stub kept for compatibility with pipeline scripts. 
    ImageNet weights now load automatically via torchvision.
    """
    logging.info("Using PyTorch Native ResNet-18 ImageNet weights instead of Places2.")
    return True

class BigLaMa(nn.Module):
    """
    The PyTorch architecture for Large Mask Inpainting using a ResNet-18 backbone.
    Structured specifically to isolate layers for the layer-freezing ablation study.
    """
    def __init__(self):
        super().__init__()
        
        # 1. PRE-TRAINED BACKBONE (ImageNet Geometry)
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the first layer to accept 4 channels (RGB + Mask) instead of 3
        self.encoder_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Copy pre-trained RGB weights
            self.encoder_conv1.weight[:, :3, :, :] = resnet.conv1.weight
            # Initialize the new Mask channel weights to zero
            self.encoder_conv1.weight[:, 3, :, :] = 0
            
        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool
        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3 # Outputs a 256-channel, 16x16 bottleneck
        
        # 2. PLASTICITY BLOCKS (Custom FFC/ResNet blocks to learn Oxford Textures)
        self.ffc_blocks = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # 3. UPSAMPLING (Restoring from 16x16 back to 256x256 resolution)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Squish final pixel values between -1 and 1
        )

    def forward(self, image, mask):
        # Combine the RGB image and the Mask into a 4-channel input
        x = torch.cat([image, mask], dim=1)
        
        # Pass through frozen geometry layers
        x = self.encoder_conv1(x)
        x = self.encoder_bn1(x)
        x = self.encoder_relu(x)
        x = self.encoder_maxpool(x)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        
        # Pass through plastic texture layers
        x = self.ffc_blocks(x)
        x = self.upsample(x)
        
        return x
        
    def get_encoder_parameters(self):
        """Helper function to cleanly grab just the ResNet backbone for Stage 6.3 freezing."""
        return list(self.encoder_conv1.parameters()) + \
               list(self.encoder_bn1.parameters()) + \
               list(self.encoder_layer1.parameters()) + \
               list(self.encoder_layer2.parameters()) + \
               list(self.encoder_layer3.parameters())
