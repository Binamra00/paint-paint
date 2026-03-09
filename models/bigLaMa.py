"""
BigLaMa Architecture & Loader
Pipeline Stage: Architecture Setup & Transfer Learning Initialization
"""
import os
import logging
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def fetch_pretrained_lama(target_dir="models/weights"):
    """Downloads the official BigLaMa pre-trained checkpoint."""
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "big-lama.zip")
    lama_url = "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
    
    if not os.path.exists(zip_path):
        logging.info("Downloading pre-trained BigLaMa (Places2) weights...")
        os.system(f"wget -qO {zip_path} {lama_url}")
    else:
        logging.info("Pre-trained BigLaMa weights found locally.")
    return True

class BigLaMa(nn.Module):
    """
    The PyTorch architecture for Large Mask Inpainting.
    Structured specifically to isolate layers for the layer-freezing ablation study.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Initial Downsampling Layers
        # (These are the layers you will FREEZE during Run B to retain global geometry)
        self.downsample = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # 2. Fast Fourier Convolution (FFC) ResNet Blocks
        # (These are the layers you keep UNFROZEN to learn the new flower textures)
        self.ffc_blocks = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # 3. Upsampling Layers (Restoring the image resolution)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid() # Squish final pixel values between 0 and 1
        )

    def forward(self, image, mask):
        # Combine the RGB image and the Mask into a 4-channel input
        x = torch.cat([image, mask], dim=1)
        
        # Pass through the isolated network blocks
        x = self.downsample(x)
        x = self.ffc_blocks(x)
        x = self.upsample(x)
        
        return x
