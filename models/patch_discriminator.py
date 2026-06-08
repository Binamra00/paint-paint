"""
Phase 6: Adversarial Upgrade (LaMa Architecture)
Architecture: Spectral Dilated PatchGAN Discriminator
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class PatchGANDiscriminator(nn.Module):
    """
    An advanced PatchGAN equipped with Spectral Normalization and Dilated Convolutions.
    
    Dilation artificially expands the receptive field, allowing the discriminator
    to 'see' massive masks (e.g., 50% of the image) without losing local texture analysis.
    Spectral Norm stabilizes the GAN training, preventing mode collapse and checkerboard artifacts.
    """
    def __init__(self, in_channels=3, base_filters=64):
        super().__init__()
        
        # Layer 1: Standard downsampling
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, base_filters, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 2: Standard downsampling (BatchNorm removed for Spectral Norm stability)
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 3: DILATION INTRODUCED
        # Stride 2 + Dilation 2 drastically expands the network's field of view
        # Padding is increased to 2 to safely handle the dilated spatial math
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=2, dilation=2)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 4: Dilation continues. Stride 1 to preserve spatial resolution.
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=4, stride=1, padding=2, dilation=2)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 5: Final output prediction map
        # We do NOT use a Sigmoid here because we use BCEWithLogitsLoss for numerical stability
        self.final = nn.Conv2d(base_filters * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.final(x)

# Quick validation test
if __name__ == "__main__":
    dummy_image = torch.randn(1, 3, 256, 256)
    netD = PatchGANDiscriminator()
    out = netD(dummy_image)
    
    print("=====================================================")
    print("✅ Spectral Dilated PatchGAN Successfully Initialized!")
    print(f"-> Output tensor shape: {out.shape}")
    print("=====================================================")
    # Note: Due to dilation, the output grid dimensions will change slightly from the old 30x30,
    # but BCEWithLogitsLoss will automatically broadcast and handle the new grid size perfectly.
