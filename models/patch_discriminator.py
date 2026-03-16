"""
Phase 6: Adversarial Upgrade
Architecture: PatchGAN Discriminator (70x70 Receptive Field)
"""
import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    Evaluates image realism on a localized patch level rather than globally.
    Outputs a 2D feature map where each value judges a 70x70 pixel region.
    Expects a 3-channel RGB image (the completed inpainting).
    """
    def __init__(self, in_channels=3, base_filters=64):
        super().__init__()
        
        # Layer 1: No BatchNorm on the first layer (standard GAN practice)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 4: Stride 1 to keep the spatial resolution from dropping too low
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 5: Final output layer (maps to a 1-channel prediction map)
        # We do NOT use a Sigmoid here because we will use BCEWithLogitsLoss later for numerical stability
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
    print(f"✅ PatchGAN Initialized! Output shape: {out.shape}") 
    # Expected shape: [1, 1, 30, 30] - The 30x30 grid of "Real/Fake" judgments
