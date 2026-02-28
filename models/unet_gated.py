"""
Baseline Model: U-Net with Gated Convolutions
Pipeline Stage: Architecture Setup
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedConv2d(nn.Module):
    """
    The mathematical implementation of Gated Convolution.
    Runs two parallel convolutions: one for features, one for the gate.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Parallel convolutions for feature and gate
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        feature = self.feature_conv(x)
        gate = torch.sigmoid(self.gate_conv(x)) # Squish gate between 0 and 1
        return feature * gate # Multiply to mask out garbage pixels

class BaselineUNet(nn.Module):
    """
    A simplified U-Net utilizing Gated Convolutions for Image Inpainting.
    Expects a 4-channel input (RGB image + 1-channel Mask).
    """
    def __init__(self):
        super().__init__()
        # Downsampling Path (Encoder)
        self.enc1 = GatedConv2d(4, 64)
        self.enc2 = GatedConv2d(64, 128, stride=2)
        self.enc3 = GatedConv2d(128, 256, stride=2)
        
        # Bottleneck
        self.bottleneck = GatedConv2d(256, 256)
        
        # Upsampling Path (Decoder)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec1 = GatedConv2d(256, 128) # 256 because of skip connection
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec2 = GatedConv2d(128, 64)
        
        # Final Output (3-channel RGB image)
        self.final = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, image, mask):
        # Concatenate image and mask to feed into the network
        x = torch.cat([image, mask], dim=1)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        b = self.bottleneck(e3)
        
        # Decoder with Skip Connections
        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        
        out = self.final(d2)
        return torch.sigmoid(out) # Return final image
