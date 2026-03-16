"""
Unified Inpainting Loss Module
Supports Early Stages (HRFPL) and Phase 10 (Adversarial GAN)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG19PerceptualLoss(nn.Module):
    """
    Calculates the Perceptual Loss using a pre-trained VGG19 feature extractor.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        self.blocks = nn.ModuleList([
            vgg[:4],    # relu1_2
            vgg[4:9],   # relu2_2
            vgg[9:18],  # relu3_4
            vgg[18:27], # relu4_4
            vgg[27:36]  # relu5_4
        ])
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0
        
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        loss = 0.0
        x, y = pred, target
        
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
            
        return loss

# --- ORIGINAL LOSS (For Stages 4 through 9) ---
class InpaintingLoss(nn.Module):
    """
    The standard master loss function combining pixel-perfect L1 loss with HRFPL.
    Used for the Control Variable and Ablation Study runs.
    """
    def __init__(self, perceptual_weight=0.1):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGG19PerceptualLoss()
        self.perceptual_weight = perceptual_weight

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        return l1 + (self.perceptual_weight * perceptual)

# --- ADVERSARIAL LOSSES (For Stage 10 GAN) ---
class GeneratorLoss(nn.Module):
    """
    The master loss function for the GAN Generator. 
    Combines L1, VGG Perceptual, and Adversarial BCE loss.
    """
    def __init__(self, perceptual_weight=0.1, adv_weight=0.1):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGG19PerceptualLoss()
        self.adv_loss = nn.BCEWithLogitsLoss()
        
        self.perceptual_weight = perceptual_weight
        self.adv_weight = adv_weight

    def forward(self, pred_fake, target_real, disc_pred_fake):
        l1 = self.l1_loss(pred_fake, target_real)
        perceptual = self.perceptual_loss(pred_fake, target_real)
        
        target_tensor = torch.ones_like(disc_pred_fake)
        adv = self.adv_loss(disc_pred_fake, target_tensor)
        
        return l1 + (self.perceptual_weight * perceptual) + (self.adv_weight * adv)

class DiscriminatorLoss(nn.Module):
    """
    The loss function for the PatchGAN Discriminator.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, disc_pred_real, disc_pred_fake):
        real_loss = self.loss_fn(disc_pred_real, torch.ones_like(disc_pred_real))
        fake_loss = self.loss_fn(disc_pred_fake, torch.zeros_like(disc_pred_fake))
        return (real_loss + fake_loss) / 2.0
