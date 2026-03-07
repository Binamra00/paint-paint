"""
Dynamic Geometric Mask Generator
Pipeline Stage: 4 (Core Experiments)
"""
import cv2
import numpy as np
import torch

class DynamicMaskGenerator:
    """
    Generates aggressive random geometric masks on the fly during training
    to serve as the self-supervised signal for the inpainting models.
    """
    def __init__(self, height=256, width=256):
        self.height = height
        self.width = width

    def _draw_random_lines(self, img):
        num_lines = np.random.randint(1, 6)
        for _ in range(num_lines):
            x1, y1 = np.random.randint(0, self.width), np.random.randint(0, self.height)
            x2, y2 = np.random.randint(0, self.width), np.random.randint(0, self.height)
            thickness = np.random.randint(10, 30) # Aggressive thickness
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness)
        return img

    def _draw_random_circles(self, img):
        num_circles = np.random.randint(1, 5)
        for _ in range(num_circles):
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            radius = np.random.randint(15, 50) # Aggressive radius
            cv2.circle(img, (x, y), radius, (255, 255, 255), -1)
        return img

    def generate_mask(self):
        """
        Generates the mask and returns a single-channel PyTorch tensor (1, H, W).
        1.0 represents the missing/masked pixels, 0.0 represents valid pixels.
        """
        # Start with a black image canvas
        img = np.zeros((self.height, self.width, 1), np.uint8)
        
        # Apply the randomized geometric shapes
        img = self._draw_random_lines(img)
        img = self._draw_random_circles(img)
        
        # Normalize to [0, 1] and convert to PyTorch Tensor
        mask_tensor = torch.from_numpy(img).float() / 255.0
        
        # Rearrange dimensions from (H, W, C) to (C, H, W) for PyTorch
        return mask_tensor.permute(2, 0, 1)

# Local testing block
if __name__ == "__main__":
    generator = DynamicMaskGenerator()
    sample_mask = generator.generate_mask()
    print(f"Successfully generated mask tensor of shape: {sample_mask.shape}")
