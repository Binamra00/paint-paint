"""
Dynamic Geometric Mask Generator (Large Masks)
Pipeline Stage: 4 (Core Experiments)
"""
import cv2
import numpy as np
import torch

class DynamicMaskGenerator:
    """
    Generates 1 to 2 massive geometric masks on the fly during training.
    Designed specifically to test the global receptive field of FFCs.
    """
    def __init__(self, height=256, width=256):
        self.height = height
        self.width = width

    def _draw_large_rectangle(self, img):
        # Generate a rectangle taking up at least a third of the image dimension
        x1 = np.random.randint(0, self.width // 2)
        y1 = np.random.randint(0, self.height // 2)
        x2 = np.clip(x1 + np.random.randint(self.width // 3, self.width), 0, self.width)
        y2 = np.clip(y1 + np.random.randint(self.height // 3, self.height), 0, self.height)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
        return img

    def _draw_large_ellipse(self, img):
        # Generate a massive ellipse near the center
        center = (np.random.randint(self.width // 4, self.width * 3 // 4), 
                  np.random.randint(self.height // 4, self.height * 3 // 4))
        axes = (np.random.randint(self.width // 4, self.width // 2), 
                np.random.randint(self.height // 4, self.height // 2))
        angle = np.random.randint(0, 360)
        cv2.ellipse(img, center, axes, angle, 0, 360, (255, 255, 255), -1)
        return img

    def _draw_large_polygon(self, img):
        # Generate a large random polygon (triangle to hexagon)
        num_points = np.random.randint(3, 7)
        points = []
        for _ in range(num_points):
            points.append([np.random.randint(0, self.width), np.random.randint(0, self.height)])
        
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], (255, 255, 255))
        return img

    def _draw_massive_line(self, img):
        # Generate a brush stroke that is extremely thick
        x1, y1 = np.random.randint(0, self.width), np.random.randint(0, self.height)
        x2, y2 = np.random.randint(0, self.width), np.random.randint(0, self.height)
        thickness = np.random.randint(40, 100) # Extremely thick
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness)
        return img

    def generate_mask(self):
        """
        Generates the mask and returns a single-channel PyTorch tensor (1, H, W).
        1.0 represents the missing/masked pixels, 0.0 represents valid pixels.
        """
        img = np.zeros((self.height, self.width, 1), np.uint8)
        
        # Pick 1 or 2 massive shapes to draw
        num_shapes = np.random.randint(1, 3)
        shape_funcs = [
            self._draw_large_rectangle, 
            self._draw_large_ellipse, 
            self._draw_large_polygon, 
            self._draw_massive_line
        ]
        
        for _ in range(num_shapes):
            # Randomly select one of the shape functions and apply it
            func = np.random.choice(shape_funcs)
            img = func(img)
        
        # Normalize to [0, 1] and convert to PyTorch Tensor
        mask_tensor = torch.from_numpy(img).float() / 255.0
        
        # Rearrange dimensions from (H, W, C) to (C, H, W) for PyTorch
        return mask_tensor.permute(2, 0, 1)

# Local testing block
if __name__ == "__main__":
    generator = DynamicMaskGenerator()
    sample_mask = generator.generate_mask()
    print(f"Successfully generated mask tensor of shape: {sample_mask.shape}")
