"""
Oxford 102 Flowers PyTorch Dataset
Pipeline Stage: 4 (Core Experiments)
"""
import os
import scipy.io
import urllib.request
import logging
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OxfordFlowersDataset(Dataset):
    """
    Custom PyTorch Dataset for Oxford 102 Flowers.
    Automatically handles official dataset splits and dynamic geometric masking.
    """
    def __init__(self, root_dir, split="train", mask_generator=None):
        self.root_dir = root_dir
        self.jpg_dir = os.path.join(root_dir, "jpg")
        self.split = split
        self.mask_generator = mask_generator
        
        # 1. Standardize and Normalize: Resize to 256x256, scale to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 2. Fetch and parse the official splits
        self.image_ids = self._get_split_ids()

    def _get_split_ids(self):
        """Downloads setid.mat if missing and extracts the correct image IDs."""
        mat_path = os.path.join(self.root_dir, "setid.mat")
        mat_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"
        
        if not os.path.exists(mat_path):
            logging.info(f"Downloading official splits from {mat_url}...")
            urllib.request.urlretrieve(mat_url, mat_path)
            
        # Load the MATLAB file
        mat_data = scipy.io.loadmat(mat_path)
        
        # Oxford 102 splits: 'trnid' (train), 'valid' (validation), 'tstid' (test)
        split_map = {
            "train": "trnid",
            "val": "valid",
            "test": "tstid"
        }
        
        if self.split not in split_map:
            raise ValueError(f"Invalid split '{self.split}'. Use 'train', 'val', or 'test'.")
            
        # Extract the IDs (flattened to a simple 1D list)
        split_key = split_map[self.split]
        ids = mat_data[split_key][0].tolist()
        logging.info(f"Loaded {len(ids)} images for the '{self.split}' split.")
        return ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 3. Load the image based on the ID (Oxford uses 1-based indexing: image_00001.jpg)
        img_id = self.image_ids[idx]
        img_name = f"image_{img_id:05d}.jpg"
        img_path = os.path.join(self.jpg_dir, img_name)
        
        # Open image and convert to RGB (to ensure consistency)
        original_image = Image.open(img_path).convert("RGB")
        
        # 4. Apply geometric standardization and normalization
        ground_truth = self.transform(original_image)
        
        # 5. Apply dynamic masking on the fly
        if self.mask_generator:
            mask = self.mask_generator.generate_mask()
            # Mask out the image: 1.0 means missing pixel, 0.0 means valid pixel
            # Multiply by (1 - mask) to turn the masked regions black (0)
            masked_image = ground_truth * (1.0 - mask)
        else:
            # Fallback if no mask generator is provided
            mask = torch.zeros((1, 256, 256))
            masked_image = ground_truth

        # Return a dictionary containing everything the training loop needs
        return {
            "ground_truth": ground_truth,
            "masked_image": masked_image,
            "mask": mask
        }
