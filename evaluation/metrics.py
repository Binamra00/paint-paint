"""
Phase 5, Step 7: Final Metric Extraction (PSNR, SSIM, LPIPS)
Utility script to evaluate models on the test split.
"""
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# Import custom modules
from data.dataset import OxfordFlowersDataset
from training.mask_generator import DynamicMaskGenerator
from models.unet_gated import BaselineUNet
from models.bigLaMa import BigLaMa

def evaluate_model(model, model_name, dataloader, device, metrics):
    """Evaluates a single model and returns the averaged metrics."""
    print(f"\nEvaluating {model_name}...")
    model.eval()
    
    psnr_metric, ssim_metric, lpips_metric = metrics
    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Scoring {model_name}"):
            gt = batch["ground_truth"].to(device)
            masked = batch["masked_image"].to(device)
            masks = batch["mask"].to(device)
            
            # Generate the inpainted image
            out = model(masked, masks)
            
            # Calculate metrics
            total_psnr += psnr_metric(out, gt).item()
            total_ssim += ssim_metric(out, gt).item()
            total_lpips += lpips_metric(out, gt).mean().item()
            batches += 1
            
    return {
        "PSNR": total_psnr / batches,
        "SSIM": total_ssim / batches,
        "LPIPS": total_lpips / batches
    }

def run_full_evaluation():
    """Loads data, models, and runs evaluation on all configurations."""
    print("Initializing Phase 5: Quantitative Evaluation (Test Split)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    # 1. Initialize the Unseen Test Data
    mask_gen = DynamicMaskGenerator(height=256, width=256)
    test_dataset = OxfordFlowersDataset(root_dir="data/raw/oxford_102", split="test", mask_generator=mask_gen)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 2. Initialize the Metrics (Data range 2.0 for Tanh [-1, 1])
    psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    lpips_metric = lpips.LPIPS(net='vgg').to(device)
    metrics = (psnr_metric, ssim_metric, lpips_metric)

    # 3. Load the Saved Weights
    model_base = BaselineUNet().to(device)
    if os.path.exists("models/saved_weights/Baseline_UNet.pth"):
        model_base.load_state_dict(torch.load("models/saved_weights/Baseline_UNet.pth", map_location=device, weights_only=True))
    else:
        print("⚠️ Warning: Baseline_UNet.pth not found.")

    model_run_a = BigLaMa().to(device)
    if os.path.exists("models/saved_weights/ResNetLaMa_Unfrozen.pth"):
        model_run_a.load_state_dict(torch.load("models/saved_weights/ResNetLaMa_Unfrozen.pth", map_location=device, weights_only=True))
    else:
        print("⚠️ Warning: ResNetLaMa_Unfrozen.pth not found.")

    model_run_b = BigLaMa().to(device)
    if os.path.exists("models/saved_weights/ResNetLaMa_Frozen_Encoder.pth"):
        model_run_b.load_state_dict(torch.load("models/saved_weights/ResNetLaMa_Frozen_Encoder.pth", map_location=device, weights_only=True))
    else:
        print("⚠️ Warning: ResNetLaMa_Frozen_Encoder.pth not found.")

    # 4. Execute Evaluation
    results = {
        "Baseline U-Net": evaluate_model(model_base, "Baseline U-Net", test_loader, device, metrics),
        "Run A: ResNet-LaMa (Unfrozen)": evaluate_model(model_run_a, "Run A: ResNet-LaMa (Unfrozen)", test_loader, device, metrics),
        "Run B: ResNet-LaMa (Frozen)": evaluate_model(model_run_b, "Run B: ResNet-LaMa (Frozen)", test_loader, device, metrics)
    }
    
    return results

if __name__ == "__main__":
    # Fallback: if run directly from the terminal, just print the raw dictionary
    results = run_full_evaluation()
    print("\nRaw Evaluation Results:", results)
