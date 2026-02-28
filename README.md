# Paint-Paint: Domain Adaptive Image Inpainting via Self-Supervised Fine-Tuning

## 📌 Project Overview
General-purpose image inpainting models (trained on massive datasets like Places2 or ImageNet) often fail to generalize to specialized domains with unique, high-frequency textures, such as medical imagery or specific biological datasets. 

This project investigates **domain adaptation in generative AI**. We aim to answer the core research question: 
> *"How can parameter-efficient fine-tuning and selective layer freezing optimize the trade-off between structural stability and textural plasticity when adapting large inpainting models to specialized, data-scarce domains?"*

To test this, we are adapting the **BigLaMa (Large Mask Inpainting)** architecture to the **Oxford 102 Flowers** dataset using self-supervised fine-tuning, benchmarking its performance against a baseline **U-Net with Gated Convolutions**.

## 🚀 Key Architectural Features
* **Fast Fourier Convolutions (FFCs):** Utilizing LaMa's global receptive field to handle large missing areas by predicting repetitive periodic structures.
* **Dynamic Geometric Masking:** Generating aggressive, randomized masks on-the-fly during training as a self-supervised signal.
* **Layer-Freezing Ablation:** Experimenting with freezing initial downsampling layers versus unfreezing FFC ResNet blocks to balance pre-trained structural knowledge with new textural learning.
* **HRFPL Loss:** Penalizing the model using High Receptive Field Perceptual Loss to ensure photorealistic generation.

## 📂 Repository Structure
To ensure global reproducibility and adherence to ML pipeline best practices, this repository is structured as follows:

```text
paint-paint/
├── data/               # Scripts to fetch and preprocess the Oxford 102 Flowers dataset
├── models/             # PyTorch definitions for BigLaMa and Baseline U-Net
├── training/           # Training loops, dynamic masking generators, and loss functions
├── evaluation/         # Scripts to calculate LPIPS, PSNR, and SSIM
├── notebooks/          # Primary orchestrator notebook (domain_inpaint.ipynb)
├── requirements.txt    # Exact environment dependencies
└── README.md
```
