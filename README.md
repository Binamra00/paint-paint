[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Binamra00/paint-paint/blob/main/notebooks/domain_inpaint.ipynb)

# Paint-Paint: Domain Adaptive Image Inpainting via Self-Supervised & Adversarial Fine-Tuning

## 📌 Project Overview
General-purpose image inpainting models (trained on massive datasets like ImageNet) often fail to generalize to specialized domains with unique, high-frequency textures, such as medical imagery or specific biological datasets. 

This project investigates **domain adaptation in generative AI**. We initially aimed to answer a core research question regarding structural stability vs. textural plasticity: 
> *"How can parameter-efficient fine-tuning and selective layer freezing optimize the adaptation of large inpainting models to specialized, data-scarce domains?"*

However, during our ablation study on the **Oxford 102 Flowers** dataset, visual proof revealed a critical limitation in standard mathematical objective functions (L1/MSE): **oversmoothing**. To minimize pixel-level mathematical penalties in heavily masked regions, the models safely guessed "muddy averages" rather than risking the generation of sharp, high-frequency petal textures.

To cure this, we evolved our pipeline into a **Generative Adversarial Network (GAN)**, pitting our hybrid ResNet-LaMa architecture against a localized **PatchGAN Discriminator** to force the hallucination of photorealistic, high-frequency domain textures.

## 🚀 Key Architectural Features
* **ResNet-18 Backbone:** Leveraging robust, pre-trained ImageNet weights to extract high-level structural geometry and boundaries. 
* **Fast Fourier Convolutions (FFCs):** Utilizing a global receptive field to handle large missing areas by predicting repetitive periodic structures. 
* **Dynamic Geometric Masking:** Generating aggressive, randomized masks on-the-fly during training as a self-supervised signal.
* **The Ablation Study (Layer-Freezing):** Proved that freezing early structural layers degrades performance in high-frequency domains; the network requires full textural plasticity to succeed.
* **PatchGAN Discriminator:** An algorithmic "art critic" that analyzes the image in 70x70 patches, mathematically penalizing the Generator for producing blurry, safe averages.
* **Dual-Objective Loss:** A unified loss module combining High Receptive Field Perceptual Loss (HRFPL) via a VGG19 feature extractor, strict L1 pixel loss, and Binary Cross-Entropy (BCE) adversarial loss.

## 📊 Key Findings & Metrics
The final evaluation mathematically validated the adversarial upgrade across the unseen test split:
1. **Mathematical Safety vs. Perceptual Realism:** The purely mathematical ResNet (trained only on L1/HRFPL) secured higher structural scores (**PSNR: 18.03**), but produced visually blurry outputs.
2. **The GAN Victory:** The GAN-powered ResNet sacrificed strict pixel-reconstruction scores to successfully trick the PatchGAN critic, resulting in a superior perceptual realism score (**LPIPS: 0.4518**) and visually sharp, hallucinated petal textures.

## 📂 Repository Structure
To ensure global reproducibility and adherence to ML pipeline best practices, this repository is structured as follows:

```text
paint-paint/
├── data/               # Scripts to fetch and preprocess the Oxford 102 Flowers dataset
├── models/             # PyTorch definitions (ResNet-LaMa, U-Net Gated, PatchGAN Discriminator)
├── training/           # Universal training loops, dynamic maskers, and unified Loss modules
├── evaluation/         # Scripts to calculate LPIPS, PSNR, and SSIM metrics
├── notebooks/          # Primary orchestrator notebook (domain_inpaint.ipynb)
├── requirements.txt    # Exact environment dependencies
└── README.md
