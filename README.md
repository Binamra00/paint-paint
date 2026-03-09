[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Binamra00/paint-paint/blob/main/notebooks/domain_inpaint.ipynb)

# Paint-Paint: Domain Adaptive Image Inpainting via Self-Supervised Fine-Tuning

## 📌 Project Overview
General-purpose image inpainting models (trained on massive datasets like ImageNet) often fail to generalize to specialized domains with unique, high-frequency textures, such as medical imagery or specific biological datasets. 

This project investigates **domain adaptation in generative AI**. We aim to answer the core research question: 
> *"How can parameter-efficient fine-tuning and selective layer freezing optimize the trade-off between structural stability and textural plasticity when adapting large inpainting models to specialized, data-scarce domains?"*

To test this, we engineered a hybrid **ResNet-LaMa** architecture—merging a pre-trained ResNet-18 ImageNet backbone with Fast Fourier Convolution (FFC) blocks. We are adapting this model to the **Oxford 102 Flowers** dataset using self-supervised fine-tuning, benchmarking its performance against a baseline **U-Net with Gated Convolutions**.

## 🚀 Key Architectural Features
* **ResNet-18 Backbone:** Leveraging robust, pre-trained ImageNet weights to extract high-level structural geometry and boundaries. 
* **Fast Fourier Convolutions (FFCs):** Utilizing a global receptive field to handle large missing areas by predicting repetitive periodic structures and learning new high-frequency textures. 
* **Dynamic Geometric Masking:** Generating aggressive, randomized masks on-the-fly during training as a self-supervised signal.
* **Layer-Freezing Ablation:** Experimenting with freezing the initial ResNet-18 downsampling layers (preserving structural knowledge) versus unfreezing the entire network to balance pre-trained geometry with new textural plasticity.
* **HRFPL Loss:** Penalizing the model using High Receptive Field Perceptual Loss to ensure photorealistic generation.

## 📂 Repository Structure
To ensure global reproducibility and adherence to ML pipeline best practices, this repository is structured as follows:

```text
paint-paint/
├── data/               # Scripts to fetch and preprocess the Oxford 102 Flowers dataset
├── models/             # PyTorch definitions for ResNet-LaMa and Baseline U-Net
├── training/           # Training loops, dynamic masking generators, and loss functions
├── evaluation/         # Scripts to calculate LPIPS, PSNR, and SSIM
├── notebooks/          # Primary orchestrator notebook (domain_inpaint.ipynb)
├── requirements.txt    # Exact environment dependencies
└── README.md
