# HiSAM-MambaNet

**HiSAM-MambaNet: A Dual-Path Hierarchical Hybrid Architecture for Motor Imagery EEG Classification**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)

This repository contains the official PyTorch implementation of the paper **"HiSAM-MambaNet"**.

## 📖 Introduction

Motor Imagery EEG decoding is challenging due to complex spatiotemporal dynamics and the efficiency-expressiveness trade-off in existing architectures. 

**HiSAM-MambaNet** proposes a hierarchical dual-path architecture:
- **Lower Layers:** Utilize Spatiotemporal Mamba for efficient local feature extraction.
- **Higher Layers:** Combine Temporal Mamba with Spatial Hybrid Attention (Local + Global) for long-range dependency modeling.
- **Dual-Path Strategy:** A main path and an auxiliary path process features in complementary spatiotemporal orders with adaptive fusion.

![Architecture](figures/architecture.png)
*(Note: Please upload the architecture image from your paper to a folder named 'figures' and uncomment the line above)*

## 🚀 Performance

Experiments on BCI Competition IV Datasets demonstrate state-of-the-art performance:

| Dataset | Accuracy |
| :--- | :--- |
| **BCI Competition IV-2a** | **82.27%** |
| **BCI Competition IV-2b** | **88.63%** |

## 🛠️ Requirements

The code requires a Python environment (suggested python 3.8+) with the following dependencies. Crucially, it requires `mamba-ssm` which needs a GPU environment.

```bash
pip install torch torchvision
pip install numpy scipy matplotlib scikit-learn pandas
pip install einops
pip install mamba-ssm  # Requires CUDA
