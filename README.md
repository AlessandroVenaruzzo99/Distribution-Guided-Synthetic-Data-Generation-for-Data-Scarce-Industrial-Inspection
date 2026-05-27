# Distribution-Guided Synthetic Data Generation for Data-Scarce Industrial Inspection

This repository contains the official implementation of the framework proposed in the paper: 
**"Distribution-Guided Synthetic Data Generation for Data-Scarce Industrial Inspection"** (Submitted to *The International Journal of Advanced Manufacturing Technology*).

## 📌 Overview
Our framework addresses the challenge of data scarcity in industrial quality inspection. Unlike standard augmentation techniques, this pipeline explicitly models:
1. **Instance-level distribution control**: Regulates per-class frequencies.
2. **Dataset-driven spatial compositing**: Uses empirical overlap and placement statistics from original training data to generate realistic synthetic scenes.
3. **Automated background reconstruction**: Ensures compatibility with structured industrial acquisition conditions.

## 🛠 Installation
To set up the environment, we recommend using Conda:

```bash
(https://github.com/AlessandroVenaruzzo99/Distribution-Guided-Synthetic-Data-Generation-for-Data-Scarce-Industrial-Inspection.git)

```
📊 Data Availability & Reproducibility

The industrial dataset used in the paper is proprietary and cannot be publicly
shared due to industrial collaboration constraints BUT the similar alternative Open is provided at the following link: https://universe.roboflow.com/alessandro-venaruzzo-4oaiw/manufacturing-backlight-dataset


⚖️ License

This project is licensed under the MIT License.
