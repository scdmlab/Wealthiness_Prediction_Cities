# README

## Overview
This project implements a Multi-gate Mixture-of-Experts (MMOE) model with Swin Transformer for fine-grained image-based wealthiness classification. It includes two Python modules: `data.py` for data preprocessing and dataset creation, and `model.py` for the implementation of the MMOE model and its training pipeline.

---

## File Descriptions

### 1. `data.py`
- **Purpose:** Prepares the dataset and processes comparisons to compute TrueSkill scores.
- **Key Features:**
  - **`RegressionDataset`:** A custom PyTorch dataset to handle image preprocessing and loading.
  - **TrueSkill Algorithm:** Updates image scores based on temporal decay and spatial influence.
  - **`preprocess_csv`:** Processes input CSV data to compute wealthiness labels.
  - **Transforms:** Includes CLAHE preprocessing and feature extraction using a pre-trained model.

### 2. `model.py`
- **Purpose:** Defines the MMOE model and integrates it with the Swin Transformer for multi-task classification.
- **Key Features:**
  - **`MMOEModel`:** Implements the MMOE architecture with expert and gate networks.
  - **`build_classification_trainer`:** Configures and returns a Hugging Face `Trainer` for training the model.
  - **Base Model:** Utilizes a pre-trained Swin Transformer for feature extraction.

