<div align="center">

# mit-bih-ecg-arrhythmia-ml-dl-system

## Arrhythmia Analysis System Based on Machine Learning & Deep Learning

### ECG intelligent diagnosis platform validated on MIT-BIH Arrhythmia Database

[🇨🇳 Chinese Version](./README_CN.md) | [🇺🇸 English Version](./README.md)

![PyQt6](https://img.shields.io/badge/PyQt6-GUI-green)
![MIT](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---

# 📋 Table of Contents

1. [Overview](#-overview)
2. [MIT-BIH Database Introduction](#mit-bih-database-introduction)
3. [Core Features](#-core-features)
4. [System Workflow](#-system-workflow)
5. [Tech Stack & Dependencies](#️-tech-stack--dependencies)
6. [Project Structure](#-project-structure)
7. [Quick Start](#-quick-start)
8. [Experimental Results](#-experimental-results)
9. [License](#-license)

---

# 📖 Overview

This system is a complete ECG arrhythmia intelligent analysis platform developed for biomedical signal processing course project. All training, verification and testing are based on the **MIT-BIH Arrhythmia Database**, the gold standard dataset for ECG research.

The platform integrates full pipeline: raw ECG reading, wavelet denoising, heartbeat segmentation, handcrafted feature extraction (TSFEL), traditional machine learning (Random Forest, SVM), 1D-CNN deep learning classification, and PyQt6 visual interactive GUI. It supports offline model training, performance evaluation, and real-time ECG reasoning simulation.

---

# MIT-BIH Database Introduction

The MIT-BIH Arrhythmia Database is jointly created by MIT and Beth Israel Hospital, containing 48 long-term Holter ECG records (total 110 hours, sampling rate 360Hz) with expert beat-by-beat annotations. It provides raw ECG waveforms and corresponding annotation files, widely used to verify arrhythmia classification algorithms.

The system divides ECG into 5 categories:

- Normal (N)
- Atrial Premature (A)
- Left Bundle Branch (L)
- Right Bundle Branch (R)
- Ventricular Premature (V)

---

# ✨ Core Features

### 📂 MIT-BIH Data Parsing

- Read `.hea/.dat/.atr` files via WFDB toolkit
- Auto R-peak detection & heartbeat segmentation
- 300-sample heartbeat extraction centered on R peak

### 🧹 ECG Preprocessing

- db4 wavelet threshold denoising
- Two processing modes:
  - Full sequence
  - Segmented heartbeat

### 📊 Time Series Feature Engineering

- TSFEL automatic feature extraction
- Time-domain, frequency-domain and statistical features
- Feature filtering
- PCA dimensionality reduction

### 🤖 Machine Learning Module

#### Random Forest

- Adjustable tree number
- Adjustable max depth
- Adjustable leaf samples
- Full metrics evaluation

#### SVM (RBF Kernel)

- Hyperparameter tuning
- Multi-combination comparison
- C & gamma optimization

### 🧠 Deep Learning Module

- 1D-CNN based on PyTorch
- End-to-end ECG classification
- No manual feature extraction required
- Support expandable architectures:
  - ResNet
  - LSTM
  - Transformer
- Adam optimizer
- CrossEntropy Loss
- Batch training
- Convergence curve visualization

### 📈 Comprehensive Model Evaluation

- Accuracy
- Precision
- Recall
- F1-Score
- Specificity
- ROC-AUC
- Confusion Matrix
- Per-class performance charts

### 🖥️ PyQt6 Visual GUI

- One-click full workflow
- Waveform comparison
- Training progress monitoring
- Real-time ECG inference
- Probability distribution visualization

---

# 🏗️ System Workflow

```text
MIT-BIH Raw ECG Data (.dat/.hea/.atr)
        ↓
WFDB Read & R-Peak Location
        ↓
Single Heartbeat Segmentation (300 points)
        ↓
db4 Wavelet Denoising

├─ Traditional ML Pipeline ──┐
│
│ TSFEL Feature Extraction
│        ↓
│ Feature Filter + PCA
│        ↓
│ Random Forest / SVM Training & Evaluation
│
└─ Deep Learning Pipeline ──┘

      Direct Input to 1D-CNN
                ↓
       Train & Validate
                ↓
      Model Weights Save

                ↓

GUI Real-Time Inference Module
Load Model + ECG Signal
Predict Rhythm Class
Confidence Visualization
```

---

# 🛠️ Tech Stack & Dependencies

## Required Python Libraries

```txt
python>=3.10

wfdb
pywavelets
tsfel
numpy
pandas
scikit-learn

torch
matplotlib
seaborn

pyqt6
pyqtgraph
```

---

# 📂 Project Structure

```plaintext
mit-bih-ecg-arrhythmia-ml-dl-system
│
├── data/                 # MIT-BIH ECG dataset storage
├── preprocess/           # ECG read, segmentation, wavelet denoise
├── feature/              # TSFEL feature extraction & PCA
├── ml_models/            # Random Forest & SVM code
├── dl_models/            # 1D-CNN network & train scripts
├── gui/                  # PyQt6 GUI logic & drawing
├── utils/                # Metrics calculation & plot tools
├── weights/              # Saved .pth / .pkl model files
├── main.py               # Program entry
├── requirements.txt
├── README.md
└── README_CN.md
```

---

# 🚀 Quick Start

## Clone Repository

```bash
git clone https://github.com/YourName/mit-bih-ecg-arrhythmia-ml-dl-system.git
cd mit-bih-ecg-arrhythmia-ml-dl-system
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Prepare Dataset

Put MIT-BIH `.hea/.dat/.atr` files into:

```text
./data/
```

## Launch GUI System

```bash
python main.py
```

---

# 📊 Experimental Results

## Traditional Machine Learning

### Random Forest

```text
Accuracy = 0.8794
```

Poor performance on Atrial Premature (A).

### Optimized SVM

```text
C = 1
gamma = 0.1
Accuracy = 0.8689
```

Balanced precision and recall.

---

## 1D-CNN Deep Learning

- Overall accuracy nearly 0.99 after convergence
- N/L/R/V beat recall > 0.96
- Only minor missing detection on class A
- Stable loss and accuracy curves
- No obvious overfitting

---

## Conclusion

1D-CNN automatically mines deep temporal features of ECG and significantly outperforms traditional machine learning methods in classification accuracy and generalization capability.

---

# 📄 License

This project is only for academic research and course design, distributed under the MIT License.
