<div align="center">

# mit-bih-ecg-arrhythmia-ml-dl-system

## Arrhythmia Analysis System Based on Machine Learning & Deep Learning

### ECG Intelligent Diagnosis Platform Validated on MIT-BIH Arrhythmia Database

[Chinese 简体中文](./README_CN.md) | [English](./README.md)

![PyQt6](https://img.shields.io/badge/PyQt6-GUI-green)
![MIT](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---

# 📖 Overview

This system is a complete ECG arrhythmia intelligent analysis platform developed for biomedical signal processing course project. All training, validation and testing are based on the **MIT-BIH Arrhythmia Database**, one of the most widely used benchmark datasets in ECG research.

The platform integrates the complete workflow of ECG signal processing, including signal acquisition, wavelet denoising, heartbeat segmentation, handcrafted feature extraction, machine learning classification, deep learning classification and PyQt6 graphical user interface.

---

# 🖥️ System Demonstration

<img width="1918" height="1017" alt="c6b020519191bff01190f423bdef128a" src="https://github.com/user-attachments/assets/3ec70a9d-1889-41f4-9d07-2bdb53c1dd4b" />


---

# 📋 Table of Contents

1. Overview
2. MIT-BIH Database Introduction
3. Core Features
4. System Workflow
5. Tech Stack & Dependencies
6. Project Structure
7. Quick Start
8. Experimental Results
9. License

---

# MIT-BIH Database Introduction

The MIT-BIH Arrhythmia Database is jointly developed by MIT and Beth Israel Hospital. It contains 48 long-term ECG recordings with expert beat-by-beat annotations, totaling approximately 110 hours of ECG data sampled at 360 Hz.

This project classifies ECG beats into five categories:

- Normal (N)
- Atrial Premature Beat (A)
- Left Bundle Branch Block Beat (L)
- Right Bundle Branch Block Beat (R)
- Ventricular Premature Beat (V)

---

# ✨ Core Features

## 📂 MIT-BIH Data Processing

- WFDB-based ECG data reading
- Automatic R-peak detection
- Heartbeat segmentation
- Single heartbeat extraction centered on R peaks

## 🧹 ECG Preprocessing

- db4 wavelet denoising
- Noise suppression
- Signal enhancement

## 📊 Feature Engineering

- TSFEL feature extraction
- Time-domain features
- Frequency-domain features
- Statistical features
- PCA dimensionality reduction

## 🤖 Machine Learning

### Random Forest

- Adjustable hyperparameters
- Model evaluation
- Classification visualization

### SVM

- RBF kernel
- C parameter optimization
- Gamma parameter optimization

## 🧠 Deep Learning

- 1D-CNN architecture
- PyTorch implementation
- End-to-end ECG classification
- Adam optimizer
- CrossEntropy loss

Expandable architectures:

- ResNet
- LSTM
- Transformer

## 📈 Performance Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- Specificity
- ROC-AUC
- Confusion Matrix

## 🖥️ GUI System

- PyQt6 interface
- ECG waveform visualization
- Model management
- Real-time prediction
- Probability distribution display

---

# 🏗️ System Workflow

```text
MIT-BIH Raw ECG Data
        ↓
WFDB Reading
        ↓
R-Peak Detection
        ↓
Heartbeat Segmentation
        ↓
Wavelet Denoising

Traditional ML:
TSFEL → PCA → RF/SVM

Deep Learning:
Heartbeat → 1D-CNN

        ↓

GUI Inference System
        ↓
Classification Result
````

---

# 🛠️ Tech Stack & Dependencies

```txt
Python >= 3.10

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
├── data/
├── preprocess/
├── feature/
├── ml_models/
├── dl_models/
├── gui/
├── utils/
├── weights/
├── main.py
├── requirements.txt
├── README.md
└── README_CN.md
```

---

# 🚀 Quick Start

## Clone Repository

```bash
git clone  https://github.com/ChenPinan001/mit-bih-ecg-arrhythmia-ml-analysis.git
cd mit-bih-ecg-arrhythmia-ml-dl-system
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Prepare Dataset

Place all MIT-BIH files into:

```text
./data/
```

## Launch Application

```bash
python main.py
```

---

## 📊 Experimental Results

### Random Forest

Model Parameters:

```text
n_estimators = 50
max_depth = 8
min_samples_split = 3
min_samples_leaf = 2
max_features = sqrt
```

Performance:

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.8794 |
| Precision | 0.8634 |
| Recall    | 0.6153 |
| F1-Score  | 0.6967 |
| ROC-AUC   | 0.4260 |

The model achieved the best performance on Normal beats (N), while the recall of Atrial Premature beats (A) was relatively low.

### SVM (RBF Kernel)

Best Hyperparameter Combination:

```text
C = 1.0
gamma = 0.1
```

Performance:

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.8689 |
| Precision | 0.8256 |
| Recall    | 0.5634 |
| F1-Score  | 0.6134 |
| ROC-AUC   | 0.4258 |

Among the four parameter combinations tested, the configuration with C=1 and gamma=0.1 achieved the best overall classification performance.

### 1D-CNN

Training Configuration:

```text
Batch Size = 256
Epoch = 20
Learning Rate = 0.001
Optimizer = Adam
Loss = CrossEntropyLoss
```

Experimental observations:

* Training and validation losses converged steadily.
* Validation accuracy stabilized around 0.99 after training.
* Recall rates of N, L, R and V classes were higher than 0.96.
* Recall rate of class A reached 0.879.
* No obvious overfitting was observed.

### Conclusion

Experimental results demonstrate that the 1D-CNN model outperforms traditional machine learning methods in ECG arrhythmia classification and exhibits stronger feature learning and generalization capabilities.



# 📄 License

This project is developed for academic research and course design purposes and is distributed under the MIT License.

```
```
