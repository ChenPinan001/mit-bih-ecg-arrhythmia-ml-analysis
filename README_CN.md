<div align="center">

# mit-bih-ecg-arrhythmia-ml-dl-system

## 基于机器学习与深度学习的心律失常分析系统

### 基于 MIT-BIH 心律失常数据库验证的 ECG 智能诊断平台

[简体中文](./README_CN.md) | [English](./README.md)

![PyQt6](https://img.shields.io/badge/PyQt6-GUI-green)
![MIT](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---

# 📖 项目简介

本系统是面向《生物医学信号处理》课程设计开发的 ECG 心律失常智能分析平台。训练、验证与测试均基于国际经典 ECG 数据集 MIT-BIH Arrhythmia Database。

系统集成了 ECG 信号读取、小波降噪、心拍分割、TSFEL 特征提取、传统机器学习分类、深度学习分类以及 PyQt6 可视化界面，实现完整的 ECG 智能诊断流程。

---

# 🖥️ 系统演示

<img width="1918" height="1017" alt="c6b020519191bff01190f423bdef128a" src="https://github.com/user-attachments/assets/38d0a2c0-4e32-4c78-a86f-82dd592535db" />


---

# 📋 目录

1. 项目简介
2. MIT-BIH 数据库介绍
3. 核心功能
4. 系统流程
5. 技术栈与依赖
6. 项目结构
7. 快速开始
8. 实验结果
9. 许可证

---

# MIT-BIH 数据库介绍

MIT-BIH Arrhythmia Database 由 MIT 与 Beth Israel Hospital 联合建立，是目前心律失常研究中最经典的数据集之一。

本项目将 ECG 心拍分为五类：

- 正常心拍（N）
- 房性早搏（A）
- 左束支阻滞（L）
- 右束支阻滞（R）
- 室性早搏（V）

---

# ✨ 核心功能

## 📂 数据处理

- WFDB 数据读取
- R峰自动定位
- 单心拍分割

## 🧹 ECG 预处理

- db4 小波降噪
- 信号增强
- 噪声抑制

## 📊 特征工程

- TSFEL 自动特征提取
- 时域特征
- 频域特征
- 统计特征
- PCA降维

## 🤖 机器学习模块

### Random Forest

- 参数可调
- 模型评估
- 分类结果展示

### SVM

- RBF核
- 参数优化

## 🧠 深度学习模块

- PyTorch实现
- 1D-CNN分类模型
- Adam优化器
- CrossEntropy损失函数

支持扩展：

- ResNet
- LSTM
- Transformer

## 📈 性能评估

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- 混淆矩阵

## 🖥️ 图形界面

- PyQt6界面
- ECG波形显示
- 模型管理
- 实时预测
- 概率可视化

---

# 🏗️ 系统流程

```text
MIT-BIH ECG数据
        ↓
WFDB读取
        ↓
R峰检测
        ↓
心拍分割
        ↓
小波降噪

传统机器学习：
TSFEL → PCA → RF/SVM

深度学习：
Heartbeat → 1D-CNN

        ↓

GUI诊断系统
        ↓
分类结果输出
````

---

# 🛠️ 技术栈与依赖

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

# 📂 项目结构

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

# 🚀 快速开始

## 克隆项目

```bash
git clone https://github.com/YourName/mit-bih-ecg-arrhythmia-ml-dl-system.git
cd mit-bih-ecg-arrhythmia-ml-dl-system
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 准备数据集

将 MIT-BIH 数据放入：

```text
./data/
```

## 启动系统

```bash
python main.py
```

---

## 📊 实验结果

### Random Forest

模型参数：

```text
n_estimators = 50
max_depth = 8
min_samples_split = 3
min_samples_leaf = 2
max_features = sqrt
```

性能指标：

| 指标        | 数值     |
| --------- | ------ |
| Accuracy  | 0.8794 |
| Precision | 0.8634 |
| Recall    | 0.6153 |
| F1-Score  | 0.6967 |
| ROC-AUC   | 0.4260 |

随机森林对正常心拍（N）的识别效果较好，但对房性早搏（A）的召回率相对较低。

### SVM（RBF核）

最优参数组合：

```text
C = 1.0
gamma = 0.1
```

性能指标：

| 指标        | 数值     |
| --------- | ------ |
| Accuracy  | 0.8689 |
| Precision | 0.8256 |
| Recall    | 0.5634 |
| F1-Score  | 0.6134 |
| ROC-AUC   | 0.4258 |

在四组参数组合中，C=1、gamma=0.1 的模型取得了最佳综合性能。

### 1D-CNN

训练参数：

```text
Batch Size = 256
Epoch = 20
Learning Rate = 0.001
Optimizer = Adam
Loss = CrossEntropyLoss
```

实验现象：

* 训练损失与验证损失均稳定收敛；
* 验证集准确率最终稳定在 0.99 附近；
* N、L、R、V 四类召回率均高于 0.96；
* A 类召回率达到 0.879；
* 未出现明显过拟合现象。

### 结论

实验结果表明，1D-CNN 能够自动学习 ECG 信号中的深层时序特征，在分类性能和泛化能力方面均优于传统机器学习模型。

---

# 📄 许可证

本项目仅用于学术研究与课程设计，遵循 MIT License 开源协议。

```
```
