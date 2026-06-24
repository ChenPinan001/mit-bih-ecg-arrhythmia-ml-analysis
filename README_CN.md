<div align="center">

# mit-bih-ecg-arrhythmia-ml-dl-system

## 基于机器学习与深度学习的心律失常分析系统

### 基于 MIT-BIH 心律失常数据库验证的 ECG 智能诊断平台

[🇨🇳 中文版](./README_CN.md) | [🇺🇸 English Version](./README.md)

![PyQt6](https://img.shields.io/badge/PyQt6-GUI-green)
![MIT](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---

# 📋 目录

1. [项目简介](#-项目简介)
2. [MIT-BIH 数据库介绍](#mit-bih-数据库介绍)
3. [核心功能](#-核心功能)
4. [系统流程](#-系统流程)
5. [技术栈与依赖](#️-技术栈与依赖)
6. [项目结构](#-项目结构)
7. [快速开始](#-快速开始)
8. [实验结果](#-实验结果)
9. [许可证](#-许可证)

---

# 📖 项目简介

本系统是面向《生物医学信号处理》课程设计开发的完整 ECG（心电图）心律失常智能分析平台。系统的训练、验证与测试均基于心电研究领域的标准数据集 **MIT-BIH Arrhythmia Database** 完成。

平台集成了从原始 ECG 信号读取、小波降噪、心拍分割、人工特征提取（TSFEL）、传统机器学习分类（Random Forest、SVM）、1D-CNN 深度学习分类，到 PyQt6 可视化交互界面的完整流程，实现了 ECG 数据的离线训练、性能评估以及实时推理模拟功能。

---

# MIT-BIH 数据库介绍

MIT-BIH Arrhythmia Database 由麻省理工学院（MIT）与 Beth Israel Hospital 联合建立，共包含 48 组长时程 Holter 心电记录，总时长约 110 小时，采样频率为 360Hz，并提供专家逐拍标注信息。

该数据库同时提供原始 ECG 波形数据和对应的标注文件，是目前心律失常分类算法研究与验证最广泛使用的数据集之一。

本系统将心拍划分为以下五类：

* 正常心拍（N）
* 房性早搏（A）
* 左束支传导阻滞（L）
* 右束支传导阻滞（R）
* 室性早搏（V）

---

# ✨ 核心功能

### 📂 MIT-BIH 数据解析

* 基于 WFDB 工具读取 `.hea/.dat/.atr` 文件
* 自动定位 R 峰
* 以 R 峰为中心进行心拍分割
* 每个心拍长度为 300 个采样点

### 🧹 ECG 信号预处理

* db4 小波阈值降噪
* 支持两种处理模式：

  * 整段 ECG 信号处理
  * 单心拍处理

### 📊 时间序列特征工程

* TSFEL 自动提取特征
* 时域特征
* 频域特征
* 统计学特征
* 特征筛选
* PCA 降维

### 🤖 机器学习模块

#### 随机森林（Random Forest）

* 可调树数量
* 可调最大深度
* 可调叶节点样本数
* 完整性能评估

#### 支持向量机（SVM）

* RBF 核函数
* 超参数调优
* 多组参数组合对比
* 支持 C 与 gamma 参数优化

### 🧠 深度学习模块

* 基于 PyTorch 构建 1D-CNN 网络
* 端到端 ECG 分类
* 无需人工特征提取
* 支持扩展网络结构：

  * ResNet
  * LSTM
  * Transformer
* Adam 优化器
* CrossEntropy 损失函数
* Batch 训练
* 收敛曲线可视化

### 📈 综合模型评估

支持以下指标：

* Accuracy（准确率）
* Precision（精确率）
* Recall（召回率）
* F1-Score
* Specificity（特异度）
* ROC-AUC
* 混淆矩阵
* 各类别性能统计图

### 🖥️ PyQt6 可视化界面

* 一键完成完整流程
* ECG 波形对比显示
* 模型训练进度展示
* 实时 ECG 推理分析
* 分类概率分布可视化

---

# 🏗️ 系统流程

```text
MIT-BIH 原始 ECG 数据 (.dat/.hea/.atr)
                ↓
       WFDB 读取与 R 峰定位
                ↓
      单心拍分割（300点）
                ↓
          db4 小波降噪

├─ 传统机器学习流程 ─────┐
│
│ TSFEL 特征提取
│          ↓
│ 特征筛选 + PCA降维
│          ↓
│ Random Forest / SVM
│ 训练与性能评估
│
└─ 深度学习流程 ────────┘

          直接输入 1D-CNN
                  ↓
             训练与验证
                  ↓
            保存模型权重

                  ↓

      GUI 实时推理模块
      加载模型 + ECG信号
                  ↓
      输出分类结果与置信度
```

---

# 🛠️ 技术栈与依赖

## Python 环境要求

```txt
python>=3.10
```

## 主要依赖库

```txt
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
├── data/                 # MIT-BIH 数据集存放目录
├── preprocess/           # ECG读取、分割、小波降噪
├── feature/              # TSFEL特征提取与PCA降维
├── ml_models/            # Random Forest 与 SVM
├── dl_models/            # 1D-CNN网络与训练脚本
├── gui/                  # PyQt6界面与绘图逻辑
├── utils/                # 指标计算与可视化工具
├── weights/              # 保存的 .pth / .pkl 模型文件
├── main.py               # 程序入口
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

将 MIT-BIH 数据集中的 `.hea`、`.dat`、`.atr` 文件放入：

```text
./data/
```

## 启动系统

```bash
python main.py
```

---

# 📊 实验结果

## 传统机器学习

### Random Forest

```text
Accuracy = 0.8794
```

在房性早搏（A 类）上的分类效果较弱。

### 优化后的 SVM

```text
C = 1
gamma = 0.1
Accuracy = 0.8689
```

精确率与召回率较为均衡。

---

## 1D-CNN 深度学习模型

* 收敛后总体准确率接近 0.99
* N/L/R/V 类召回率均大于 0.96
* A 类仅存在少量漏检
* Loss 与 Accuracy 曲线稳定
* 无明显过拟合现象

---

## 结论

1D-CNN 能够自动挖掘 ECG 信号中的深层时序特征，在分类准确率和泛化能力方面均明显优于传统机器学习方法。

---

# 📄 许可证

本项目仅用于学术研究与课程设计，采用 MIT License 开源协议发布。
