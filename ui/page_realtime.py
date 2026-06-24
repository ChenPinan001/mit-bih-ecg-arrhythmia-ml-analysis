import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle
import torch
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QGroupBox, QComboBox, QFormLayout, QFrame, QLineEdit, QSplitter, QProgressBar
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from dl.cnn_model import MODEL_REGISTRY
from dl.train_dl import load_dl_model, predict_dl_model

LABEL_NAMES = {'N': '正常搏动', 'A': '房性早搏', 'L': '左束支阻滞', 'R': '右束支阻滞', 'V': '室性早搏'}
LABEL_COLORS = {'N': '#a6e3a1', 'A': '#fab387', 'L': '#89b4fa', 'R': '#cba6f7', 'V': '#f38ba8'}
ALL_MODEL_TYPES = list(MODEL_REGISTRY.keys()) + ["RandomForest", "SVM"]


class RealtimePage(QWidget):
    def __init__(self):
        super().__init__()
        self.ecg_signal = None
        self.current_model = None
        self.current_model_type = None
        self.device = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(12, 8, 12, 8)

        layout.addWidget(self._make_title("实时心电分类"))

        config_splitter = QSplitter(Qt.Orientation.Horizontal)

        dir_group = self._grp("数据来源")
        dir_layout = QVBoxLayout()
        dir_layout.setContentsMargins(8, 6, 8, 6)
        dir_layout.setSpacing(4)

        r1 = QHBoxLayout()
        r1.setSpacing(4)
        r1.addWidget(QLabel("MIT目录:"))
        self.data_dir_edit = QLineEdit("F:/Database/ECG")
        self.data_dir_edit.editingFinished.connect(self._scan_records)
        r1.addWidget(self.data_dir_edit, 1)
        browse_data_btn = QPushButton("浏览")
        browse_data_btn.setFixedWidth(65)
        browse_data_btn.clicked.connect(self.browse_data_dir)
        r1.addWidget(browse_data_btn)
        dir_layout.addLayout(r1)

        r2 = QHBoxLayout()
        r2.setSpacing(4)
        r2.addWidget(QLabel("记录号:"))
        self.record_combo = QComboBox()
        self.record_combo.setEditable(True)
        self.record_combo.setFixedWidth(80)
        self.record_combo.addItems([str(i) for i in range(100, 235)])
        self.record_combo.setCurrentText("100")
        r2.addWidget(self.record_combo)
        r2.addStretch()
        dir_layout.addLayout(r2)

        dir_group.setLayout(dir_layout)
        config_splitter.addWidget(dir_group)

        model_group = self._grp("模型配置")
        model_layout = QVBoxLayout()
        model_layout.setContentsMargins(8, 6, 8, 6)
        model_layout.setSpacing(4)

        r3 = QHBoxLayout()
        r3.setSpacing(4)
        r3.addWidget(QLabel("类型:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(ALL_MODEL_TYPES)
        self.model_type_combo.setFixedWidth(110)
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        r3.addWidget(self.model_type_combo)
        r3.addStretch()
        model_layout.addLayout(r3)

        r4 = QHBoxLayout()
        r4.setSpacing(4)
        r4.addWidget(QLabel("路径:"))
        self.model_path_edit = QLineEdit("F:/Database/Models/cnn_model.pth")
        r4.addWidget(self.model_path_edit, 1)
        browse_model_btn = QPushButton("浏览")
        browse_model_btn.setFixedWidth(65)
        browse_model_btn.clicked.connect(self.browse_model)
        r4.addWidget(browse_model_btn)
        model_layout.addLayout(r4)

        model_group.setLayout(model_layout)
        config_splitter.addWidget(model_group)

        config_splitter.setSizes([500, 500])
        layout.addWidget(config_splitter)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        load_signal_btn = QPushButton("加载ECG信号")
        load_signal_btn.setMinimumHeight(38)
        load_signal_btn.clicked.connect(self.load_signal)
        btn_layout.addWidget(load_signal_btn)

        load_model_btn = QPushButton("加载模型")
        load_model_btn.setMinimumHeight(38)
        load_model_btn.clicked.connect(self.load_model)
        btn_layout.addWidget(load_model_btn)

        self.predict_btn = QPushButton("预测分类")
        self.predict_btn.setMinimumHeight(38)
        self.predict_btn.setEnabled(False)
        self.predict_btn.clicked.connect(self.predict)
        btn_layout.addWidget(self.predict_btn)
        layout.addLayout(btn_layout)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #6c7086; font-size: 12px;")
        layout.addWidget(self.status_label)

        plot_group = QGroupBox("ECG 波形")
        plot_layout = QVBoxLayout()
        plot_layout.setContentsMargins(4, 8, 4, 4)
        self.ecg_plot = pg.PlotWidget()
        self.ecg_plot.setBackground('#181825')
        self.ecg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ecg_plot.setLabel('left', 'mV')
        self.ecg_plot.setLabel('bottom', 'Sample')
        self.ecg_plot.setMinimumHeight(200)
        plot_layout.addWidget(self.ecg_plot)
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group, 1)

        result_splitter = QSplitter(Qt.Orientation.Horizontal)

        result_info_widget = QWidget()
        result_info_layout = QVBoxLayout(result_info_widget)
        result_info_layout.setContentsMargins(16, 12, 12, 12)
        result_info_layout.setSpacing(4)

        title_lbl = QLabel("预测结果")
        title_lbl.setStyleSheet("color: #89b4fa; font-weight: bold; font-size: 13px;")
        result_info_layout.addWidget(title_lbl)

        self.result_label = QLabel("等待预测...")
        self.result_label.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setMinimumHeight(50)
        self.result_label.setStyleSheet("color: #6c7086; padding: 8px;")
        result_info_layout.addWidget(self.result_label)

        result_info_layout.addSpacing(8)

        result_info_layout.addSpacing(4)

        self.detail_bars = {}
        self.detail_labels = {}
        for code, cn_name in LABEL_NAMES.items():
            row_widget = QWidget()
            row_widget.setFixedHeight(24)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            name_lbl = QLabel(f"{code}-{cn_name}")
            name_lbl.setFixedWidth(110)
            name_lbl.setStyleSheet(f"color: {LABEL_COLORS[code]}; font-size: 12px; font-weight: bold;")
            row_layout.addWidget(name_lbl)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(12)
            bar.setTextVisible(False)
            bar.setStyleSheet(f"""
                QProgressBar {{
                    border: none;
                    background-color: #313244;
                    border-radius: 3px;
                }}
                QProgressBar::chunk {{
                    background-color: {LABEL_COLORS[code]};
                    border-radius: 3px;
                }}
            """)
            row_layout.addWidget(bar, 1)
            self.detail_bars[code] = bar

            pct_lbl = QLabel("0.0%")
            pct_lbl.setFixedWidth(45)
            pct_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            pct_lbl.setStyleSheet("font-size: 11px; color: #cdd6f4; font-weight: bold;")
            row_layout.addWidget(pct_lbl)
            self.detail_labels[code] = pct_lbl

            result_info_layout.addWidget(row_widget)

        result_info_layout.addStretch()
        result_splitter.addWidget(result_info_widget)

        prob_group = QGroupBox("概率分布")
        prob_layout = QVBoxLayout()
        prob_layout.setContentsMargins(4, 8, 4, 4)
        self.prob_figure = Figure(figsize=(6, 4), facecolor='#1e1e2e')
        self.prob_canvas = FigureCanvas(self.prob_figure)
        prob_layout.addWidget(self.prob_canvas)
        prob_group.setLayout(prob_layout)
        result_splitter.addWidget(prob_group)

        result_splitter.setSizes([450, 450])
        layout.addWidget(result_splitter, 1)

    def _make_title(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Microsoft YaHei", 16, QFont.Weight.Bold))
        lbl.setStyleSheet("color: #89b4fa; padding: 4px 0px;")
        return lbl

    def _grp(self, title):
        g = QGroupBox(title)
        g.setStyleSheet("""
            QGroupBox { color: #89b4fa; border: 1px solid #45475a; border-radius: 4px;
                         margin-top: 14px; padding: 4px 8px 4px 8px; font-weight: bold; font-size: 13px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
                               left: 12px; top: 1px; padding: 0 4px; }
        """)
        return g

    def on_model_type_changed(self, text):
        if text in MODEL_REGISTRY:
            safe = text.lower().replace('1d', '').replace(' ', '_')
            self.model_path_edit.setText(f"F:/Database/Models/{safe}_model.pth")
        elif text == 'RandomForest':
            self.model_path_edit.setText("F:/Database/Models/random_forest.pkl")
        elif text == 'SVM':
            self.model_path_edit.setText("F:/Database/Models/svm.pkl")

    def browse_data_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择 MIT-BIH 数据目录")
        if d:
            self.data_dir_edit.setText(d)
            self._scan_records()

    def _scan_records(self):
        data_dir = self.data_dir_edit.text()
        self.record_combo.clear()
        if data_dir and os.path.isdir(data_dir):
            records = sorted([os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.hea')])
            self.record_combo.addItems(records)
        if self.record_combo.count() == 0:
            self.record_combo.addItems([str(i) for i in range(100, 235)])

    def browse_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "F:/Database/Models", "Model files (*.pkl *.pth);;All (*)")
        if f:
            self.model_path_edit.setText(f)

    def load_signal(self):
        data_dir = self.data_dir_edit.text()
        record_name = self.record_combo.currentText().strip()
        try:
            if data_dir and record_name and os.path.isdir(data_dir):
                from utils.data_loader import load_mit_bih_record
                data = load_mit_bih_record(data_dir, record_name)
                signal = data['signal'][:300]
                self.status_label.setText(f"已加载记录 {record_name} ({len(data['signal'])} samples)")
            else:
                signal = np.sin(np.linspace(0, 4 * np.pi, 300)) + 0.3 * np.random.randn(300)
                self.status_label.setText("已加载模拟信号")

            self.ecg_signal = signal
            self.plot_ecg(signal)
            self.predict_btn.setEnabled(True)
            self.result_label.setText("信号已加载，点击预测")
            self.result_label.setStyleSheet("color: #6c7086; font-size: 20px; font-weight: bold; padding: 8px;")
        except Exception as e:
            self.status_label.setText(f"加载错误: {str(e)}")

    def plot_ecg(self, signal):
        self.ecg_plot.clear()
        pen = pg.mkPen(color='#89b4fa', width=1.5)
        self.ecg_plot.plot(signal, pen=pen)
        self.ecg_plot.autoRange()

    def load_model(self):
        model_type = self.model_type_combo.currentText()
        model_path = self.model_path_edit.text()
        try:
            if model_type in MODEL_REGISTRY:
                self.current_model, self.device = load_dl_model(model_type, model_path, input_length=300)
                self.current_model_type = 'dl'
            else:
                with open(model_path, 'rb') as f:
                    self.current_model = pickle.load(f)
                self.current_model_type = 'ml'
            self.status_label.setText(f"{model_type} 模型已加载: {model_path}")
        except Exception as e:
            self.status_label.setText(f"模型加载失败: {str(e)}")

    def predict(self):
        if self.ecg_signal is None or self.current_model is None:
            self.status_label.setText("请先加载信号和模型")
            return
        try:
            signal = self.ecg_signal.copy()
            if len(signal) != 300:
                signal = signal[:300] if len(signal) > 300 else np.pad(signal, (0, 300 - len(signal)))

            if self.current_model_type == 'dl':
                pred_idx, prob_values = predict_dl_model(self.current_model, self.device, signal.reshape(1, -1))
                pred_label = ['N', 'A', 'L', 'R', 'V'][pred_idx[0]]
                prob_values = prob_values[0]
            else:
                X = np.nan_to_num(signal.reshape(1, -1), nan=0.0)
                pred_label = self.current_model.predict(X)[0]
                prob_values = self.current_model.predict_proba(X)[0]

            cn_name = LABEL_NAMES.get(pred_label, pred_label)
            color = LABEL_COLORS.get(pred_label, '#cdd6f4')
            self.result_label.setText(f"{pred_label} - {cn_name}")
            self.result_label.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: bold; padding: 8px;")

            classes = ['N', 'A', 'L', 'R', 'V']
            max_prob = float(max(prob_values))
            for i, cls in enumerate(classes):
                pct = prob_values[i] * 100
                self.detail_labels[cls].setText(f"{pct:.1f}%")
                self.detail_bars[cls].setValue(int(pct))

            self.plot_probability(prob_values, classes)
            self.status_label.setText(f"预测完成: {pred_label} - {cn_name} (置信度: {max_prob*100:.1f}%)")
        except Exception as e:
            import traceback
            self.status_label.setText(f"预测错误: {str(e)}")
            print(traceback.format_exc())

    def plot_probability(self, probs, classes):
        self.prob_figure.clear()
        ax = self.prob_figure.add_subplot(111)
        cn_names = [f"{c}-{LABEL_NAMES[c]}" for c in classes]
        colors = [LABEL_COLORS[c] for c in classes]
        bars = ax.bar(cn_names, probs, color=colors, width=0.6, edgecolor='#45475a', linewidth=0.5)
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{prob*100:.1f}%', ha='center', va='bottom', color='#cdd6f4', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.2)
        ax.set_title('预测概率分布', color='#cdd6f4', fontsize=13)
        ax.set_ylabel('概率', color='#cdd6f4', fontsize=11)
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#cdd6f4', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#45475a')
        self.prob_figure.tight_layout()
        self.prob_canvas.draw()
