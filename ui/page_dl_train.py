import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QGroupBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QFileDialog, QTabWidget, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from dl.cnn_model import MODEL_REGISTRY
from dl.train_dl import train_dl_model, plot_dl_results
from ui.expanding_tab import ExpandingTabWidget


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=5):
        self.fig = Figure(figsize=(width, height), facecolor='#1e1e2e')
        super().__init__(self.fig)
        self.setParent(parent)
        from PyQt6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(200)


class DLTrainWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, model_name, train_csv, val_csv, output_dir, params):
        super().__init__()
        self.model_name = model_name
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.output_dir = output_dir
        self.params = params

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            model, results = train_dl_model(
                self.model_name, self.train_csv, self.val_csv,
                self.output_dir, self.params, self.progress.emit
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class DLTrainPage(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.plot_canvases = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 4, 8, 4)

        layout.addWidget(self._make_title("模型训练"))

        dir_group = self._grp("数据目录")
        dir_layout = QHBoxLayout()
        dir_layout.setContentsMargins(16, 8, 8, 8)
        self.data_dir_edit = QLineEdit("F:/Database/Processed")
        dir_layout.addWidget(self.data_dir_edit, 1)
        browse_btn = QPushButton("浏览")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self.browse_data)
        dir_layout.addWidget(browse_btn)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        hint = QLabel("提示: 请先完成数据预处理，确保目录下有 train.csv 和 val.csv")
        hint.setStyleSheet("color: #fab387; font-size: 12px; padding-left: 20px;")
        layout.addWidget(hint)

        config_group = self._grp("训练配置")
        config_group.setStyleSheet("""
            QGroupBox { color: #89b4fa; border: 1px solid #45475a; border-radius: 4px;
                         margin-top: 14px; padding: 4px 8px 4px 8px; font-weight: bold; font-size: 13px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
                               left: 12px; top: 1px; padding: 0 4px; }
        """)
        config_layout = QHBoxLayout()
        config_layout.setContentsMargins(12, 4, 8, 4)
        config_layout.setSpacing(12)

        config_layout.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(MODEL_REGISTRY.keys()))
        self.model_combo.setFixedWidth(100)
        config_layout.addWidget(self.model_combo)

        config_layout.addWidget(QLabel("训练轮次:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500); self.epochs_spin.setValue(30)
        self.epochs_spin.setFixedWidth(80)
        config_layout.addWidget(self.epochs_spin)

        config_layout.addWidget(QLabel("学习率:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1); self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5); self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setFixedWidth(105)
        self.lr_spin.setStyleSheet("padding-left: 2px; padding-right: 2px;")
        config_layout.addWidget(self.lr_spin)

        config_layout.addWidget(QLabel("批次:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(16, 1024); self.batch_spin.setValue(256)
        self.batch_spin.setSingleStep(16);         self.batch_spin.setFixedWidth(90)
        config_layout.addWidget(self.batch_spin)

        config_layout.addWidget(QLabel("随机种子:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999); self.seed_spin.setValue(42)
        self.seed_spin.setFixedWidth(80)
        config_layout.addWidget(self.seed_spin)

        config_layout.addWidget(QLabel("设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["GPU", "CPU"])
        self.device_combo.setFixedWidth(80)
        config_layout.addWidget(self.device_combo)

        config_layout.addWidget(QLabel("优化器:"))
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop", "AdamW"])
        self.optimizer_combo.setFixedWidth(90)
        config_layout.addWidget(self.optimizer_combo)

        config_layout.addWidget(QLabel("损失函数:"))
        self.loss_combo = QComboBox()
        self.loss_combo.addItems([
            "CrossEntropy (多分类)",
            "WeightedCrossEntropy (多分类)",
            "FocalLoss (多分类)",
            "BCEWithLogitsLoss (二分类)"
        ])
        self.loss_combo.setFixedWidth(180)
        config_layout.addWidget(self.loss_combo)

        config_layout.addStretch()
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        self.train_btn = QPushButton("开始训练")
        self.train_btn.setMinimumHeight(34)
        self.train_btn.clicked.connect(self.start_training)
        btn_layout.addWidget(self.train_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setMinimumHeight(34)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        btn_layout.addWidget(self.stop_btn)
        self.save_btn = QPushButton("保存模型")
        self.save_btn.setMinimumHeight(34)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(14)
        layout.addWidget(self.progress_bar)

        self.results_tabs = ExpandingTabWidget()

        tab_names = ["训练日志", "Loss曲线", "准确率", "召回率", "F1分数", "混淆矩阵"]
        self.tab_contents = {}
        for name in tab_names:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tab_layout.setContentsMargins(4, 4, 4, 4)
            tab_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tab_contents[name] = tab
            self.results_tabs.addTab(tab, name)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tab_contents["训练日志"].layout().addWidget(self.log_text)

        for name in ["Loss曲线", "准确率", "召回率", "F1分数", "混淆矩阵"]:
            canvas = PlotCanvas(width=8, height=5)
            self.plot_canvases[name] = canvas
            wrapper = QWidget()
            wl = QVBoxLayout(wrapper)
            wl.setContentsMargins(0, 0, 0, 0)
            wl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            wl.addWidget(canvas)
            export_btn = QPushButton("导出图片")
            export_btn.setFixedWidth(100)
            export_btn.clicked.connect(lambda checked, n=name: self.export_plot(n))
            wl.addWidget(export_btn, alignment=Qt.AlignmentFlag.AlignCenter)
            self.tab_contents[name].layout().addWidget(wrapper)

        layout.addWidget(self.results_tabs, 1)

    def _make_title(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        lbl.setStyleSheet("color: #89b4fa; padding: 2px 0px;")
        return lbl

    def _grp(self, title):
        g = QGroupBox(title)
        g.setStyleSheet("""
            QGroupBox { color: #89b4fa; border: 1px solid #45475a; border-radius: 4px;
                         margin-top: 16px; font-weight: bold; font-size: 13px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
                               left: 12px; top: 1px; padding: 0 4px; }
        """)
        return g

    def browse_data(self):
        d = QFileDialog.getExistingDirectory(self, "选择数据目录")
        if d:
            self.data_dir_edit.setText(d)

    def start_training(self):
        data_dir = self.data_dir_edit.text()
        train_csv = os.path.join(data_dir, "train.csv")
        val_csv = os.path.join(data_dir, "val.csv")
        if not os.path.exists(train_csv) or not os.path.exists(val_csv):
            self.log_text.append("错误: 数据目录中未找到 train.csv / val.csv")
            return

        params = {
            'batch_size': self.batch_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'epochs': self.epochs_spin.value(),
            'seed': self.seed_spin.value(),
            'device': self.device_combo.currentText(),
            'optimizer': self.optimizer_combo.currentText(),
            'loss_function': self.loss_combo.currentText()
        }

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.log_text.clear()
        self.log_text.append(f"开始训练 {self.model_combo.currentText()}...")
        self.progress_bar.setValue(0)

        self.worker = DLTrainWorker(
            self.model_combo.currentText(), train_csv, val_csv, "F:/Database/Models", params
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def stop_training(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.log_text.append("训练已停止")

    def on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.log_text.append(message)

    def on_finished(self, results):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.log_text.append(f"\n=== {results['model_name']} 训练完成 ===")
        self.log_text.append(f"准确率: {results['accuracy']:.4f}")
        self.log_text.append(f"精确率: {results['precision']:.4f}")
        self.log_text.append(f"召回率: {results['recall']:.4f}")
        self.log_text.append(f"F1分数: {results['f1']:.4f}")
        self.log_text.append(f"ROC AUC: {results['roc_auc']:.4f}")
        self.log_text.append(f"设备: {results['device']}")
        self.log_text.append(f"模型已保存: {results['model_path']}")

        self._plot_loss(results)
        self._plot_accuracy(results)
        self._plot_recall(results)
        self._plot_f1(results)
        self._plot_confusion(results)

    def _plot_loss(self, results):
        canvas = self.plot_canvases["Loss曲线"]
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        ax.plot(results['train_losses'], label='Train Loss', color='#89b4fa', linewidth=2)
        ax.plot(results['val_losses'], label='Val Loss', color='#f38ba8', linewidth=2)
        ax.set_xlabel('Epoch', color='#cdd6f4', fontsize=12)
        ax.set_ylabel('Loss', color='#cdd6f4', fontsize=12)
        ax.set_title('Training & Validation Loss', color='#cdd6f4', fontsize=14)
        ax.legend(fontsize=11, facecolor='#313244', edgecolor='#45475a', labelcolor='#cdd6f4')
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#cdd6f4')
        for spine in ax.spines.values(): spine.set_color('#45475a')
        canvas.fig.tight_layout(rect=[0, 0, 1, 0.95])
        canvas.draw()

    def _plot_accuracy(self, results):
        canvas = self.plot_canvases["准确率"]
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        ax.plot(results['val_accuracies'], label='Val Accuracy', color='#a6e3a1', linewidth=2)
        ax.set_xlabel('Epoch', color='#cdd6f4', fontsize=12)
        ax.set_ylabel('Accuracy', color='#cdd6f4', fontsize=12)
        ax.set_title('Validation Accuracy', color='#cdd6f4', fontsize=14)
        ax.legend(fontsize=11, facecolor='#313244', edgecolor='#45475a', labelcolor='#cdd6f4')
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#cdd6f4')
        for spine in ax.spines.values(): spine.set_color('#45475a')
        canvas.fig.tight_layout(rect=[0, 0, 1, 0.95])
        canvas.draw()

    def _plot_recall(self, results):
        canvas = self.plot_canvases["召回率"]
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        from sklearn.metrics import recall_score
        cm = results['confusion_matrix']
        classes = results['classes']
        recalls = [cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0 for i in range(len(classes))]
        bars = ax.bar(classes, recalls, color=['#a6e3a1', '#fab387', '#89b4fa', '#cba6f7', '#f38ba8'])
        ax.set_ylim(0, 1.1)
        ax.set_title('召回率 (Recall) per Class', color='#cdd6f4', fontsize=14)
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#cdd6f4')
        for bar, v in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.03, f'{v:.3f}', ha='center', color='#cdd6f4', fontsize=10)
        for spine in ax.spines.values(): spine.set_color('#45475a')
        canvas.fig.tight_layout(rect=[0, 0, 1, 0.95])
        canvas.draw()

    def _plot_f1(self, results):
        canvas = self.plot_canvases["F1分数"]
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        cm = results['confusion_matrix']
        classes = results['classes']
        f1s = []
        for i in range(len(classes)):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1s.append(f1)
        bars = ax.bar(classes, f1s, color=['#a6e3a1', '#fab387', '#89b4fa', '#cba6f7', '#f38ba8'])
        ax.set_ylim(0, 1.1)
        ax.set_title('F1分数 per Class', color='#cdd6f4', fontsize=14)
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#cdd6f4')
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.03, f'{v:.3f}', ha='center', color='#cdd6f4', fontsize=10)
        for spine in ax.spines.values(): spine.set_color('#45475a')
        canvas.fig.tight_layout(rect=[0, 0, 1, 0.95])
        canvas.draw()

    def _plot_confusion(self, results):
        canvas = self.plot_canvases["混淆矩阵"]
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        cm = results['confusion_matrix']
        classes = results['classes']
        if cm.ndim != 2 or cm.shape[0] != len(classes):
            ax.text(0.5, 0.5, '混淆矩阵数据异常', ha='center', va='center',
                    color='#f38ba8', fontsize=14, transform=ax.transAxes)
            ax.set_facecolor('#181825')
            canvas.fig.tight_layout(rect=[0, 0, 1, 0.95])
            canvas.draw()
            return
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title('混淆矩阵 (Confusion Matrix)', color='#cdd6f4', fontsize=14)
        ax.set_xlabel('Predicted', color='#cdd6f4', fontsize=12)
        ax.set_ylabel('True', color='#cdd6f4', fontsize=12)
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, color='#cdd6f4')
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, color='#cdd6f4')
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=11)
        ax.set_facecolor('#181825')
        canvas.fig.tight_layout(rect=[0, 0, 1, 0.95])
        canvas.draw()

    def export_plot(self, name):
        canvas = self.plot_canvases.get(name)
        if not canvas:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出图片", f"{name}.png", "PNG (*.png);;PDF (*.pdf)")
        if path:
            canvas.fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=canvas.fig.get_facecolor())

    def on_error(self, msg):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append(f"错误: {msg}")
