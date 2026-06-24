import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle
import torch
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QGroupBox, QComboBox,
    QFileDialog, QFormLayout, QGridLayout, QTableWidget, QTableWidgetItem,
    QHeaderView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from ui.expanding_tab import ExpandingTabWidget


class TestWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, model_type, model_path, test_csv, scaler_path=None):
        super().__init__()
        self.model_type = model_type
        self.model_path = model_path
        self.test_csv = test_csv
        self.scaler_path = scaler_path

    def run(self):
        try:
            self.progress.emit(10, "Loading test data...")
            df = pd.read_csv(self.test_csv)
            y_true = df['label'].values
            X = df.drop(columns=['label', 'sample_id']).values if 'sample_id' in df.columns else df.drop(columns=['label']).values
            X = np.nan_to_num(X, nan=0.0)

            self.progress.emit(30, "Loading model...")
            classes = ['N', 'A', 'L', 'R', 'V']

            if self.model_type == 'random_forest':
                with open(self.model_path, 'rb') as f:
                    model = pickle.load(f)
                y_pred = model.predict(X)
                y_proba = model.predict_proba(X)

            elif self.model_type == 'svm':
                with open(self.model_path, 'rb') as f:
                    model = pickle.load(f)
                if self.scaler_path and os.path.exists(self.scaler_path):
                    with open(self.scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    X = scaler.transform(X)
                y_pred = model.predict(X)
                y_proba = model.predict_proba(X)

            elif self.model_type == 'cnn':
                self.progress.emit(50, "Loading CNN model...")
                from dl.cnn_model import ECG_CNN
                from dl.train_dl import load_cnn_model, predict_cnn
                model, device = load_cnn_model(self.model_path, input_length=X.shape[1])
                X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
                y_pred_idx, y_proba = predict_cnn(model, device, X)
                y_pred = np.array([classes[i] for i in y_pred_idx])

            self.progress.emit(70, "Calculating metrics...")
            accuracy = accuracy_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            cm = confusion_matrix(y_true, y_pred, labels=classes)
            y_true_bin = label_binarize(y_true, classes=classes)
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
            roc_auc_val = auc(fpr, tpr)

            self.progress.emit(90, "Generating results...")

            per_class = {}
            for i, cls in enumerate(classes):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn
                per_class[cls] = {
                    'precision': precision_score(y_true, y_pred, labels=[cls], average='micro', zero_division=0),
                    'recall': recall_score(y_true, y_pred, labels=[cls], average='micro', zero_division=0),
                    'f1': f1_score(y_true, y_pred, labels=[cls], average='micro', zero_division=0),
                    'support': int(cm[i, :].sum())
                }

            sample_indices = np.random.choice(len(y_true), min(20, len(y_true)), replace=False)

            results = {
                'accuracy': accuracy,
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'confusion_matrix': cm,
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc_val,
                'classes': classes,
                'per_class': per_class,
                'y_true': y_true[sample_indices],
                'y_pred': y_pred[sample_indices] if isinstance(y_pred, np.ndarray) else np.array(y_pred)[sample_indices],
                'y_proba': y_proba[sample_indices] if isinstance(y_proba, np.ndarray) else np.array(y_proba)[sample_indices],
                'total_samples': len(y_true),
                'model_type': self.model_type
            }

            self.progress.emit(100, "Test complete!")
            self.finished.emit(results)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class ModelTestPage(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        config_group = self._grp("测试配置")
        config_layout = QGridLayout()
        config_layout.setContentsMargins(12, 8, 8, 8)
        config_layout.setSpacing(8)

        config_layout.addWidget(QLabel("模型类型:"), 0, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["随机森林 (Random Forest)", "支持向量机 (SVM)", "CNN (深度学习)"])
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        config_layout.addWidget(self.model_type_combo, 0, 1)

        config_layout.addWidget(QLabel("模型路径:"), 0, 2)
        model_row = QHBoxLayout()
        self.model_path_combo = QComboBox()
        self.model_path_combo.setEditable(True)
        self.model_path_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.model_path_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._scan_models()
        model_row.addWidget(self.model_path_combo, 1)
        browse_model_btn = QPushButton("浏览")
        browse_model_btn.setFixedWidth(70)
        browse_model_btn.clicked.connect(self.browse_model)
        model_row.addWidget(browse_model_btn)
        config_layout.addLayout(model_row, 0, 3)

        config_layout.addWidget(QLabel("Scaler路径:"), 1, 0)
        self.scaler_path_combo = QComboBox()
        self.scaler_path_combo.setEditable(True)
        self.scaler_path_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.scaler_path_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._scan_scalers()
        config_layout.addWidget(self.scaler_path_combo, 1, 1)

        config_layout.addWidget(QLabel("测试数据:"), 1, 2)
        test_row = QHBoxLayout()
        self.test_csv_combo = QComboBox()
        self.test_csv_combo.setEditable(True)
        self.test_csv_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.test_csv_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._scan_test_data()
        test_row.addWidget(self.test_csv_combo, 1)
        browse_test_btn = QPushButton("浏览")
        browse_test_btn.setFixedWidth(70)
        browse_test_btn.clicked.connect(self.browse_test)
        test_row.addWidget(browse_test_btn)
        config_layout.addLayout(test_row, 1, 3)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        metrics_group = self._grp("总体指标")
        metrics_group.setStyleSheet("""
            QGroupBox { color: #89b4fa; border: 1px solid #45475a; border-radius: 4px;
                         margin-top: 12px; padding: 2px 8px 2px 8px; font-weight: bold; font-size: 12px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
                               left: 12px; top: 0px; padding: 0 4px; }
        """)
        metrics_inner = QVBoxLayout()
        metrics_inner.setContentsMargins(4, 2, 4, 2)
        metrics_inner.setSpacing(2)

        btn_progress_row = QHBoxLayout()
        btn_progress_row.setSpacing(8)
        self.test_btn = QPushButton("开始测试")
        self.test_btn.setMinimumHeight(30)
        self.test_btn.clicked.connect(self.start_testing)
        btn_progress_row.addWidget(self.test_btn)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(12)
        btn_progress_row.addWidget(self.progress_bar, 1)
        metrics_inner.addLayout(btn_progress_row)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        metrics_inner.addWidget(self.status_label)

        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(0)
        self.metric_labels = {}
        for name in ['准确率', '精确率', '召回率', 'F1分数', 'ROC AUC']:
            lbl = QLabel(f"{name}: -")
            lbl.setStyleSheet("font-size: 14px; font-weight: bold; padding: 4px 8px; color: #89b4fa;")
            metrics_layout.addWidget(lbl)
            metrics_layout.addStretch(1)
            self.metric_labels[name] = lbl
        metrics_inner.addLayout(metrics_layout)

        metrics_group.setLayout(metrics_inner)
        layout.addWidget(metrics_group)

        self.results_tabs = ExpandingTabWidget()
        layout.addWidget(self.results_tabs, 1)

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        self.results_tabs.addTab(log_tab, "测试日志")

        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        self.results_figure = Figure(figsize=(12, 5), facecolor='#1e1e2e')
        self.results_canvas = FigureCanvas(self.results_figure)
        from PyQt6.QtWidgets import QSizePolicy
        self.results_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.results_canvas.setMinimumHeight(200)
        chart_layout.addWidget(self.results_canvas)
        self.results_tabs.addTab(chart_tab, "评估图表")

        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        self.detail_table = QTableWidget()
        self.detail_table.setColumnCount(5)
        self.detail_table.setHorizontalHeaderLabels(['类别', 'Precision', 'Recall', 'F1-Score', 'Support'])
        self.detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.detail_table.verticalHeader().setVisible(False)
        self.detail_table.setStyleSheet("""
            QTableWidget { background-color: #181825; color: #cdd6f4; border: 1px solid #45475a;
                           gridline-color: #313244; }
            QTableWidget::item:selected { background-color: #45475a; }
            QHeaderView::section { background-color: #313244; color: #89b4fa;
                                   border: 1px solid #45475a; padding: 5px; font-weight: bold; }
        """)
        table_layout.addWidget(self.detail_table)
        self.results_tabs.addTab(table_tab, "分类详情")

        pred_tab = QWidget()
        pred_layout = QVBoxLayout(pred_tab)
        self.pred_table = QTableWidget()
        self.pred_table.setColumnCount(3)
        self.pred_table.setHorizontalHeaderLabels(['样本', '真实标签', '预测标签'])
        self.pred_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.pred_table.verticalHeader().setVisible(False)
        self.pred_table.setStyleSheet("""
            QTableWidget { background-color: #181825; color: #cdd6f4; border: 1px solid #45475a;
                           gridline-color: #313244; }
            QTableWidget::item:selected { background-color: #45475a; }
            QHeaderView::section { background-color: #313244; color: #89b4fa;
                                   border: 1px solid #45475a; padding: 5px; font-weight: bold; }
        """)
        pred_layout.addWidget(self.pred_table)
        self.results_tabs.addTab(pred_tab, "预测示例")

    def _grp(self, title):
        g = QGroupBox(title)
        g.setStyleSheet("""
            QGroupBox { color: #89b4fa; border: 1px solid #45475a; border-radius: 4px;
                         margin-top: 16px; font-weight: bold; font-size: 13px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
                               left: 12px; top: 1px; padding: 0 4px; }
        """)
        return g

    def _scan_models(self):
        models_dir = "F:/Database/Models"
        if os.path.isdir(models_dir):
            files = [f for f in os.listdir(models_dir) if f.endswith(('.pkl', '.pth'))]
            files.sort()
            self.model_path_combo.addItems([os.path.join(models_dir, f) for f in files])
        if self.model_path_combo.count() == 0:
            self.model_path_combo.addItem("F:/Database/Models/random_forest.pkl")

    def _scan_scalers(self):
        scaler_dir = "F:/Database/Scaler"
        if os.path.isdir(scaler_dir):
            files = [f for f in os.listdir(scaler_dir) if f.endswith('.pkl')]
            files.sort()
            self.scaler_path_combo.addItems([os.path.join(scaler_dir, f) for f in files])
        if self.scaler_path_combo.count() == 0:
            self.scaler_path_combo.addItem("F:/Database/Scaler/svm_scaler.pkl")

    def _scan_test_data(self):
        feat_dir = "F:/Database/Feature"
        if os.path.isdir(feat_dir):
            files = [f for f in os.listdir(feat_dir) if f.endswith('.csv') and 'feature' in f]
            files.sort()
            self.test_csv_combo.addItems([os.path.join(feat_dir, f) for f in files])
        if self.test_csv_combo.count() == 0:
            self.test_csv_combo.addItem("F:/Database/Feature/test_feature.csv")

    def on_model_type_changed(self, text):
        if 'SVM' in text:
            self.scaler_path_combo.setEnabled(True)
        else:
            self.scaler_path_combo.setEnabled(False)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "F:/Database/Models", "Model files (*.pkl *.pth);;All (*)")
        if file_path:
            self.model_path_combo.setCurrentText(file_path)

    def browse_test(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择测试数据", "F:/Database/Feature", "CSV files (*.csv)")
        if file_path:
            self.test_csv_combo.setCurrentText(file_path)

    def start_testing(self):
        model_types = ['random_forest', 'svm', 'cnn']
        model_type = model_types[self.model_type_combo.currentIndex()]
        model_path = self.model_path_combo.currentText()
        test_csv = self.test_csv_combo.currentText()
        scaler_path = self.scaler_path_combo.currentText()

        if not os.path.exists(model_path):
            self.log_text.append(f"错误: 模型文件不存在: {model_path}")
            return
        if not os.path.exists(test_csv):
            self.log_text.append(f"错误: 测试数据不存在: {test_csv}")
            return

        self.test_btn.setEnabled(False)
        self.log_text.clear()
        self.log_text.append(f"开始测试 {model_type}...")
        self.progress_bar.setValue(0)

        self.worker = TestWorker(model_type, model_path, test_csv, scaler_path)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def on_finished(self, results):
        self.test_btn.setEnabled(True)
        self.log_text.append("\n=== 测试完成 ===")
        self.log_text.append(f"模型类型: {results['model_type']}")
        self.log_text.append(f"测试样本数: {results['total_samples']}")
        self.log_text.append(f"准确率: {results['accuracy']:.4f}")
        self.log_text.append(f"精确率: {results['precision']:.4f}")
        self.log_text.append(f"召回率: {results['recall']:.4f}")
        self.log_text.append(f"F1分数: {results['f1']:.4f}")
        self.log_text.append(f"ROC AUC: {results['roc_auc']:.4f}")

        self.metric_labels['准确率'].setText(f"准确率: {results['accuracy']:.4f}")
        self.metric_labels['精确率'].setText(f"精确率: {results['precision']:.4f}")
        self.metric_labels['召回率'].setText(f"召回率: {results['recall']:.4f}")
        self.metric_labels['F1分数'].setText(f"F1分数: {results['f1']:.4f}")
        self.metric_labels['ROC AUC'].setText(f"ROC AUC: {results['roc_auc']:.4f}")

        self.plot_results(results)
        self.fill_detail_table(results)
        self.fill_pred_table(results)

        self.log_text.append(f"\n=== 测试完成 ===")
        self.log_text.append(f"准确率: {results['accuracy']:.4f}")
        self.log_text.append(f"精确率: {results['precision']:.4f}")
        self.log_text.append(f"召回率: {results['recall']:.4f}")
        self.log_text.append(f"F1分数: {results['f1']:.4f}")
        self.log_text.append(f"ROC AUC: {results['roc_auc']:.4f}")

        self.status_label.setText("")

    def plot_results(self, results):
        self.results_figure.clear()
        ax1 = self.results_figure.add_subplot(131)
        ax2 = self.results_figure.add_subplot(132)
        ax3 = self.results_figure.add_subplot(133)

        cm = results['confusion_matrix']
        classes = results['classes']
        im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
        ax1.set_title('混淆矩阵', color='#cdd6f4', fontsize=11)
        ax1.set_xlabel('Predicted', color='#cdd6f4', fontsize=9)
        ax1.set_ylabel('True', color='#cdd6f4', fontsize=9)
        ax1.set_xticks(range(len(classes))); ax1.set_xticklabels(classes, color='#cdd6f4', fontsize=9)
        ax1.set_yticks(range(len(classes))); ax1.set_yticklabels(classes, color='#cdd6f4', fontsize=9)
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black", fontsize=8)
        ax1.set_facecolor('#181825')
        for spine in ax1.spines.values(): spine.set_color('#45475a')

        ax2.plot(results['fpr'], results['tpr'], label=f'AUC = {results["roc_auc"]:.4f}', color='#89b4fa', linewidth=2)
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_title('ROC 曲线', color='#cdd6f4', fontsize=11)
        ax2.set_xlabel('FPR', color='#cdd6f4', fontsize=9)
        ax2.set_ylabel('TPR', color='#cdd6f4', fontsize=9)
        ax2.legend(facecolor='#313244', edgecolor='#45475a', labelcolor='#cdd6f4', fontsize=9)
        ax2.set_facecolor('#181825')
        ax2.tick_params(colors='#cdd6f4')
        for spine in ax2.spines.values(): spine.set_color('#45475a')

        per_class = results['per_class']
        cls_names = list(per_class.keys())
        cls_f1 = [per_class[c]['f1'] for c in cls_names]
        colors = ['#a6e3a1', '#89b4fa', '#fab387', '#f38ba8', '#cba6f7']
        ax3.bar(cls_names, cls_f1, color=colors[:len(cls_names)])
        ax3.set_ylim(0, 1.1)
        ax3.set_title('Per-class F1', color='#cdd6f4', fontsize=11)
        ax3.set_facecolor('#181825')
        ax3.tick_params(colors='#cdd6f4')
        for i, v in enumerate(cls_f1):
            ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', color='#cdd6f4', fontsize=9)
        for spine in ax3.spines.values(): spine.set_color('#45475a')

        self.results_figure.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])
        self.results_canvas.draw()
        self.results_canvas.draw()

    def fill_detail_table(self, results):
        per_class = results['per_class']
        self.detail_table.setRowCount(len(per_class))
        for i, (cls, metrics) in enumerate(per_class.items()):
            self.detail_table.setItem(i, 0, QTableWidgetItem(cls))
            self.detail_table.setItem(i, 1, QTableWidgetItem(f"{metrics['precision']:.4f}"))
            self.detail_table.setItem(i, 2, QTableWidgetItem(f"{metrics['recall']:.4f}"))
            self.detail_table.setItem(i, 3, QTableWidgetItem(f"{metrics['f1']:.4f}"))
            self.detail_table.setItem(i, 4, QTableWidgetItem(str(metrics['support'])))

    def fill_pred_table(self, results):
        y_true = results['y_true']
        y_pred = results['y_pred']
        self.pred_table.setRowCount(len(y_true))
        for i in range(len(y_true)):
            self.pred_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.pred_table.setItem(i, 1, QTableWidgetItem(str(y_true[i])))
            self.pred_table.setItem(i, 2, QTableWidgetItem(str(y_pred[i])))

    def on_error(self, msg):
        self.test_btn.setEnabled(True)
        self.log_text.append(f"错误: {msg}")
        self.status_label.setText(f"错误: {msg}")
