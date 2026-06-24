import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QGroupBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QFileDialog, QLineEdit, QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from ml.random_forest import train_random_forest, save_rf_model
from ml.svm_model import train_svm, save_svm_model
from ui.expanding_tab import ExpandingTabWidget

ML_MODELS = {'RandomForest': 'random_forest', 'SVM': 'svm'}


class MLTrainWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str, dict, dict)
    error = pyqtSignal(str)

    def __init__(self, model_type, train_path, val_path, output_dir, params):
        super().__init__()
        self.model_type = model_type
        self.train_path = train_path
        self.val_path = val_path
        self.output_dir = output_dir
        self.params = params

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            if self.model_type == 'random_forest':
                model, results = train_random_forest(
                    self.train_path, self.val_path, self.params, self.progress.emit
                )
                model_path = os.path.join(self.output_dir, 'random_forest.pkl')
                save_rf_model(model, model_path)
                self.finished.emit('random_forest', results, {'model_path': model_path})
            elif self.model_type == 'svm':
                model, scaler, results = train_svm(
                    self.train_path, self.val_path, self.params, self.progress.emit
                )
                model_path = os.path.join(self.output_dir, 'svm.pkl')
                scaler_path = os.path.join(self.output_dir, 'svm_scaler.pkl')
                save_svm_model(model, scaler, model_path, scaler_path)
                self.finished.emit('svm', results, {'model_path': model_path, 'scaler_path': scaler_path})
        except Exception as e:
            self.error.emit(str(e))


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=5):
        self.fig = Figure(figsize=(width, height), facecolor='#1e1e2e')
        super().__init__(self.fig)
        self.setParent(parent)
        from PyQt6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(200)


class MLTrainPage(QWidget):
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
        self.feature_dir_edit = QLineEdit("F:/Database/Feature")
        dir_layout.addWidget(self.feature_dir_edit, 1)
        browse_btn = QPushButton("浏览")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self.browse_data)
        dir_layout.addWidget(browse_btn)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        hint = QLabel("提示: 请先完成特征提取，确保目录下有 train_feature.csv 和 val_feature.csv")
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
        config_layout.setSpacing(0)

        config_layout.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(ML_MODELS.keys()))
        self.model_combo.setFixedWidth(140)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        config_layout.addWidget(self.model_combo)

        self.rf_params_widget = self._make_rf_params()
        self.svm_params_widget = self._make_svm_params()
        config_layout.addWidget(self.rf_params_widget, 1)
        config_layout.addWidget(self.svm_params_widget, 1)
        self.svm_params_widget.hide()

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

        tab_names = ["训练日志", "准确率", "召回率", "F1分数", "ROC曲线", "混淆矩阵"]
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

        self.plot_canvases = {}
        for name in ["准确率", "召回率", "F1分数", "ROC曲线", "混淆矩阵"]:
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

    def _make_rf_params(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        params = [
            ("n_est:", 100, 10, 1000, QSpinBox),
            ("max_depth:", 90, 2, 100, QSpinBox),
            ("min_split:", 80, 2, 20, QSpinBox),
            ("min_leaf:", 80, 1, 50, QSpinBox),
        ]
        self.n_estimators_spin = None
        self.max_depth_spin = None
        self.min_samples_split_spin = None
        self.min_samples_leaf_spin = None
        spin_attrs = ['n_estimators_spin', 'max_depth_spin', 'min_samples_split_spin', 'min_samples_leaf_spin']
        for i, (label_text, width, lo, hi, cls) in enumerate(params):
            h.addStretch(1)
            h.addWidget(QLabel(label_text))
            spin = cls()
            spin.setRange(lo, hi)
            spin.setFixedWidth(width)
            setattr(self, spin_attrs[i], spin)
            h.addWidget(spin)
        h.addStretch(1)
        h.addWidget(QLabel("max_feat:"))
        self.max_features_combo = QComboBox()
        self.max_features_combo.addItems(["sqrt", "log2", "None"])
        self.max_features_combo.setFixedWidth(85)
        h.addWidget(self.max_features_combo)
        h.addStretch(1)
        return w

    def _make_svm_params(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addStretch(1)
        h.addWidget(QLabel("kernel:"))
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(["rbf", "linear", "poly", "sigmoid"])
        self.kernel_combo.setFixedWidth(90)
        self.kernel_combo.currentTextChanged.connect(self.on_kernel_changed)
        h.addWidget(self.kernel_combo)
        h.addStretch(1)
        h.addWidget(QLabel("C:"))
        self.C_spin = QDoubleSpinBox()
        self.C_spin.setRange(0.01, 1000); self.C_spin.setValue(1.0)
        self.C_spin.setSingleStep(0.1); self.C_spin.setFixedWidth(100)
        h.addWidget(self.C_spin)
        h.addStretch(1)
        self.gamma_label = QLabel("gamma:")
        h.addWidget(self.gamma_label)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.0001, 100); self.gamma_spin.setValue(0.1)
        self.gamma_spin.setSingleStep(0.01); self.gamma_spin.setDecimals(4)
        self.gamma_spin.setFixedWidth(100)
        h.addWidget(self.gamma_spin)
        h.addStretch(1)
        self.degree_label = QLabel("degree:")
        h.addWidget(self.degree_label)
        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(1, 10); self.degree_spin.setValue(3)
        self.degree_spin.setFixedWidth(65)
        h.addWidget(self.degree_spin)
        h.addStretch(1)
        self.on_kernel_changed("rbf")
        return w

    def on_kernel_changed(self, kernel):
        show_gamma = kernel in ("rbf", "poly", "sigmoid")
        show_degree = kernel == "poly"
        self.gamma_label.setVisible(show_gamma)
        self.gamma_spin.setVisible(show_gamma)
        self.degree_label.setVisible(show_degree)
        self.degree_spin.setVisible(show_degree)

    def on_model_changed(self, text):
        self.rf_params_widget.setVisible(text == 'RandomForest')
        self.svm_params_widget.setVisible(text == 'SVM')

    def browse_data(self):
        d = QFileDialog.getExistingDirectory(self, "选择特征目录")
        if d:
            self.feature_dir_edit.setText(d)

    def start_training(self):
        feature_dir = self.feature_dir_edit.text()
        train_path = os.path.join(feature_dir, "train_feature.csv")
        val_path = os.path.join(feature_dir, "val_feature.csv")
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            self.log_text.append("错误: 未找到 train_feature.csv / val_feature.csv")
            return

        model_key = ML_MODELS[self.model_combo.currentText()]
        if model_key == 'random_forest':
            max_features = self.max_features_combo.currentText()
            if max_features == "None":
                max_features = None
            params = {
                'n_estimators': self.n_estimators_spin.value(),
                'max_depth': self.max_depth_spin.value(),
                'min_samples_split': self.min_samples_split_spin.value(),
                'min_samples_leaf': self.min_samples_leaf_spin.value(),
                'max_features': max_features,
                'criterion': 'gini'
            }
        else:
            params = {
                'C': self.C_spin.value(),
                'gamma': self.gamma_spin.value(),
                'kernel': self.kernel_combo.currentText(),
                'degree': self.degree_spin.value()
            }

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.log_text.clear()
        self.log_text.append(f"开始训练 {self.model_combo.currentText()}...")
        self.progress_bar.setValue(0)

        self.worker = MLTrainWorker(model_key, train_path, val_path, "F:/Database/Models", params)
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

    def on_finished(self, model_type, results, save_info):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.log_text.append(f"\n=== 训练完成 ===")
        self.log_text.append(f"模型: {model_type}")
        self.log_text.append(f"准确率: {results['accuracy']:.4f}")
        self.log_text.append(f"精确率: {results['precision']:.4f}")
        self.log_text.append(f"召回率: {results['recall']:.4f}")
        self.log_text.append(f"F1分数: {results['f1']:.4f}")
        self.log_text.append(f"ROC AUC: {results['roc_auc']:.4f}")
        self.log_text.append(f"模型已保存: {save_info.get('model_path', 'N/A')}")

        self._plot_accuracy(results)
        self._plot_recall(results)
        self._plot_f1(results)
        self._plot_roc(results)
        self._plot_confusion(results)

    def _plot_accuracy(self, results):
        canvas = self.plot_canvases["准确率"]
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        ax.bar(['Accuracy'], [results['accuracy']], color='#a6e3a1', width=0.4)
        ax.set_ylim(0, 1.1)
        ax.set_title('准确率 (Accuracy)', color='#cdd6f4', fontsize=14)
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#cdd6f4')
        ax.text(0, results['accuracy'] + 0.03, f'{results["accuracy"]:.4f}', ha='center', color='#cdd6f4', fontsize=12)
        for spine in ax.spines.values(): spine.set_color('#45475a')
        canvas.fig.tight_layout(rect=[0, 0, 1, 0.95])
        canvas.draw()

    def _plot_recall(self, results):
        canvas = self.plot_canvases["召回率"]
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        classes = results['classes']
        cm = results['confusion_matrix']
        recalls = []
        for i in range(len(classes)):
            total = cm[i, :].sum()
            recalls.append(cm[i, i] / total if total > 0 else 0)
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
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
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

    def _plot_roc(self, results):
        canvas = self.plot_canvases["ROC曲线"]
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        ax.plot(results['fpr'], results['tpr'], color='#89b4fa', linewidth=2, label=f'AUC = {results["roc_auc"]:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('FPR', color='#cdd6f4', fontsize=12)
        ax.set_ylabel('TPR', color='#cdd6f4', fontsize=12)
        ax.set_title('ROC Curve', color='#cdd6f4', fontsize=14)
        ax.legend(fontsize=12, facecolor='#313244', edgecolor='#45475a', labelcolor='#cdd6f4')
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#cdd6f4')
        for spine in ax.spines.values(): spine.set_color('#45475a')
        canvas.fig.tight_layout(rect=[0, 0, 1, 0.95])
        canvas.draw()

    def _plot_confusion(self, results):
        canvas = self.plot_canvases["混淆矩阵"]
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        cm = results['confusion_matrix']
        classes = results['classes']
        if cm.ndim != 2 or cm.shape[0] != len(classes) or cm.shape[1] != len(classes):
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
