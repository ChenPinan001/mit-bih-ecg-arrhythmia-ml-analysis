import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QGroupBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QLineEdit, QGridLayout,
    QCheckBox, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
import pyqtgraph as pg
import tsfel


class FeatureWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, data_dir, split_name, output_dir, do_filter=True, corr_threshold=0.95,
                 do_pca=False, pca_components=50, shared_objects=None):
        super().__init__()
        self.data_dir = data_dir
        self.split_name = split_name
        self.output_dir = output_dir
        self.do_filter = do_filter
        self.corr_threshold = corr_threshold
        self.do_pca = do_pca
        self.pca_components = pca_components
        self.shared_objects = shared_objects

    def run(self):
        try:
            csv_path = os.path.join(self.data_dir, f"{self.split_name}.csv")
            if not os.path.exists(csv_path):
                self.error.emit(f"未找到文件: {csv_path}")
                return

            self.progress.emit(5, "加载数据...")
            df = pd.read_csv(csv_path)
            labels = df['label'].values
            sample_ids = df['sample_id'].values
            segments = df.drop(columns=['label', 'sample_id']).values

            self.progress.emit(10, "使用 TSFEL 提取特征...")
            cfg = tsfel.get_features_by_domain()
            self.fs = 360

            all_features = []
            total = len(segments)
            batch_size = 200

            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch = segments[start:end]
                features_list = []
                for seg in batch:
                    feat = tsfel.time_series_features_extractor(cfg, seg, fs=self.fs, verbose=0)
                    features_list.append(feat.values.flatten())
                all_features.extend(features_list)
                pct = int(10 + (end / total) * 80)
                self.progress.emit(pct, f"已提取 {end}/{total} 样本")

            self.progress.emit(92, "构建特征DataFrame...")
            sample_feat = tsfel.time_series_features_extractor(cfg, segments[0], fs=self.fs, verbose=0)
            feature_names = list(sample_feat.columns)

            feature_df = pd.DataFrame(all_features, columns=feature_names)
            feature_df['label'] = labels
            feature_df['sample_id'] = sample_ids
            cols = ['sample_id'] + [c for c in feature_df.columns if c not in ['sample_id', 'label']] + ['label']
            feature_df = feature_df[cols]

            original_count = len(feature_names)

            if self.do_filter:
                is_train = self.split_name == 'train'
                if is_train or not (self.shared_objects and 'keep_cols' in self.shared_objects):
                    self.progress.emit(93, "过滤零方差特征...")
                    numeric_cols = [c for c in feature_df.columns if c not in ['sample_id', 'label']]
                    variances = feature_df[numeric_cols].var()
                    zero_var_cols = variances[variances == 0].index.tolist()
                    if zero_var_cols:
                        feature_df = feature_df.drop(columns=zero_var_cols)
                        self.progress.emit(94, f"去掉 {len(zero_var_cols)} 个零方差特征")

                    self.progress.emit(95, "过滤高相关特征...")
                    numeric_cols = [c for c in feature_df.columns if c not in ['sample_id', 'label']]
                    if len(numeric_cols) > 1:
                        corr_matrix = feature_df[numeric_cols].corr().abs()
                        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                        to_drop = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
                        if to_drop:
                            feature_df = feature_df.drop(columns=to_drop)
                            self.progress.emit(97, f"去掉 {len(to_drop)} 个高相关特征 (阈值>{self.corr_threshold})")

                    keep_cols = [c for c in feature_df.columns if c not in ['sample_id', 'label']]
                    if self.shared_objects is not None:
                        self.shared_objects['keep_cols'] = keep_cols
                else:
                    self.progress.emit(93, "应用训练集过滤规则...")
                    keep_cols = self.shared_objects['keep_cols']
                    available = [c for c in keep_cols if c in feature_df.columns]
                    missing = [c for c in keep_cols if c not in feature_df.columns]
                    if missing:
                        for mc in missing:
                            feature_df[mc] = 0.0
                    feature_df = feature_df[['sample_id'] + keep_cols + ['label']]

            feature_names_final = [c for c in feature_df.columns if c not in ['sample_id', 'label']]
            filtered_count = len(feature_names_final)

            pca_n = None
            if self.do_pca and len(feature_names_final) > self.pca_components:
                self.progress.emit(98, f"PCA降维: {filtered_count} → {self.pca_components} 维...")
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA

                X_pca = feature_df[feature_names_final].values
                X_pca = np.nan_to_num(X_pca, nan=0.0)

                if self.shared_objects and 'scaler' in self.shared_objects:
                    scaler = self.shared_objects['scaler']
                    pca = self.shared_objects['pca']
                    X_pca = scaler.transform(X_pca)
                    X_pca = pca.transform(X_pca)
                else:
                    scaler = StandardScaler()
                    X_pca = scaler.fit_transform(X_pca)
                    n_comp = min(self.pca_components, X_pca.shape[1], X_pca.shape[0])
                    pca = PCA(n_components=n_comp)
                    X_pca = pca.fit_transform(X_pca)
                    if self.shared_objects is not None:
                        self.shared_objects['scaler'] = scaler
                        self.shared_objects['pca'] = pca

                pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
                feature_df = pd.DataFrame(X_pca, columns=pca_cols)
                feature_df['label'] = labels
                feature_df['sample_id'] = sample_ids
                feature_names_final = pca_cols
                pca_n = X_pca.shape[1]
                self.progress.emit(99, f"PCA完成: 保留 {pca_n} 个主成分 (解释方差: {sum(self.shared_objects['pca'].explained_variance_ratio_)*100:.1f}%)")

            output_path = os.path.join(self.output_dir, f"{self.split_name}_feature.csv")
            os.makedirs(self.output_dir, exist_ok=True)
            feature_df.to_csv(output_path, index=False)

            self.progress.emit(100, "特征提取完成!")

            stats = {
                'feature_count': original_count,
                'filtered_count': filtered_count,
                'removed_count': original_count - filtered_count,
                'pca_count': pca_n,
                'sample_count': len(labels),
                'feature_names': feature_names_final[:20],
                'output_path': output_path,
                'label_dist': {l: int(np.sum(labels == l)) for l in ['N', 'A', 'L', 'R', 'V']},
                'sample_features': all_features[:5] if all_features else [],
            }
            self.finished.emit(stats)
        except Exception as e:
            self.error.emit(str(e))


class FeaturePage(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.setContentsMargins(8, 4, 8, 4)

        title = QLabel("特征提取")
        title.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #89b4fa; padding: 2px 0px;")
        root.addWidget(title)

        top_grid = QGridLayout()
        top_grid.setSpacing(6)
        top_grid.setColumnStretch(0, 1)
        top_grid.setColumnStretch(1, 1)

        g1 = self._grp("数据目录 (输入)")
        g1l = QVBoxLayout()
        g1l.setContentsMargins(10, 6, 8, 6)
        g1l.setSpacing(3)
        r1 = QHBoxLayout()
        r1.setSpacing(4)
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setText("F:/Database/Processed")
        self.data_dir_edit.setPlaceholderText("选择预处理后的目录 (含 train.csv / val.csv / test.csv)")
        r1.addWidget(self.data_dir_edit, 1)
        b1 = QPushButton("浏览")
        b1.setFixedWidth(70)
        b1.clicked.connect(self.browse_input)
        r1.addWidget(b1)
        g1l.addLayout(r1)
        g1l.addWidget(self._hint("选择上一步预处理输出的含 train.csv / val.csv / test.csv 的目录"))
        g1.setLayout(g1l)
        top_grid.addWidget(g1, 0, 0)

        g3 = self._grp("数据目录 (输出)")
        g3l = QVBoxLayout()
        g3l.setContentsMargins(10, 6, 8, 6)
        g3l.setSpacing(3)
        r3 = QHBoxLayout()
        r3.setSpacing(4)
        self.output_dir_edit = QLineEdit("F:/Database/Feature")
        r3.addWidget(self.output_dir_edit, 1)
        b3 = QPushButton("浏览")
        b3.setFixedWidth(70)
        b3.clicked.connect(self.browse_output)
        r3.addWidget(b3)
        g3l.addLayout(r3)
        g3l.addWidget(self._hint("特征文件 train_feature.csv / val_feature.csv / test_feature.csv 保存位置"))
        g3.setLayout(g3l)
        top_grid.addWidget(g3, 0, 1)

        g2 = self._grp("提取配置")
        g2l = QVBoxLayout()
        g2l.setContentsMargins(10, 6, 8, 6)
        g2l.setSpacing(3)
        r2 = QHBoxLayout()
        r2.setSpacing(16)
        r2.setContentsMargins(8, 0, 8, 0)

        r2.addWidget(QLabel("采样频率:"))
        self.fs_spin = QSpinBox()
        self.fs_spin.setRange(100, 1000)
        self.fs_spin.setValue(360)
        self.fs_spin.setSuffix(" Hz")
        self.fs_spin.setFixedWidth(110)
        r2.addWidget(self.fs_spin)

        r2.addWidget(QLabel("窗宽:"))
        self.win_spin = QSpinBox()
        self.win_spin.setRange(50, 1000)
        self.win_spin.setValue(300)
        self.win_spin.setFixedWidth(90)
        r2.addWidget(self.win_spin)

        r2.addWidget(QLabel("数据集:"))
        cb_ss = ("QCheckBox { font-size: 14px; spacing: 6px; }"
                 "QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px;"
                 "border: 1px solid #45475a; background-color: #313244; }"
                 "QCheckBox::indicator:checked { background-color: #a6e3a1; border-color: #a6e3a1; }")
        self.train_cb = QCheckBox("训练集")
        self.train_cb.setChecked(True)
        self.train_cb.setStyleSheet(cb_ss)
        r2.addWidget(self.train_cb)
        self.val_cb = QCheckBox("验证集")
        self.val_cb.setChecked(True)
        self.val_cb.setStyleSheet(cb_ss)
        r2.addWidget(self.val_cb)
        self.test_cb = QCheckBox("测试集")
        self.test_cb.setChecked(True)
        self.test_cb.setStyleSheet(cb_ss)
        r2.addWidget(self.test_cb)

        r2.addStretch(1)
        self.filter_cb = QCheckBox("特征过滤")
        self.filter_cb.setChecked(True)
        self.filter_cb.setStyleSheet(cb_ss)
        r2.addWidget(self.filter_cb)
        r2.addWidget(QLabel("相关阈值:"))
        self.corr_spin = QDoubleSpinBox()
        self.corr_spin.setRange(0.5, 0.99)
        self.corr_spin.setValue(0.95)
        self.corr_spin.setSingleStep(0.01)
        self.corr_spin.setDecimals(2)
        self.corr_spin.setFixedWidth(85)
        r2.addWidget(self.corr_spin)

        r2.addWidget(QLabel("PCA:"))
        self.pca_cb = QCheckBox("降维")
        self.pca_cb.setChecked(True)
        self.pca_cb.setStyleSheet(cb_ss)
        r2.addWidget(self.pca_cb)
        r2.addWidget(QLabel("维度:"))
        self.pca_dim_spin = QSpinBox()
        self.pca_dim_spin.setRange(5, 200)
        self.pca_dim_spin.setValue(50)
        self.pca_dim_spin.setFixedWidth(80)
        r2.addWidget(self.pca_dim_spin)

        r2.addStretch(1)
        self.extract_btn = QPushButton("开始特征提取")
        self.extract_btn.setMinimumHeight(34)
        self.extract_btn.setStyleSheet("font-size: 14px; padding: 6px 16px;")
        self.extract_btn.clicked.connect(self.start_extraction)
        r2.addWidget(self.extract_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setMinimumHeight(34)
        self.stop_btn.setStyleSheet("font-size: 14px; padding: 6px 16px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_extraction)
        r2.addWidget(self.stop_btn)
        g2l.addLayout(r2)
        g2l.addSpacing(6)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(10)
        g2l.addWidget(self.progress_bar)
        g2.setLayout(g2l)
        top_grid.addWidget(g2, 1, 0, 1, 2)

        root.addLayout(top_grid)

        mid_grid = QHBoxLayout()
        mid_grid.setSpacing(4)

        left_col = self._grp("特征信息")
        left_lay = QVBoxLayout()
        left_lay.setContentsMargins(8, 6, 8, 6)
        self.feature_table = QTableWidget()
        self.feature_table.setColumnCount(3)
        self.feature_table.setHorizontalHeaderLabels(['特征名称', '类型', '说明'])
        self.feature_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.feature_table.verticalHeader().setVisible(False)
        self.feature_table.setStyleSheet("""
            QTableWidget { background-color: #181825; color: #cdd6f4; border: 1px solid #45475a;
                           gridline-color: #313244; }
            QTableWidget::item:selected { background-color: #45475a; }
            QHeaderView::section { background-color: #313244; color: #89b4fa;
                                   border: 1px solid #45475a; padding: 5px; font-weight: bold; }
        """)
        left_lay.addWidget(self.feature_table)
        left_col.setLayout(left_lay)
        mid_grid.addWidget(left_col, 1)

        mid_col = self._grp("特征分布预览")
        mid_lay = QVBoxLayout()
        mid_lay.setContentsMargins(8, 6, 8, 6)
        self.feature_plot = pg.PlotWidget()
        self.feature_plot.setBackground('#0D1B2A')
        self.feature_plot.showGrid(x=True, y=True, alpha=0.3)
        self.feature_plot.setLabel('left', 'Count')
        self.feature_plot.setLabel('bottom', 'Feature Value')
        mid_lay.addWidget(self.feature_plot)
        mid_col.setLayout(mid_lay)
        mid_grid.addWidget(mid_col, 2)

        right_col = self._grp("提取日志")
        right_lay = QVBoxLayout()
        right_lay.setContentsMargins(8, 6, 8, 6)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_lay.addWidget(self.log_text)
        right_col.setLayout(right_lay)
        mid_grid.addWidget(right_col, 1)

        root.addLayout(mid_grid, 1)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        root.addWidget(self.status_label)

    def _grp(self, title):
        g = QGroupBox(title)
        g.setStyleSheet("""
            QGroupBox { color: #89b4fa; border: 1px solid #45475a; border-radius: 4px;
                         margin-top: 14px; font-weight: bold; font-size: 12px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
                               left: 12px; top: 1px; padding: 0 4px; }
        """)
        return g

    def _hint(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #fab387; font-size: 10px;")
        return lbl

    def _checkbox_ss(self):
        return """
            QCheckBox { spacing: 4px; font-size: 12px; }
            QCheckBox::indicator { width: 14px; height: 14px; border-radius: 3px;
                border: 1px solid #45475a; background-color: #313244; }
            QCheckBox::indicator:checked { background-color: #a6e3a1; border-color: #a6e3a1; }
        """

    def browse_input(self):
        d = QFileDialog.getExistingDirectory(self, "选择预处理数据目录")
        if d:
            self.data_dir_edit.setText(d)

    def browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            self.output_dir_edit.setText(d)

    def start_extraction(self):
        data_dir = self.data_dir_edit.text()
        output_dir = self.output_dir_edit.text()

        if not data_dir or not os.path.isdir(data_dir):
            self.log_text.append("错误: 请选择有效的数据目录")
            return

        splits = []
        if self.train_cb.isChecked():
            splits.append("train")
        if self.val_cb.isChecked():
            splits.append("val")
        if self.test_cb.isChecked():
            splits.append("test")

        if not splits:
            self.log_text.append("错误: 请至少选择一个数据集")
            return

        for s in splits:
            if not os.path.exists(os.path.join(data_dir, f"{s}.csv")):
                self.log_text.append(f"错误: 目录中未找到 {s}.csv")
                return

        self.extract_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_text.clear()
        self.log_text.append(f"开始特征提取: {', '.join(splits)}")
        self.log_text.append(f"采样频率: {self.fs_spin.value()} Hz  窗宽: {self.win_spin.value()}")
        self.progress_bar.setValue(0)

        self._pending_splits = splits
        self._pending_idx = 0
        self._pending_output = output_dir
        self._shared_objects = {}
        self._start_next_split()

    def stop_extraction(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
        self.extract_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append("已停止")

    def _start_next_split(self):
        if self._pending_idx >= len(self._pending_splits):
            self.extract_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.log_text.append("\n=== 全部提取完成 ===")
            self.status_label.setText("全部提取完成")
            return
        split_name = self._pending_splits[self._pending_idx]
        self.log_text.append(f"\n--- 提取 {split_name} ---")
        data_dir = self.data_dir_edit.text()
        do_filter = self.filter_cb.isChecked()
        corr_threshold = self.corr_spin.value()
        do_pca = self.pca_cb.isChecked()
        pca_dim = self.pca_dim_spin.value()
        self.worker = FeatureWorker(data_dir, split_name, self._pending_output,
                                     do_filter, corr_threshold, do_pca, pca_dim,
                                     self._shared_objects)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        self.log_text.append(message)

    def on_finished(self, stats):
        self.log_text.append(f"原始特征数: {stats['feature_count']}")
        if 'filtered_count' in stats:
            self.log_text.append(f"过滤后特征数: {stats['filtered_count']}")
            self.log_text.append(f"去掉特征数: {stats['removed_count']}")
        if stats.get('pca_count'):
            self.log_text.append(f"PCA降维后: {stats['pca_count']} 维")
        self.log_text.append(f"样本数量: {stats['sample_count']}")
        self.log_text.append(f"输出路径: {stats['output_path']}")
        for label, count in stats['label_dist'].items():
            self.log_text.append(f"  {label}: {count}")

        self.feature_table.setRowCount(len(stats['feature_names']))
        for i, name in enumerate(stats['feature_names']):
            self.feature_table.setItem(i, 0, QTableWidgetItem(name))
            self.feature_table.setItem(i, 1, QTableWidgetItem("时域/频域"))
            self.feature_table.setItem(i, 2, QTableWidgetItem("-"))

        if stats['sample_features']:
            self.plot_feature_distribution(stats['sample_features'])

        self._pending_idx += 1
        self._start_next_split()

    def on_error(self, msg):
        self.extract_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append(f"错误: {msg}")

    def plot_feature_distribution(self, features):
        self.feature_plot.clear()
        features_arr = np.array(features, dtype=float)
        if features_arr.ndim != 2 or features_arr.shape[1] == 0:
            return
        for col_idx in range(min(features_arr.shape[1], 10)):
            values = features_arr[:, col_idx]
            values = values[np.isfinite(values)]
            if len(values) > 2 and np.std(values) > 1e-10:
                break
        else:
            return
        hist, bins = np.histogram(values, bins=20)
        x = (bins[:-1] + bins[1:]) / 2
        width = (bins[1] - bins[0]) * 0.8
        if width <= 0:
            width = 1
        bar = pg.BarGraphItem(x=x, height=hist, width=width,
                              brush=pg.mkBrush(137, 180, 250, 150),
                              pen=pg.mkPen(color='#89b4fa'))
        self.feature_plot.addItem(bar)
        self.feature_plot.autoRange()
