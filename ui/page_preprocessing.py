import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import wfdb
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QTextEdit, QGroupBox, QComboBox,
    QLineEdit, QCheckBox, QSplitter, QSpinBox, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
import pyqtgraph as pg
from utils.preprocessing import wavelet_denoise, split_dataset, save_split_data

CLASS_MAP = {'N': 'N', 'L': 'L', 'R': 'R', 'A': 'A', 'V': 'V',
             'a': 'A', 'e': 'A', 'j': 'R', 'S': 'N', 'E': 'V',
             'f': 'V', 'F': 'V', 'x': 'N', 'p': 'N', 'Q': 'N',
             'J': 'R', '/': 'N', 'n': 'N', 'B': 'V'}
TARGET_LABELS = ['N', 'A', 'L', 'R', 'V']


class PreprocessWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, data_dir, output_dir):
        super().__init__()
        self.data_dir = data_dir
        self.output_dir = output_dir

    def run(self):
        try:
            records = sorted([os.path.splitext(f)[0] for f in os.listdir(self.data_dir) if f.endswith('.hea')])
            if not records:
                self.error.emit("未找到 .hea 文件")
                return
            all_segs, all_labs = [], []
            total = len(records)
            for i, rec in enumerate(records):
                self.progress.emit(int((i / total) * 80), f"处理 {rec}...")
                try:
                    path = os.path.join(self.data_dir, rec)
                    record = wfdb.rdrecord(path)
                    ann = wfdb.rdann(path, 'atr')
                    lead = record.sig_name.index('MLII') if 'MLII' in record.sig_name else 0
                    sig = record.p_signal[:, lead]
                    rpk, labs = [], []
                    for pk, s in zip(ann.sample, ann.symbol):
                        m = CLASS_MAP.get(s)
                        if m in TARGET_LABELS:
                            rpk.append(int(pk))
                            labs.append(m)
                    if not rpk:
                        continue
                    sig_dn = wavelet_denoise(sig)
                    segs, sl = [], []
                    for pk, lb in zip(np.array(rpk), np.array(labs)):
                        s, e = pk - 150, pk + 150
                        if s >= 0 and e <= len(sig_dn):
                            segs.append(sig_dn[s:e])
                            sl.append(lb)
                    if segs:
                        all_segs.append(np.array(segs))
                        all_labs.append(np.array(sl))
                except Exception:
                    continue
            if not all_segs:
                self.error.emit("无有效数据")
                return
            self.progress.emit(85, "合并...")
            all_segs = np.vstack(all_segs)
            all_labs = np.concatenate(all_labs)
            self.progress.emit(90, "划分数据集...")
            splits = split_dataset(all_segs, all_labs)
            save_split_data(splits, self.output_dir)
            self.progress.emit(100, "完成!")
            self.finished.emit({
                'total_samples': len(all_labs),
                'train_count': len(splits['train'][1]),
                'val_count': len(splits['val'][1]),
                'test_count': len(splits['test'][1]),
                'label_dist': {l: int(np.sum(all_labs == l)) for l in TARGET_LABELS},
                'records_count': len(records),
            })
        except Exception as e:
            self.error.emit(str(e))


class ECGPlotWidget(pg.PlotWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setBackground('#0D1B2A')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setLabel('left', 'mV')
        self.setLabel('bottom', 'Sample')
        self.setTitle(title, color='#8899AA', size='10pt')
        self.setMinimumHeight(230)
        self.getAxis('bottom').setHeight(30)
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='#555', style=Qt.PenStyle.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(color='#555', style=Qt.PenStyle.DashLine))
        self.addItem(self.vLine, ignoreBounds=True)
        self.addItem(self.hLine, ignoreBounds=True)
        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)
        self._coord_label = None
        self.signal_curve = None
        self.rpeak_scatter = None

    def set_coord_label(self, lbl):
        self._coord_label = lbl

    def _mouse_moved(self, evt):
        pos = evt[0]
        if self.sceneBoundingRect().contains(pos):
            mp = self.getViewBox().mapSceneToView(pos)
            self.vLine.setPos(mp.x())
            self.hLine.setPos(mp.y())
            if self._coord_label:
                self._coord_label.setText(f"X: {mp.x():.2f}  Y: {mp.y():.4f}")

    def plot_signal(self, signal, color='#00FF88'):
        if self.signal_curve is None:
            self.signal_curve = self.plot(pen=pg.mkPen(color=color, width=1.5))
        self.signal_curve.setData(np.arange(len(signal)), signal)

    def plot_rpeaks(self, r_peaks, signal):
        if self.rpeak_scatter is not None:
            self.removeItem(self.rpeak_scatter)
            self.rpeak_scatter = None
        if len(r_peaks) > 0:
            v = r_peaks[(r_peaks >= 0) & (r_peaks < len(signal))]
            self.rpeak_scatter = pg.ScatterPlotItem(x=v, y=signal[v], brush=pg.mkBrush('#FF4444'), size=5, pen=pg.mkPen(None))
            self.addItem(self.rpeak_scatter)

    def clear_annotations(self):
        if self.rpeak_scatter is not None:
            self.removeItem(self.rpeak_scatter)
            self.rpeak_scatter = None


class PreprocessingPage(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.data_dir = ""
        self.current_signal = None
        self.current_signal_dn = None
        self.current_fs = 360
        self.current_r_peaks = None
        self.setup_ui()

    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(2)
        root.setContentsMargins(6, 2, 6, 2)

        title = QLabel("数据预处理")
        title.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #89b4fa; padding: 1px 0px;")
        root.addWidget(title)

        top_grid = QGridLayout()
        top_grid.setSpacing(6)
        top_grid.setColumnStretch(0, 1)
        top_grid.setColumnStretch(1, 1)

        g1 = self._grp("数据目录 (输入)")
        g1l = QVBoxLayout()
        g1l.setContentsMargins(10, 6, 8, 6)
        g1l.setSpacing(3)
        r = QHBoxLayout()
        r.setSpacing(4)
        self.dir_edit = QLineEdit()
        self.dir_edit.setText("F:/Database/ECG")
        self.dir_edit.setPlaceholderText("MIT-BIH 数据目录")
        r.addWidget(self.dir_edit)
        for txt, fn, w in [("浏览", self.browse_dir, 70), ("刷新", self.refresh_records, 70)]:
            b = QPushButton(txt)
            b.setFixedWidth(w)
            b.clicked.connect(fn)
            r.addWidget(b)
        g1l.addLayout(r)
        g1l.addWidget(self._hint("选择包含 .dat / .hea / .atr 的目录"))
        g1.setLayout(g1l)
        top_grid.addWidget(g1, 0, 0)

        g3 = self._grp("数据目录 (输出)")
        g3l = QVBoxLayout()
        g3l.setContentsMargins(10, 6, 8, 6)
        g3l.setSpacing(3)
        r4 = QHBoxLayout()
        r4.setSpacing(4)
        self.output_edit = QLineEdit("F:/Database/Processed")
        r4.addWidget(self.output_edit, 1)
        b = QPushButton("浏览")
        b.setFixedWidth(70)
        b.clicked.connect(self.browse_output)
        r4.addWidget(b)
        g3l.addLayout(r4)
        g3l.addWidget(self._hint("train.csv / val.csv / test.csv 保存位置"))
        g3.setLayout(g3l)
        top_grid.addWidget(g3, 0, 1)

        g2 = self._grp("记录与通道")
        g2l = QVBoxLayout()
        g2l.setContentsMargins(10, 6, 8, 6)
        g2l.setSpacing(3)
        r2 = QHBoxLayout()
        r2.setSpacing(0)
        r2.addWidget(QLabel("记录:"))
        r2.addSpacing(4)
        self.record_combo = QComboBox()
        self.record_combo.setFixedWidth(95)
        self.record_combo.currentTextChanged.connect(self.on_record_changed)
        r2.addWidget(self.record_combo)
        r2.addStretch(1)
        r2.addWidget(QLabel("通道:"))
        r2.addSpacing(4)
        self.channel_combo = QComboBox()
        self.channel_combo.setFixedWidth(95)
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)
        r2.addWidget(self.channel_combo)
        r2.addStretch(1)
        self.show_rpeaks_cb = QCheckBox("显示R峰标记")
        self.show_rpeaks_cb.setChecked(True)
        self.show_rpeaks_cb.toggled.connect(self.toggle_rpeaks)
        r2.addWidget(self.show_rpeaks_cb)
        r2.addStretch(1)
        self.random_beat_cb = QCheckBox("随机心拍去噪展示")
        self.random_beat_cb.setChecked(False)
        self.random_beat_cb.toggled.connect(self.on_random_beat_toggled)
        r2.addWidget(self.random_beat_cb)
        self.refresh_beat_btn = QPushButton("换一条")
        self.refresh_beat_btn.setFixedWidth(80)
        self.refresh_beat_btn.setEnabled(False)
        self.refresh_beat_btn.clicked.connect(self.show_random_beat)
        r2.addWidget(self.refresh_beat_btn)
        g2l.addLayout(r2)
        g2.setLayout(g2l)
        top_grid.addWidget(g2, 1, 0)

        g4 = self._grp("预处理参数")
        g4l = QVBoxLayout()
        g4l.setContentsMargins(10, 6, 8, 6)
        g4l.setSpacing(3)
        p1 = QHBoxLayout()
        p1.setSpacing(0)
        p1.addWidget(QLabel("分段方式:"))
        p1.addSpacing(6)
        p1.addWidget(QLabel("R峰中心"))
        p1.addStretch(1)
        p1.addWidget(QLabel("前:"))
        p1.addSpacing(4)
        self.before_spin = QSpinBox()
        self.before_spin.setRange(50, 300)
        self.before_spin.setValue(150)
        self.before_spin.setFixedWidth(100)
        p1.addWidget(self.before_spin)
        p1.addStretch(1)
        p1.addWidget(QLabel("后:"))
        p1.addSpacing(4)
        self.after_spin = QSpinBox()
        self.after_spin.setRange(50, 300)
        self.after_spin.setValue(150)
        self.after_spin.setFixedWidth(100)
        p1.addWidget(self.after_spin)
        p1.addStretch(1)
        p1.addWidget(QLabel("总=300"))
        p1.addStretch(2)
        self.start_btn = QPushButton("开始预处理")
        self.start_btn.setMinimumHeight(26)
        self.start_btn.clicked.connect(self.start_preprocessing)
        p1.addWidget(self.start_btn)
        p1.addSpacing(8)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setMinimumHeight(26)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_preprocessing)
        p1.addWidget(self.stop_btn)
        g4l.addLayout(p1)
        g4l.addSpacing(4)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(10)
        g4l.addWidget(self.progress_bar)
        g4.setLayout(g4l)
        top_grid.addWidget(g4, 1, 1)

        root.addLayout(top_grid)

        self.coord_label = QLabel("")
        self.coord_label.setStyleSheet("color: #6c7086; font-size: 10px; font-family: Consolas; padding: 0;")
        root.addWidget(self.coord_label)

        mid_grid = QGridLayout()
        mid_grid.setSpacing(4)
        mid_grid.setColumnStretch(0, 1)
        mid_grid.setColumnStretch(1, 1)
        mid_grid.setContentsMargins(0, 0, 0, 5)

        self.plot_raw = ECGPlotWidget("原始信号 (Raw)")
        self.plot_raw.set_coord_label(self.coord_label)
        self.plot_denoised = ECGPlotWidget("小波去噪 (db4)")
        self.plot_denoised.set_coord_label(self.coord_label)
        mid_grid.addWidget(self.plot_raw, 0, 0)
        mid_grid.addWidget(self.plot_denoised, 0, 1)

        lg = self._grp("处理日志")
        lgl = QVBoxLayout()
        lgl.setContentsMargins(8, 6, 8, 6)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(80)
        lgl.addWidget(self.log_text)
        lg.setLayout(lgl)
        mid_grid.addWidget(lg, 1, 0)

        sg = self._grp("数据统计")
        sgl = QHBoxLayout()
        sgl.setContentsMargins(10, 6, 8, 6)
        sgl.setSpacing(12)
        self.stats_labels = {}
        for name in ['总样本数', '训练集', '验证集', '测试集', '记录数']:
            lbl = QLabel(f"{name}: -")
            lbl.setStyleSheet("font-size: 11px; color: #a6e3a1;")
            sgl.addWidget(lbl)
            self.stats_labels[name] = lbl
        sg.setLayout(sgl)
        mid_grid.addWidget(sg, 1, 1)

        root.addLayout(mid_grid, 1)

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

    def browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择MIT-BIH目录")
        if d:
            self.dir_edit.setText(d)
            self.refresh_records()

    def browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            self.output_edit.setText(d)

    def refresh_records(self):
        self.data_dir = self.dir_edit.text()
        if not self.data_dir or not os.path.isdir(self.data_dir):
            return
        records = sorted([os.path.splitext(f)[0] for f in os.listdir(self.data_dir) if f.endswith('.hea')])
        self.record_combo.blockSignals(True)
        self.record_combo.clear()
        self.record_combo.addItems(records)
        self.record_combo.blockSignals(False)
        if records:
            self.load_record(records[0])

    def on_record_changed(self, name):
        if name:
            self.load_record(name)

    def load_record(self, record_name):
        if not self.data_dir:
            return
        try:
            record = wfdb.rdrecord(os.path.join(self.data_dir, record_name))
            self.channel_combo.blockSignals(True)
            self.channel_combo.clear()
            self.channel_combo.addItems(record.sig_name)
            self.channel_combo.blockSignals(False)
            lead = record.sig_name.index('MLII') if 'MLII' in record.sig_name else 0
            self.channel_combo.setCurrentIndex(lead)
            self.current_fs = record.fs
            self._load_channel(record, lead, record_name)
        except Exception as e:
            self.log_text.append(f"加载失败: {e}")

    def on_channel_changed(self, idx):
        if idx < 0 or not self.data_dir:
            return
        try:
            record = wfdb.rdrecord(os.path.join(self.data_dir, self.record_combo.currentText()))
            self._load_channel(record, idx, self.record_combo.currentText())
        except Exception:
            pass

    def _load_channel(self, record, ch_idx, record_name):
        self.current_signal = record.p_signal[:, ch_idx]
        self.current_signal_dn = wavelet_denoise(self.current_signal)
        try:
            ann = wfdb.rdann(os.path.join(self.data_dir, record_name), 'atr')
            rpk = [int(pk) for pk, s in zip(ann.sample, ann.symbol) if CLASS_MAP.get(s) in TARGET_LABELS]
            self.current_r_peaks = np.array(rpk)
        except Exception:
            self.current_r_peaks = np.array([], dtype=int)
        self.update_plots()

    def update_plots(self):
        if self.current_signal is None:
            return
        if self.random_beat_cb.isChecked():
            self.show_random_beat()
            return
        self.plot_raw.plot_signal(self.current_signal)
        self.plot_raw.setTitle("原始信号 (Raw)")
        self.plot_denoised.plot_signal(self.current_signal_dn)
        self.plot_denoised.setTitle("小波去噪 (db4)")
        if self.show_rpeaks_cb.isChecked() and len(self.current_r_peaks) > 0:
            self.plot_raw.plot_rpeaks(self.current_r_peaks, self.current_signal)
            self.plot_denoised.plot_rpeaks(self.current_r_peaks, self.current_signal_dn)

    def toggle_rpeaks(self, checked):
        if not self.random_beat_cb.isChecked():
            if checked:
                self.update_plots()
            else:
                self.plot_raw.clear_annotations()
                self.plot_denoised.clear_annotations()

    def on_random_beat_toggled(self, checked):
        self.refresh_beat_btn.setEnabled(checked)
        if checked:
            self.show_random_beat()
        else:
            self.update_plots()

    def show_random_beat(self):
        if self.current_signal is None or len(self.current_r_peaks) == 0:
            return
        before = self.before_spin.value()
        after = self.after_spin.value()
        idx = np.random.choice(self.current_r_peaks)
        s = max(0, idx - before)
        e = min(len(self.current_signal), idx + after)
        raw_beat = self.current_signal[s:e]
        dn_beat = self.current_signal_dn[s:e]
        self.plot_raw.setTitle(f"原始心拍 (R-peak={idx})")
        self.plot_raw.plot_signal(raw_beat)
        self.plot_raw.clear_annotations()
        self.plot_denoised.setTitle(f"去噪心拍 (R-peak={idx})")
        self.plot_denoised.plot_signal(dn_beat)
        self.plot_denoised.clear_annotations()

    def start_preprocessing(self):
        data_dir = self.dir_edit.text()
        output_dir = self.output_edit.text()
        if not data_dir or not os.path.isdir(data_dir):
            self.log_text.append("错误: 请选择有效目录")
            return
        os.makedirs(output_dir, exist_ok=True)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_text.clear()
        self.log_text.append("开始预处理...")
        self.progress_bar.setValue(0)
        self.worker = PreprocessWorker(data_dir, output_dir)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def stop_preprocessing(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.log_text.append("已停止")

    def on_progress(self, v, msg):
        self.progress_bar.setValue(v)
        self.log_text.append(msg)

    def on_finished(self, stats):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append(f"=== 完成 === 记录:{stats['records_count']} 样本:{stats['total_samples']}")
        self.log_text.append(f"训练:{stats['train_count']} 验证:{stats['val_count']} 测试:{stats['test_count']}")
        for l, c in stats['label_dist'].items():
            self.log_text.append(f"  {l}: {c}")
        self.stats_labels['总样本数'].setText(f"总样本数: {stats['total_samples']}")
        self.stats_labels['训练集'].setText(f"训练集: {stats['train_count']}")
        self.stats_labels['验证集'].setText(f"验证集: {stats['val_count']}")
        self.stats_labels['测试集'].setText(f"测试集: {stats['test_count']}")
        self.stats_labels['记录数'].setText(f"记录数: {stats['records_count']}")

    def on_error(self, msg):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append(f"错误: {msg}")
