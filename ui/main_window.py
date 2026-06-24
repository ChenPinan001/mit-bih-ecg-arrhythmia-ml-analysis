import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QLabel, QApplication, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from ui.page_preprocessing import PreprocessingPage
from ui.page_feature import FeaturePage
from ui.page_ml_train import MLTrainPage
from ui.page_dl_train import DLTrainPage
from ui.page_model_test import ModelTestPage
from ui.page_realtime import RealtimePage
from ui.page_model_manager import ModelManagerPage

DARK_STYLESHEET = """
QMainWindow {
    background-color: #1e1e2e;
}
QTabWidget::pane {
    border: 1px solid #45475a;
    background-color: #1e1e2e;
}
QTabBar::tab {
    background-color: #313244;
    color: #cdd6f4;
    padding: 10px 0px;
    border: 1px solid #45475a;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-size: 14px;
    font-weight: bold;
    min-width: 0px;
}
QTabBar {
    qproperty-drawBase: 0;
}
QTabBar::tab:selected {
    background-color: #45475a;
    color: #89b4fa;
    border-bottom: 2px solid #89b4fa;
}
QTabBar::tab:hover:!selected {
    background-color: #585b70;
}
QTabBar::tab:selected {
    background-color: #45475a;
    color: #89b4fa;
    border-bottom: 2px solid #89b4fa;
}
QTabBar::tab:hover:!selected {
    background-color: #585b70;
}
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: bold;
    font-size: 14px;
}
QPushButton:hover {
    background-color: #74c7ec;
}
QPushButton:pressed {
    background-color: #89dceb;
}
QPushButton:disabled {
    background-color: #45475a;
    color: #6c7086;
}
QLabel {
    color: #cdd6f4;
    font-size: 14px;
}
QTextEdit, QPlainTextEdit {
    background-color: #181825;
    color: #a6e3a1;
    border: 1px solid #45475a;
    border-radius: 4px;
    font-family: Consolas, monospace;
    font-size: 13px;
}
QProgressBar {
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
    color: #cdd6f4;
    background-color: #313244;
}
QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 3px;
}
QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 5px;
    font-size: 14px;
}
QComboBox::drop-down {
    border: none;
}
QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    selection-background-color: #89b4fa;
}
QLineEdit {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 5px;
    font-size: 14px;
}
QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 5px 20px 5px 6px;
    font-size: 14px;
}
QGroupBox {
    color: #89b4fa;
    border: 1px solid #45475a;
    border-radius: 4px;
    margin-top: 18px;
    padding: 8px 8px 8px 12px;
    font-weight: bold;
    font-size: 14px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 2px;
    padding: 0 4px;
}
QScrollBar:vertical {
    background-color: #1e1e2e;
    width: 10px;
    border: none;
}
QScrollBar::handle:vertical {
    background-color: #45475a;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QCheckBox {
    color: #cdd6f4;
    font-size: 14px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #45475a;
    background-color: #313244;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
}
QTableWidget {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #45475a;
    gridline-color: #313244;
    font-size: 13px;
}
QTableWidget::item:selected {
    background-color: #45475a;
}
QHeaderView::section {
    background-color: #313244;
    color: #89b4fa;
    border: 1px solid #45475a;
    padding: 5px;
    font-weight: bold;
    font-size: 13px;
}
QSplitter::handle {
    background-color: #45475a;
    width: 2px;
    height: 2px;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于机器学习与深度学习的心律失常分析系统")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)

        header = QLabel("基于机器学习与深度学习的心律失常分析系统")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        header.setStyleSheet("color: #89b4fa; padding: 10px; margin-bottom: 5px;")
        layout.addWidget(header)

        subtitle = QLabel("MIT-BIH Arrhythmia Database | ECG Signal Analysis & Classification")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setFont(QFont("Microsoft YaHei", 10))
        subtitle.setStyleSheet("color: #6c7086; margin-bottom: 10px;")
        layout.addWidget(subtitle)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #45475a; max-height: 2px;")
        layout.addWidget(separator)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        tabBar = self.tabs.tabBar()
        tabBar.setExpanding(True)
        tabBar.setUsesScrollButtons(False)
        tabBar.setDrawBase(False)
        from PyQt6.QtWidgets import QSizePolicy
        tabBar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.tabs.setStyleSheet("""
            QTabBar { qproperty-drawBase: 0; }
            QTabWidget::pane { border: 1px solid #45475a; background-color: #1e1e2e; }
        """)
        layout.addWidget(self.tabs)

        self.preprocessing_page = PreprocessingPage()
        self.feature_page = FeaturePage()
        self.ml_train_page = MLTrainPage()
        self.dl_train_page = DLTrainPage()
        self.model_test_page = ModelTestPage()
        self.realtime_page = RealtimePage()
        self.model_manager_page = ModelManagerPage()

        self.tabs.addTab(self.preprocessing_page, "数据预处理")
        self.tabs.addTab(self.feature_page, "特征提取")
        self.tabs.addTab(self.ml_train_page, "机器学习训练")
        self.tabs.addTab(self.dl_train_page, "深度学习训练")
        self.tabs.addTab(self.model_test_page, "模型测试")
        self.tabs.addTab(self.realtime_page, "实时演示")
        self.tabs.addTab(self.model_manager_page, "模型管理")


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
