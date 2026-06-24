import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle
import torch
import json
import shutil
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QGroupBox, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from dl.cnn_model import MODEL_REGISTRY

MODELS_DIR = "F:/Database/Models"
REGISTRY_FILE = os.path.join(MODELS_DIR, "_registry.json")


def load_registry():
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"models": []}


def save_registry(registry):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(REGISTRY_FILE, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


class ModelManagerPage(QWidget):
    def __init__(self):
        super().__init__()
        self.registry = load_registry()
        self.setup_ui()
        self.refresh_table()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 8, 12, 8)

        layout.addWidget(self._make_title("模型管理"))

        import_group = QGroupBox("导入模型")
        import_layout = QHBoxLayout()
        import_layout.setSpacing(12)

        import_layout.addWidget(QLabel("模型类型:"))
        self.model_type_combo = QComboBox()
        all_types = list(MODEL_REGISTRY.keys()) + ["RandomForest", "SVM"]
        self.model_type_combo.addItems(all_types)
        self.model_type_combo.setFixedWidth(160)
        import_layout.addWidget(self.model_type_combo)

        import_layout.addWidget(QLabel("名称:"))
        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("（导入后自动显示）")
        self.model_name_edit.setFixedWidth(200)
        import_layout.addWidget(self.model_name_edit)

        import_btn = QPushButton("导入模型文件")
        import_btn.setFixedWidth(130)
        import_btn.clicked.connect(self.import_model)
        import_layout.addWidget(import_btn)

        import_layout.addStretch()
        import_group.setLayout(import_layout)
        layout.addWidget(import_group)

        table_group = QGroupBox("已注册模型")
        table_layout = QVBoxLayout()
        self.model_table = QTableWidget()
        self.model_table.setColumnCount(7)
        self.model_table.setHorizontalHeaderLabels(
            ['名称', '类型', '路径', '创建时间', '准确率', '文件大小', '操作']
        )
        self.model_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.model_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)
        self.model_table.setColumnWidth(6, 100)
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table_layout.addWidget(self.model_table)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group, 1)

    def _make_title(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Microsoft YaHei", 16, QFont.Weight.Bold))
        lbl.setStyleSheet("color: #89b4fa; padding: 4px 0px;")
        return lbl

    def import_model(self):
        model_type = self.model_type_combo.currentText()
        is_dl = model_type in MODEL_REGISTRY

        if is_dl:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择模型文件", MODELS_DIR, "PyTorch模型 (*.pth);;所有文件 (*)"
            )
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择模型文件", MODELS_DIR, "Pickle模型 (*.pkl);;所有文件 (*)"
            )

        if not file_path:
            return

        if is_dl:
            try:
                input_length = 300
                model_cls = MODEL_REGISTRY[model_type]
                model = model_cls(input_length=input_length, num_classes=5)
                model.load_state_dict(torch.load(file_path, map_location='cpu'))
                model_name = self.model_name_edit.text().strip() or os.path.splitext(os.path.basename(file_path))[0]
            except Exception as e:
                QMessageBox.warning(self, "导入失败", f"无法加载模型: {str(e)}")
                return
        else:
            try:
                with open(file_path, 'rb') as f:
                    pickle.load(f)
                model_name = self.model_name_edit.text().strip() or os.path.splitext(os.path.basename(file_path))[0]
            except Exception as e:
                QMessageBox.warning(self, "导入失败", f"无法加载模型: {str(e)}")
                return

        dest_path = os.path.join(MODELS_DIR, os.path.basename(file_path))
        if os.path.abspath(file_path) != os.path.abspath(dest_path):
            os.makedirs(MODELS_DIR, exist_ok=True)
            shutil.copy2(file_path, dest_path)

        file_size = os.path.getsize(dest_path)
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"

        entry = {
            "name": model_name,
            "type": model_type,
            "path": dest_path,
            "created": datetime.now().isoformat(),
            "accuracy": "N/A",
            "size": size_str,
        }

        self.registry["models"].append(entry)
        save_registry(self.registry)
        self.refresh_table()
        self.model_name_edit.clear()
        QMessageBox.information(self, "导入成功", f"模型 {model_name} 已导入")

    def refresh_table(self):
        models = self.registry.get("models", [])
        self.model_table.setRowCount(len(models))

        for i, m in enumerate(models):
            self.model_table.setItem(i, 0, QTableWidgetItem(m.get("name", "")))
            self.model_table.setItem(i, 1, QTableWidgetItem(m.get("type", "")))
            self.model_table.setItem(i, 2, QTableWidgetItem(m.get("path", "")))
            self.model_table.setItem(i, 3, QTableWidgetItem(m.get("created", "")[:19]))
            self.model_table.setItem(i, 4, QTableWidgetItem(str(m.get("accuracy", "N/A"))))
            self.model_table.setItem(i, 5, QTableWidgetItem(m.get("size", "")))

            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(4, 2, 4, 2)
            load_btn = QPushButton("加载")
            load_btn.setFixedWidth(60)
            load_btn.setStyleSheet("background-color: #45475a; color: #cdd6f4; padding: 4px 8px; border-radius: 4px; font-size: 12px;")
            load_btn.clicked.connect(lambda checked, idx=i: self.load_model_at(idx))
            btn_layout.addWidget(load_btn)
            unload_btn = QPushButton("卸载")
            unload_btn.setFixedWidth(60)
            unload_btn.setStyleSheet("background-color: #45475a; color: #f38ba8; padding: 4px 8px; border-radius: 4px; font-size: 12px;")
            unload_btn.clicked.connect(lambda checked, idx=i: self.unload_model_at(idx))
            btn_layout.addWidget(unload_btn)
            btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.model_table.setCellWidget(i, 6, btn_widget)

    def load_model_at(self, idx):
        if idx < len(self.registry["models"]):
            m = self.registry["models"][idx]
            QMessageBox.information(self, "加载模型",
                f"名称: {m['name']}\n类型: {m['type']}\n路径: {m['path']}\n\n请在模型测试或实时演示页面中使用此模型路径进行预测。")

    def unload_model_at(self, idx):
        if idx < len(self.registry["models"]):
            m = self.registry["models"][idx]
            reply = QMessageBox.question(self, "卸载模型",
                f"确定要卸载模型 {m['name']} 吗？\n（将从注册表中移除，文件保留在磁盘）",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.registry["models"].pop(idx)
                save_registry(self.registry)
                self.refresh_table()
