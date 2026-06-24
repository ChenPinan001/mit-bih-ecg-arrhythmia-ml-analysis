from PyQt6.QtWidgets import QTabBar, QTabWidget
from PyQt6.QtCore import QSize


class ExpandingTabBar(QTabBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setExpanding(True)

    def tabSizeHint(self, index):
        count = self.count()
        if count > 0:
            total_w = self.width()
            if total_w > 0:
                w = total_w // count
                return QSize(w, super().tabSizeHint(index).height())
        return super().tabSizeHint(index)


class ExpandingTabWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        tabBar = ExpandingTabBar(self)
        self.setTabBar(tabBar)
        self.setDocumentMode(True)
        tabBar.setExpanding(True)
        tabBar.setUsesScrollButtons(False)
        tabBar.setDrawBase(False)
