"""Интерфейс собирается (ловит ошибки импорта и создания вкладок)."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_main_window_builds():
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mtinv.gui.main_window import MainWindow

    w = MainWindow()
    assert w.tabs.count() == 6
    w.close()
    app.processEvents()
