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


def test_curves_fill_buttons():
    import glob

    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mtinv.gui.main_window import MainWindow

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    w = MainWindow()
    w.load_paths(sorted(glob.glob(os.path.join(root, "demo_data", "*.edi")))[:3])
    st = next(iter(w.stations.values()))
    st.mask[10:11] = True
    w.curves.set_station(st)
    w.curves.fill_all()
    assert st.filled[10:11].all() and not st.mask[10:11].any()
    w.curves.unfill_all()
    assert st.mask[10:11].all()
    w.close()
    app.processEvents()
