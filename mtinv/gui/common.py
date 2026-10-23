"""Общие элементы интерфейса: холст matplotlib, фоновые задачи."""

from __future__ import annotations

import traceback

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6 import QtCore, QtWidgets  # noqa: E402

COLORS = {"xy": "tab:red", "yx": "tab:blue", "det": "black", "ssq": "tab:green",
          "xx": "tab:orange", "yy": "tab:purple"}
CMAPS = ["jet_r", "turbo_r", "Spectral", "RdYlBu", "viridis_r", "rainbow_r"]


class MplWidget(QtWidgets.QWidget):
    """Холст matplotlib с панелью инструментов (масштаб, сохранение PNG/PDF)."""

    def __init__(self, parent=None, figsize=(8, 6)):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)

    def draw(self):
        self.canvas.draw_idle()


class Worker(QtCore.QObject):
    """Запуск долгой функции в отдельном потоке.

    fn(progress, log, stop_flag) -> result
    """

    progress = QtCore.Signal(object)
    log = QtCore.Signal(str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self._stop = False

    def stop(self):
        self._stop = True

    @QtCore.Slot()
    def run(self):
        try:
            res = self.fn(self.progress.emit, self.log.emit, lambda: self._stop)
            self.finished.emit(res)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


def start_worker(owner, fn, on_done, on_fail, on_progress=None, on_log=None):
    """Создаёт поток + Worker; возвращает worker (для stop())."""
    thread = QtCore.QThread(owner)
    worker = Worker(fn)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(on_done)
    worker.failed.connect(on_fail)
    if on_progress:
        worker.progress.connect(on_progress)
    if on_log:
        worker.log.connect(on_log)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(thread.deleteLater)
    owner._threads = getattr(owner, "_threads", []) + [(thread, worker)]
    thread.start()
    return worker


def dspin(lo, hi, val, step=1.0, dec=2, suffix=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(step)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


def ispin(lo, hi, val):
    w = QtWidgets.QSpinBox()
    w.setRange(lo, hi)
    w.setValue(val)
    return w


def error_box(parent, title, text):
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Warning)
    box.setWindowTitle(title)
    first, _, rest = text.partition("\n\n")
    box.setText(first)
    if rest:
        box.setDetailedText(rest)
    box.exec()
