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
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.labelcolor": "#42596b", "axes.edgecolor": "#cbd6df",
    "axes.titleweight": "semibold", "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#596b7c", "ytick.color": "#596b7c",
    "grid.color": "#dfe6ed", "grid.linewidth": .6,
    "legend.frameon": False, "savefig.facecolor": "white",
})
# Палитры сопротивления: по умолчанию красный — высокоомное, синий — низкоомное.
# Варианты с «_r» — обратная шкала (красный — проводник).
CMAPS = ["jet", "turbo", "Spectral_r", "RdYlBu_r", "rainbow", "viridis",
         "jet_r", "turbo_r"]


class MplWidget(QtWidgets.QWidget):
    """Холст matplotlib с панелью инструментов (масштаб, сохранение PNG/PDF)."""

    def __init__(self, parent=None, figsize=(8, 6)):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setIconSize(QtCore.QSize(17, 17))
        self.toolbar.setMovable(False)
        self.toolbar.setAccessibleName("Инструменты графика")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)

    def draw(self):
        if not self.figure.axes and not self.figure.texts:
            self.figure.text(.5, .52, "Здесь появится результат", ha="center", va="center",
                             fontsize=13, color="#596b7c")
            self.figure.text(.5, .46, "Выберите данные и параметры построения", ha="center",
                             va="center", fontsize=9, color="#758797")
        self.canvas.draw_idle()


class FlowLayout(QtWidgets.QLayout):
    """Wrap toolbar controls instead of forcing the window beyond its screen."""
    def __init__(self, parent=None, spacing=8):
        super().__init__(parent)
        self.items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item):
        self.items.append(item)

    def addSpacing(self, size):
        self.addItem(QtWidgets.QSpacerItem(size, 1, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum))

    def addStretch(self, stretch=0):
        pass  # each row remains left-aligned

    def count(self):
        return len(self.items)

    def itemAt(self, index):
        return self.items[index] if 0 <= index < len(self.items) else None

    def takeAt(self, index):
        return self.items.pop(index) if 0 <= index < len(self.items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self.arrange(QtCore.QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.arrange(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self.items:
            size = size.expandedTo(item.minimumSize())
        return size

    def arrange(self, rect, measure):
        x, y, height = rect.x(), rect.y(), 0
        for item in self.items:
            if item.widget() is not None and item.widget().isHidden():
                continue
            hint = item.sizeHint().expandedTo(item.minimumSize())
            width = min(hint.width(), max(1, rect.width()))
            if x + width > rect.right() + 1 and height:
                x, y, height = rect.x(), y + height + self.spacing(), 0
            if not measure:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), QtCore.QSize(width, hint.height())))
            x += width + self.spacing()
            height = max(height, hint.height())
        return y + height - rect.y()


class CollapsiblePanel(QtWidgets.QWidget):
    def __init__(self, title, layouts, expanded=False):
        super().__init__()
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        self.toggle = QtWidgets.QToolButton(objectName="disclosure")
        self.toggle.setText(title)
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        root.addWidget(self.toggle)
        self.body = QtWidgets.QWidget()
        body = QtWidgets.QVBoxLayout(self.body)
        body.setContentsMargins(10, 2, 10, 8)
        body.setSpacing(10)
        for layout in layouts:
            body.addLayout(layout)
        root.addWidget(self.body)
        self.toggle.toggled.connect(self.set_expanded)
        self.set_expanded(expanded)

    def set_expanded(self, expanded):
        self.toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.body.setVisible(expanded)


def section_label(text):
    label = QtWidgets.QLabel(text)
    label.setProperty("role", "section")
    return label


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
    shell = getattr(getattr(owner, "main", None), "shell", None)
    if shell is not None:
        shell.job_started()
        worker.finished.connect(shell.job_finished)
        worker.failed.connect(shell.job_finished)
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

