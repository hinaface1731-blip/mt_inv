"""Interaction and responsive layout checks for the desktop workspace."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets, QtTest
from mtinv.gui.main_window import MainWindow
from mtinv.gui.common import FlowLayout, start_worker


def make_window():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.processEvents()
    return app, w


def test_navigation_shortcuts_and_programmatic_selection_stay_in_sync():
    app, w = make_window()
    w.shell.nav.setCurrentRow(4)
    assert w.tabs.currentWidget() is w.tab2d
    assert w.shell.title.text() == "Инверсия 2D"
    w.tabs.setCurrentWidget(w.map)
    assert w.shell.nav.currentRow() == 1
    QtTest.QTest.keyClick(w, QtCore.Qt.Key_3, QtCore.Qt.ControlModifier)
    app.processEvents()
    assert w.tabs.currentWidget() is w.pseudo
    w.close()


def test_empty_state_demo_search_and_clear():
    app, w = make_window()
    assert w.shell.stack.currentWidget() is w.shell.empty
    w.shell.load_demo()
    assert len(w.stations) == 15 and len(w.profiles) == 1
    # Call directly as well: Qt signal handlers otherwise print exceptions
    # without necessarily failing the test process.
    w.curves.set_station(w.current_station())
    assert len(w.curves.plot.figure.axes) == 6
    axes = w.curves.plot.figure.axes
    assert all(axes[i].has_data() for i in (0, 1, 3, 4, 5))
    assert any("нет типпера" in text.get_text() for text in axes[2].texts)
    assert "TMR001" in w.info.text()
    assert w.shell.stack.currentWidget() is w.shell.panel
    w.station_search.setText("tmr001")
    shown = [w.list.item(i).text() for i in range(w.list.count()) if not w.list.item(i).isHidden()]
    assert shown == ["TMR001"]
    assert len(w.stations) == 15  # filtering must never mutate or mask data
    w.station_search.clear()
    assert all(not w.list.item(i).isHidden() for i in range(w.list.count()))
    w.clear_all()
    assert w.shell.stack.currentWidget() is w.shell.empty
    assert w.station_count.text() == "0"
    w.close()


def test_compact_navigation_and_station_toggle():
    app, w = make_window()
    w.resize(1100, 760)
    app.processEvents()
    assert w.width() == 1100
    assert w.shell.rail.width() == 66
    assert all(w.shell.nav.item(i).toolTip() for i in range(6))
    w.shell.station_toggle.setChecked(False)
    assert w.shell.station_panel.isHidden()
    w.shell.station_toggle.setChecked(True)
    assert not w.shell.station_panel.isHidden()
    w.resize(1500, 940)
    app.processEvents()
    assert w.shell.rail.width() == 210
    w.close()


def test_busy_state_counts_concurrent_jobs():
    app, w = make_window()
    w.shell.job_started()
    w.shell.job_started()
    assert not w.shell.progress.isHidden()
    w.shell.job_finished()
    assert w.shell.jobs == 1
    w.shell.job_finished()
    assert w.shell.jobs == 0 and w.shell.progress.isHidden()
    w.close()


def test_toolbar_wraps_at_small_width():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    host = QtWidgets.QWidget()
    flow = FlowLayout(host)
    for text in ("Компоненты", "Поворот", "Отбраковка", "Экспорт"):
        flow.addWidget(QtWidgets.QPushButton(text))
    assert flow.heightForWidth(180) > flow.heightForWidth(700)


def test_background_job_returns_workspace_to_ready():
    app, w = make_window()
    results, errors = [], []
    start_worker(w.tab2d, lambda progress, log, stop: 42, results.append, errors.append)
    assert w.shell.jobs == 1
    timer = QtCore.QElapsedTimer()
    timer.start()
    while w.shell.jobs and timer.elapsed() < 3000:
        QtTest.QTest.qWait(10)
    QtTest.QTest.qWait(30)  # allow the worker thread's queued quit/delete events
    app.processEvents()
    assert results == [42] and errors == []
    assert w.shell.jobs == 0 and w.shell.progress.isHidden()
    w.close()
