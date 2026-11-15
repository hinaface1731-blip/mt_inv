"""Desktop workspace: stable navigation, data explorer and contextual header."""
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__
from ..profile import Profile
from .theme import icon

PAGES = [
    ("Кривые", "Контроль качества импеданса, фаз и типпера", "wave"),
    ("Карта и профили", "Станции наблюдений и геометрия профилей", "map"),
    ("Псевдоразрезы", "Распределение сопротивления и фаз вдоль профиля", "section"),
    ("Инверсия 1D", "Слоистые модели и сшитые разрезы", "layers"),
    ("Инверсия 2D", "Построение модели и контроль подбора TE / TM", "grid"),
    ("АМТЗ + ЗСБ", "Совместная интерпретация и статический сдвиг", "link"),
]


class WorkspaceShell(QtWidgets.QWidget):
    def __init__(self, main, tabs, station_panel):
        super().__init__()
        self.main, self.tabs = main, tabs
        self.jobs = 0
        self.rail = QtWidgets.QFrame(objectName="rail")
        self.rail.setFixedWidth(194)
        rl = QtWidgets.QVBoxLayout(self.rail)
        rl.setContentsMargins(12, 26, 12, 18)
        rl.setSpacing(12)
        self.brand = QtWidgets.QLabel("MT / INV", objectName="brand")
        self.brand_sub = QtWidgets.QLabel("ГЕОФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ", objectName="brandSub")
        self.brand_sub.setWordWrap(True)
        rl.addWidget(self.brand)
        rl.addWidget(self.brand_sub)
        rl.addSpacing(20)
        self.nav = QtWidgets.QListWidget(objectName="navigation")
        self.nav.setIconSize(QtCore.QSize(20, 20))
        self.nav.setAccessibleName("Разделы рабочей области")
        self.nav.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        for i, (title, desc, symbol) in enumerate(PAGES):
            it = QtWidgets.QListWidgetItem(icon(symbol, "#b7d5dd"), title)
            it.setSizeHint(QtCore.QSize(160, 44))
            it.setToolTip(f"{title} · Ctrl+{i + 1}\n{desc}")
            it.setData(QtCore.Qt.AccessibleTextRole, title)
            self.nav.addItem(it)
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(f"Ctrl+{i + 1}"), main)
            shortcut.activated.connect(lambda n=i: tabs.setCurrentIndex(n))
        self.nav.currentRowChanged.connect(tabs.setCurrentIndex)
        tabs.currentChanged.connect(self.page_changed)
        rl.addWidget(self.nav, 1)
        self.help_btn = QtWidgets.QPushButton("Как пользоваться")
        self.help_btn.setIcon(icon("help", "#b7d5dd"))
        self.help_btn.setToolTip("Как пользоваться MT-Inv")
        self.help_btn.clicked.connect(main.help)
        rl.addWidget(self.help_btn)
        self.foot = QtWidgets.QLabel(f"MT-Inv  {__version__}", objectName="brandSub")
        rl.addWidget(self.foot)

        self.station_panel = station_panel
        station_panel.setObjectName("stationPanel")
        station_panel.setMinimumWidth(210)
        station_panel.setMaximumWidth(300)
        workspace = QtWidgets.QWidget()
        body = QtWidgets.QVBoxLayout(workspace)
        body.setContentsMargins(20, 18, 20, 16)
        body.setSpacing(16)
        self.header = QtWidgets.QFrame(objectName="workspaceHeader")
        hl = QtWidgets.QVBoxLayout(self.header)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(10)
        top = QtWidgets.QHBoxLayout()
        self.project = QtWidgets.QLabel("НОВЫЙ ПРОЕКТ", objectName="eyebrow")
        top.addWidget(self.project, 1)
        self.activity = QtWidgets.QLabel("Готово к работе", objectName="activity")
        top.addWidget(self.activity)
        self.data_badge = QtWidgets.QLabel("Нет данных", objectName="dataBadge")
        top.addWidget(self.data_badge)
        hl.addLayout(top)
        title_row = QtWidgets.QHBoxLayout()
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(5)
        self.title = QtWidgets.QLabel(objectName="pageTitle")
        self.description = QtWidgets.QLabel(objectName="pageDescription")
        self.description.setWordWrap(True)
        titles.addWidget(self.title)
        titles.addWidget(self.description)
        title_row.addLayout(titles, 1)
        self.station_toggle = QtWidgets.QPushButton("Станции")
        self.station_toggle.setIcon(icon("sidebar"))
        self.station_toggle.setCheckable(True)
        self.station_toggle.setChecked(True)
        self.station_toggle.setToolTip("Показать / скрыть список станций · Ctrl+L")
        self.station_toggle.toggled.connect(station_panel.setVisible)
        title_row.addWidget(self.station_toggle)
        shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+L"), main)
        shortcut.activated.connect(self.station_toggle.toggle)
        open_btn = QtWidgets.QPushButton("Открыть EDI")
        open_btn.setIcon(icon("folder", "white"))
        open_btn.setProperty("role", "primary")
        open_btn.clicked.connect(main.open_files)
        title_row.addWidget(open_btn)
        self.save_btn = QtWidgets.QPushButton()
        self.save_btn.setIcon(icon("save"))
        self.save_btn.setAccessibleName("Сохранить проект")
        self.save_btn.setToolTip("Сохранить проект · Ctrl+S")
        self.save_btn.clicked.connect(main.save_project)
        title_row.addWidget(self.save_btn)
        hl.addLayout(title_row)
        body.addWidget(self.header)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(3)
        self.progress.hide()
        body.addWidget(self.progress)
        self.stack = QtWidgets.QStackedWidget()
        self.panel = QtWidgets.QFrame(objectName="workspacePanel")
        pl = QtWidgets.QVBoxLayout(self.panel)
        pl.setContentsMargins(10, 10, 10, 10)
        tabs.setObjectName("workspaceTabs")
        tabs.tabBar().hide()
        pl.addWidget(tabs)
        self.stack.addWidget(self.panel)
        self.empty = self.build_empty()
        self.stack.addWidget(self.empty)
        body.addWidget(self.stack, 1)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(station_panel)
        self.splitter.addWidget(workspace)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([244, 1000])
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.rail)
        root.addWidget(self.splitter, 1)
        self.nav.setCurrentRow(0)
        self.page_changed(0)

    def build_empty(self):
        empty = QtWidgets.QFrame(objectName="emptyState")
        l = QtWidgets.QVBoxLayout(empty)
        l.setContentsMargins(36, 30, 36, 30)
        l.setSpacing(16)
        l.addStretch()
        symbol = QtWidgets.QLabel("∿", objectName="emptySymbol")
        symbol.setFixedSize(92, 86)
        symbol.setAlignment(QtCore.Qt.AlignCenter)
        l.addWidget(symbol, 0, QtCore.Qt.AlignHCenter)
        title = QtWidgets.QLabel("Начните с полевых данных", objectName="emptyTitle")
        title.setWordWrap(True)
        title.setAlignment(QtCore.Qt.AlignCenter)
        l.addWidget(title)
        desc = QtWidgets.QLabel("Загрузите EDI, проверьте кривые и соберите профиль.\n"
                                "Затем переходите к построению геоэлектрической модели.",
                                objectName="emptyDescription")
        desc.setAlignment(QtCore.Qt.AlignCenter)
        desc.setWordWrap(True)
        l.addWidget(desc)
        row = QtWidgets.QHBoxLayout()
        row.addStretch()
        b = QtWidgets.QPushButton("Открыть EDI-файлы")
        b.setProperty("role", "primary")
        b.clicked.connect(self.main.open_files)
        row.addWidget(b)
        b = QtWidgets.QPushButton("Открыть проект")
        b.clicked.connect(self.main.open_project)
        row.addWidget(b)
        row.addStretch()
        l.addLayout(row)
        demo = QtWidgets.QPushButton("Посмотреть демоданные")
        demo.setToolTip("Синтетические данные из комплекта программы; не полевые измерения")
        demo.clicked.connect(self.load_demo)
        l.addWidget(demo, 0, QtCore.Qt.AlignHCenter)
        l.addStretch()
        return empty

    def load_demo(self):
        paths = sorted((Path(__file__).resolve().parents[2] / "demo_data").glob("*.edi"))
        self.main.load_paths([str(p) for p in paths])
        sts = list(self.main.stations.values())
        if sts and not self.main.profile_by_name("Демонстрационный профиль"):
            self.main.profiles.append(Profile.fit("Демонстрационный профиль", sts))
            self.main._profiles_changed()
        self.main.tabs.setCurrentIndex(0)
        self.main.statusBar().showMessage("Загружены демонстрационные синтетические данные", 12000)

    def page_changed(self, index):
        if not 0 <= index < len(PAGES):
            return
        self.nav.blockSignals(True)
        self.nav.setCurrentRow(index)
        self.nav.blockSignals(False)
        self.title.setText(PAGES[index][0])
        self.description.setText(PAGES[index][1])
        from .common import MplWidget
        for plot in self.tabs.widget(index).findChildren(MplWidget):
            if not plot.figure.axes:
                plot.draw()
        self.refresh_context()

    def refresh_context(self):
        self.project.setText(Path(self.main.project_path).name if self.main.project_path else "НОВЫЙ ПРОЕКТ")
        n, p = len(self.main.stations), len(self.main.profiles)
        self.data_badge.setText(f"Станций: {n}  ·  Профилей: {p}" if n else "Нет данных")
        self.stack.setCurrentWidget(self.empty if n == 0 and self.tabs.currentIndex() == 0 else self.panel)
        self.save_btn.setEnabled(bool(n or self.main.tems or self.main.project_path))

    def job_started(self):
        self.jobs += 1
        self.update_activity()

    def job_finished(self):
        self.jobs = max(0, self.jobs - 1)
        self.update_activity()

    def update_activity(self):
        self.activity.setText("Выполняется расчёт" if self.jobs else "Готово к работе")
        self.activity.setProperty("busy", bool(self.jobs))
        self.activity.style().unpolish(self.activity)
        self.activity.style().polish(self.activity)
        self.progress.setVisible(bool(self.jobs))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        compact = self.width() < 1280
        self.rail.setFixedWidth(66 if compact else 194)
        self.brand.setText("MT" if compact else "MT / INV")
        self.brand_sub.setVisible(not compact)
        self.help_btn.setText("" if compact else "Как пользоваться")
        self.foot.setVisible(not compact)
        for i, (title, _, _) in enumerate(PAGES):
            self.nav.item(i).setText("" if compact else title)
