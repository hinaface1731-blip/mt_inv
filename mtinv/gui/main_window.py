"""Главное окно MT-Inv."""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__
from ..edi import read_edi, write_edi
from ..tem import read_tem
from ..tsikl import build_soundings
from ..profile import Profile
from .common import error_box
from .curves import CurvesTab
from .mapview import MapTab
from .pseudo import PseudoTab
from .tab1d import Tab1D
from .tab2d import Tab2D
from .tab_tem import TabTEM, auto_link
from .theme import apply_theme, icon
from .shell import WorkspaceShell


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        apply_theme(self)
        self.setWindowTitle(f"MT-Inv {__version__} — инверсия МТ-данных")
        self.resize(1450, 900)
        screen = QtWidgets.QApplication.primaryScreen()
        available = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1450, 900)
        self.setMinimumSize(min(1000, available.width()), min(680, available.height()))
        self.resize(min(1450, available.width()), min(900, available.height()))
        self.stations = {}   # name -> Station (порядок загрузки)
        self.profiles = []
        self.tems = {}       # name -> TEMSounding
        self.project_path = None
        self.settings = QtCore.QSettings("MT-Inv", "MT-Inv")

        # ---- список станций
        self.list = QtWidgets.QListWidget()
        self.list.setObjectName("stationList")
        self.list.setAccessibleName("Станции наблюдений")
        self.list.setIconSize(QtCore.QSize(16, 16))
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list.currentItemChanged.connect(self._on_current)
        self.list.itemSelectionChanged.connect(self._on_list_selection)
        self.info = QtWidgets.QLabel("Выберите станцию, чтобы посмотреть её параметры.")
        self.info.setObjectName("stationInfo")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lw = QtWidgets.QFrame()
        ll = QtWidgets.QVBoxLayout(lw)
        ll.setContentsMargins(16, 24, 16, 16)
        ll.setSpacing(12)
        heading = QtWidgets.QHBoxLayout()
        heading.addWidget(QtWidgets.QLabel("Станции", objectName="stationHeading"), 1)
        self.station_count = QtWidgets.QLabel("0", objectName="stationCount")
        heading.addWidget(self.station_count)
        ll.addLayout(heading)
        self.station_search = QtWidgets.QLineEdit()
        self.station_search.setPlaceholderText("Найти станцию…")
        self.station_search.setClearButtonEnabled(True)
        self.station_search.setAccessibleName("Поиск станции по имени")
        self.station_search.addAction(icon("search"), QtWidgets.QLineEdit.LeadingPosition)
        self.station_search.textChanged.connect(self.filter_stations)
        ll.addWidget(self.station_search)
        b = QtWidgets.QPushButton("Импорт папки с EDI")
        b.setIcon(icon("folder"))
        b.clicked.connect(self.open_folder)
        ll.addWidget(b)
        hint = QtWidgets.QLabel("Ctrl / Shift — выбрать несколько")
        hint.setProperty("role", "muted")
        ll.addWidget(hint)
        ll.addWidget(self.list, 1)
        ll.addWidget(self.info)

        # ---- вкладки
        self.tabs = QtWidgets.QTabWidget()
        self.curves = CurvesTab(self)
        self.map = MapTab(self)
        self.pseudo = PseudoTab(self)
        self.tab1d = Tab1D(self)
        self.tab2d = Tab2D(self)
        self.tabtem = TabTEM(self)
        self.tabs.addTab(self.curves, "Кривые")
        self.tabs.addTab(self.map, "Карта и профили")
        self.tabs.addTab(self.pseudo, "Псевдоразрезы")
        self.tabs.addTab(self.tab1d, "1D инверсия")
        self.tabs.addTab(self.tab2d, "2D инверсия")
        self.tabs.addTab(self.tabtem, "АМТЗ + ЗСБ")
        self.shell = WorkspaceShell(self, self.tabs, lw)
        self.setCentralWidget(self.shell)
        self.map.selectionChanged.connect(self._on_map_selection)
        self.curves.maskChanged.connect(lambda: self._update_info(self.current_station()))
        self.map.profilesChanged.connect(self._profiles_changed)

        # ---- меню
        m = self.menuBar().addMenu("&Файл")
        self._act(m, "Открыть EDI…", self.open_files, "Ctrl+O")
        self._act(m, "Открыть папку с EDI…", self.open_folder, "Ctrl+Shift+O")
        self._act(m, "Загрузить ЗСБ…", self.tabtem.load_files, "Ctrl+T")
        m.addSeparator()
        self._act(m, "Открыть проект…", self.open_project)
        self._act(m, "Сохранить проект", self.save_project, "Ctrl+S")
        self._act(m, "Сохранить проект как…", lambda: self.save_project(ask=True))
        m.addSeparator()
        self._act(m, "Экспорт станций в EDI (с учётом исключений и стат. сдвига)…", self.export_edi)
        self._act(m, "Экспорт координат станций (CSV)…", self.export_coords)
        m.addSeparator()
        self._act(m, "Удалить выбранные станции", self.remove_selected, "Del")
        self._act(m, "Очистить всё", self.clear_all)
        m.addSeparator()
        self._act(m, "Выход", self.close, "Ctrl+Q")
        h = self.menuBar().addMenu("&Справка")
        self._act(h, "Как пользоваться", self.help)

        self.statusBar().showMessage("Откройте EDI-файлы: Файл → Открыть (Ctrl+O)")
        shortcuts = QtWidgets.QLabel("Ctrl+1…6  Разделы    ·    Ctrl+L  Станции")
        shortcuts.setProperty("role", "muted")
        self.statusBar().addPermanentWidget(shortcuts)

    def filter_stations(self, query):
        visible = 0
        for i in range(self.list.count()):
            item = self.list.item(i)
            show = query.casefold() in item.text().casefold()
            item.setHidden(not show)
            visible += int(show)
        total = self.list.count()
        self.station_count.setText(f"{visible} / {total}" if query else str(total))

    def _act(self, menu, text, fn, sc=None):
        a = QtGui.QAction(text, self)
        if sc:
            a.setShortcut(sc)
        a.triggered.connect(fn)
        menu.addAction(a)
        return a

    # ------------------------------------------------------------------
    def log(self, msg):
        self.statusBar().showMessage(msg, 8000)

    def profile_by_name(self, name):
        for p in self.profiles:
            if p.name == name:
                return p
        return None

    def current_station(self):
        it = self.list.currentItem()
        return self.stations.get(it.text()) if it else None

    def show_station(self, name):
        items = self.list.findItems(name, QtCore.Qt.MatchExactly)
        if items:
            self.list.setCurrentItem(items[0])
        self.tabs.setCurrentWidget(self.curves)

    # ------------------------------------------------------------------
    def open_files(self):
        start = self.settings.value("last_dir", "")
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Открыть EDI", start,
                                                          "EDI (*.edi *.EDI);;Все файлы (*)")
        if paths:
            self.settings.setValue("last_dir", os.path.dirname(paths[0]))
            self.load_paths(paths)

    def open_folder(self):
        start = self.settings.value("last_dir", "")
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Папка с EDI", start)
        if not d:
            return
        self.settings.setValue("last_dir", d)
        paths = sorted(set(glob.glob(os.path.join(d, "**", "*.edi"), recursive=True)
                           + glob.glob(os.path.join(d, "**", "*.EDI"), recursive=True)))
        if not paths:
            QtWidgets.QMessageBox.information(self, "EDI", "В папке нет файлов *.edi")
            return
        self.load_paths(paths)

    def load_paths(self, paths, masks=None):
        errors = []
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for p in paths:
                try:
                    st = read_edi(p)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{os.path.basename(p)}: {e}")
                    continue
                name = st.name
                k = 2
                while name in self.stations and self.stations[name].path != p:
                    name = f"{st.name}_{k}"
                    k += 1
                st.name = name
                if masks and p in masks and len(masks[p]) == len(st.mask):
                    st.mask[:] = np.array(masks[p], bool)
                self.stations[name] = st
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._stations_changed()
        msg = f"Загружено станций: {len(self.stations)}"
        nocoord = [n for n, s in self.stations.items() if not np.isfinite(s.lat)]
        if nocoord:
            msg += f"; без координат: {len(nocoord)}"
        self.statusBar().showMessage(msg, 10000)
        if errors:
            error_box(self, "Не все файлы прочитаны",
                      f"Ошибок: {len(errors)}\n\n" + "\n".join(errors))

    def station_data_changed(self, name):
        """Изменились маска или статический сдвиг станции."""
        cur = self.current_station()
        if cur is not None and cur.name == name:
            self._on_current(self.list.currentItem(), None)

    def load_pod_paths(self, paths, params, extra=None):
        """Файлы «Цикла»: группировка по пикетам, сшивка токов."""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            snds, skipped = build_soundings(paths, **params)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            error_box(self, "Ошибка чтения *.pod", str(e))
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self._add_tems(snds, skipped, extra)

    def load_tem_paths(self, paths, defaults=None, extra=None):
        errors = []
        snds = []
        for p in paths:
            try:
                snds.append(read_tem(p, defaults))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{os.path.basename(p)}: {e}")
        self._add_tems(snds, errors, extra)

    def _add_tems(self, snds, errors, extra):
        for snd in snds:
            p = snd.path
            if extra and p in extra:
                x = extra[p]
                snd.link = x.get("link", snd.link)
                if len(x.get("mask", [])) == len(snd.mask):
                    snd.mask[:] = np.array(x["mask"], bool)
            name = snd.name
            k = 2
            while name in self.tems and self.tems[name].path != p:
                name = f"{snd.name}_{k}"
                k += 1
            snd.name = name
            self.tems[name] = snd
        if extra is None:
            auto_link(self.tems, self.stations, 1000.0)
        self.tabtem.refresh()
        self.map.redraw(keep_view=True)
        self.statusBar().showMessage(f"Зондирований ЗСБ: {len(self.tems)}", 8000)
        if errors:
            error_box(self, "Не все файлы ЗСБ прочитаны",
                      f"Ошибок: {len(errors)}\n\n" + "\n".join(errors))
        if snds and extra is None:
            self.tabs.setCurrentWidget(self.tabtem)

    def _stations_changed(self):
        cur = self.list.currentItem().text() if self.list.currentItem() else None
        self.list.blockSignals(True)
        self.list.clear()
        for n in self.stations:
            item = QtWidgets.QListWidgetItem(icon("wave"), n)
            item.setToolTip(n)
            self.list.addItem(item)
        self.list.blockSignals(False)
        if self.stations:
            items = self.list.findItems(cur, QtCore.Qt.MatchExactly) if cur else []
            self.list.setCurrentItem(items[0] if items else self.list.item(0))
        else:
            self.curves.set_station(None)
            self.info.setText("Выберите станцию, чтобы посмотреть её параметры.")
        self.filter_stations(self.station_search.text())
        self.shell.refresh_context()
        self.map.refresh_periods()
        self.map.redraw()
        self.tabtem.refresh_stations()

    def _profiles_changed(self):
        self.map.refresh_profiles()
        self.pseudo.refresh_profiles()
        self.tab1d.refresh_profiles()
        self.tab2d.refresh_profiles()
        self.shell.refresh_context()

    def _on_current(self, cur, _prev):
        st = self.stations.get(cur.text()) if cur else None
        self.curves.set_station(st)
        self._update_info(st)

    def _update_info(self, st):
        if st is not None:
            T = st.period
            self.info.setText(
                f"<b>{st.name}</b><br>Шир: {st.lat:.5f}<br>Долг: {st.lon:.5f}<br>"
                f"Высота: {st.elev:.0f} м<br>Частот: {len(st.freq)} "
                f"(исключено {int(st.mask.sum())})<br>T: {T.min():.3g} – {T.max():.3g} с<br>"
                f"Типпер: {'есть' if st.has_tipper else 'нет'}<br>"
                + (f"<b>Стат. сдвиг:</b> ρxy×{st.ss[0]:.2f}, ρyx×{st.ss[1]:.2f}<br>"
                   if st.has_static_shift else "") +
                f"<span style='color:gray'>{os.path.basename(st.path)}</span>")

    def _on_list_selection(self):
        names = [i.text() for i in self.list.selectedItems()]
        self.map.set_selection(names)

    def _on_map_selection(self, names):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setSelected(it.text() in names)
        self.list.blockSignals(False)

    def remove_selected(self):
        names = [i.text() for i in self.list.selectedItems()]
        if not names:
            return
        if QtWidgets.QMessageBox.question(self, "Удалить", f"Убрать из проекта {len(names)} станций?\n"
                                          "(файлы на диске не трогаются)") != QtWidgets.QMessageBox.Yes:
            return
        for n in names:
            self.stations.pop(n, None)
        self._stations_changed()

    def clear_all(self):
        self.stations.clear()
        self.profiles.clear()
        self.tems.clear()
        self.tabtem.refresh()
        self._stations_changed()
        self._profiles_changed()

    # ------------------------------------------------------------------
    def save_project(self, ask=False):
        path = self.project_path
        if ask or not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить проект", "project.mtinv.json",
                                                            "Проект MT-Inv (*.json)")
            if not path:
                return
        d = dict(version=__version__,
                 stations=[dict(path=s.path,
                                mask=(s._orig[4] if getattr(s, "_orig", None) is not None
                                      else s.mask).astype(int).tolist(),
                                ss=[float(v) for v in s.ss],
                                fill=getattr(s, "fill_opts", None)
                                if getattr(s, "_orig", None) is not None else None)
                           for s in self.stations.values()],
                 tems=[dict(path=t.path, opts=t.opts, link=t.link,
                            mask=t.mask.astype(int).tolist()) for t in self.tems.values()],
                 profiles=[p.to_dict() for p in self.profiles])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        self.project_path = path
        self.statusBar().showMessage(f"Проект сохранён: {path}", 5000)
        self.shell.refresh_context()

    def open_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Открыть проект", "",
                                                        "Проект MT-Inv (*.json)")
        if path:
            self.load_project(path)

    def load_project(self, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.stations.clear()
        base = os.path.dirname(path)
        paths, masks, sss, fills = [], {}, {}, {}

        def fix(p):
            if not os.path.exists(p) and os.path.exists(os.path.join(base, os.path.basename(p))):
                return os.path.join(base, os.path.basename(p))
            return p

        for s in d.get("stations", []):
            p = fix(s["path"])
            paths.append(p)
            masks[p] = s.get("mask", [])
            sss[p] = s.get("ss", [1.0, 1.0])
            if s.get("fill"):
                fills[p] = s["fill"]
        self.load_paths(paths, masks)
        from ..mtdata import fill_gaps
        for st in self.stations.values():
            if st.path in sss:
                st.ss = np.array(sss[st.path], float)
            if st.path in fills:
                fill_gaps(st, **fills[st.path])
        self.tems.clear()
        extra, groups = {}, {}
        for t in d.get("tems", []):
            p = fix(t["path"])
            extra[p] = t
            groups.setdefault(json.dumps(t.get("opts", {}), sort_keys=True), []).append(p)
        for key, ps in groups.items():
            opts = json.loads(key)
            if opts.get("format") == "pod":
                continue
            for k in ("name", "lat", "lon", "elev", "link"):
                opts.pop(k, None)
            self.load_tem_paths(ps, opts, extra)
        # «Цикл»: пересобираем по исходным файлам с теми же параметрами
        pod_groups = {}
        for t in d.get("tems", []):
            o = t.get("opts", {})
            if o.get("format") != "pod":
                continue
            prm = {k: o.get(k) for k in ("units", "rx_turns", "t_min", "t_cross", "snr_min",
                                         "ramp", "rx_mode", "scale_small", "time_shift") if k in o}
            pod_groups.setdefault(json.dumps(prm, sort_keys=True), []).extend(
                fix(f) for f in o.get("files", []))
        for key, files in pod_groups.items():
            self.load_pod_paths(files, json.loads(key), extra)
        self.profiles = [Profile.from_dict(p) for p in d.get("profiles", [])]
        self._profiles_changed()
        self.map.redraw()
        self.project_path = path

        self.shell.refresh_context()

    def export_edi(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Папка для EDI")
        if not d:
            return
        for st in self.stations.values():
            s2 = type(st)(st.name, st.lat, st.lon, st.elev, st.freq[~st.mask], st.Zc[~st.mask],
                          st.Zerrc[~st.mask], None if st.T is None else st.T[~st.mask],
                          None if st.Terr is None else st.Terr[~st.mask])
            s2.zrot = st.zrot
            write_edi(s2, os.path.join(d, f"{st.name}.edi"))
        self.statusBar().showMessage(f"Записано {len(self.stations)} EDI в {d}", 5000)

    def export_coords(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Координаты", "stations.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("name;lat;lon;elev;file\n")
            for s in self.stations.values():
                f.write(f"{s.name};{s.lat:.6f};{s.lon:.6f};{s.elev:.1f};{s.path}\n")

    def help(self):
        QtWidgets.QMessageBox.information(self, "Как пользоваться", HELP)


HELP = """1. Файл → Открыть EDI / папку. Станции появятся в списке слева.
2. «Кривые»: кажущиеся сопротивления, фазы, типпер, фазовый тензор.
   Щелчок по точке исключает частоту (шум) — она не пойдёт в инверсию.
3. «Карта и профили»: выделите станции щелчками или рамкой (или в списке),
   нажмите «Профиль из выделенных». Подложку карты можно включить (нужен интернет).
4. «Псевдоразрезы»: rho/фаза вдоль профиля.
5. «1D инверсия»: Оккам по одной станции или «сшитый» разрез по профилю.
6. «АМТЗ + ЗСБ»: загрузите ЗСБ, привяжите к МТ-станциям, сделайте инверсию ЗСБ,
   исправьте статический сдвиг АМТЗ по модели ЗСБ и/или запустите совместную
   инверсию (с подбором сдвига). Сдвиг потом учитывается везде, в т.ч. в 2D.
7. «2D инверсия»: выберите профиль, простирание (кнопка «⟂ профилю» или «ФТ»),
   моды TE/TM и нажмите «Запустить». Разрез обновляется на каждой итерации.
   Результат можно сохранить (.npz) и выгрузить в CSV (x, глубина, rho, lat, lon).
Проект (список файлов, исключённые точки, профили): Файл → Сохранить проект."""


def main(argv=None):
    import warnings

    warnings.filterwarnings("ignore")
    app = QtWidgets.QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("MT-Inv")
    w = MainWindow()
    w.show()
    args = (sys.argv if argv is None else argv)[1:]
    if args:
        if args[0].endswith(".json"):
            w.load_project(args[0])
        else:
            paths = []
            for a in args:
                if os.path.isdir(a):
                    paths += sorted(glob.glob(os.path.join(a, "*.edi")) + glob.glob(os.path.join(a, "*.EDI")))
                else:
                    paths.append(a)
            w.load_paths(paths)
    return app.exec()

