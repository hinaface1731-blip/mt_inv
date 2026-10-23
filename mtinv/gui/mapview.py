"""Вкладка «Карта»: станции, подложка, индукционные стрелки, фазовые тензоры,
создание профилей."""

from __future__ import annotations

import numpy as np
from matplotlib.collections import EllipseCollection
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import RectangleSelector
from PySide6 import QtCore, QtWidgets

from .. import tiles
from ..profile import Profile
from .common import MplWidget


class MapTab(QtWidgets.QWidget):
    selectionChanged = QtCore.Signal(list)
    profilesChanged = QtCore.Signal()

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.plot = MplWidget(self)
        self.selected = set()
        self._tile_cache = {}

        ctl = QtWidgets.QHBoxLayout()
        ctl.addWidget(QtWidgets.QLabel("Подложка:"))
        self.base = QtWidgets.QComboBox()
        self.base.addItems(list(tiles.PROVIDERS))
        self.base.currentIndexChanged.connect(self.redraw)
        ctl.addWidget(self.base)
        self.labels = QtWidgets.QCheckBox("подписи")
        self.labels.setChecked(True)
        self.labels.toggled.connect(self.redraw)
        ctl.addWidget(self.labels)
        ctl.addSpacing(10)
        ctl.addWidget(QtWidgets.QLabel("Период:"))
        self.period = QtWidgets.QComboBox()
        self.period.setMinimumWidth(90)
        self.period.currentIndexChanged.connect(self.redraw)
        ctl.addWidget(self.period)
        self.arrows = QtWidgets.QCheckBox("индукц. стрелки")
        self.arrows.toggled.connect(self.redraw)
        ctl.addWidget(self.arrows)
        self.ellipses = QtWidgets.QCheckBox("фазовые тензоры")
        self.ellipses.toggled.connect(self.redraw)
        ctl.addWidget(self.ellipses)
        ctl.addStretch(1)

        ctl2 = QtWidgets.QHBoxLayout()
        self.rect_mode = QtWidgets.QCheckBox("выделять рамкой")
        self.rect_mode.toggled.connect(self._toggle_rect)
        ctl2.addWidget(self.rect_mode)
        b = QtWidgets.QPushButton("Снять выделение")
        b.clicked.connect(lambda: self.set_selection([], emit=True))
        ctl2.addWidget(b)
        b = QtWidgets.QPushButton("➕ Профиль из выделенных")
        b.clicked.connect(self.make_profile)
        ctl2.addWidget(b)
        ctl2.addWidget(QtWidgets.QLabel("Профили:"))
        self.prof_box = QtWidgets.QComboBox()
        self.prof_box.setMinimumWidth(120)
        self.prof_box.currentIndexChanged.connect(self._on_profile_pick)
        ctl2.addWidget(self.prof_box)
        b = QtWidgets.QPushButton("Удалить профиль")
        b.clicked.connect(self.delete_profile)
        ctl2.addWidget(b)
        ctl2.addStretch(1)

        hint = QtWidgets.QLabel("Щелчок по станции — выделить/снять; Ctrl+щелчок — открыть её кривые.")
        hint.setStyleSheet("color: gray")
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(ctl)
        lay.addLayout(ctl2)
        lay.addWidget(hint)
        lay.addWidget(self.plot, 1)
        self.plot.canvas.mpl_connect("button_press_event", self.on_click)
        self.ax = None
        self.rs = None

    # ------------------------------------------------------------------
    def refresh_periods(self):
        sts = list(self.main.stations.values())
        cur = self.period.currentText()
        self.period.blockSignals(True)
        self.period.clear()
        if sts:
            T = np.unique(np.round(np.concatenate([s.period for s in sts]), 6))
            # опорный набор: 2 значения на декаду
            ref = 10 ** np.arange(np.floor(np.log10(T.min())), np.ceil(np.log10(T.max())) + 0.01, 0.5)
            for t in ref:
                if T.min() / 1.5 <= t <= T.max() * 1.5:
                    self.period.addItem(f"{t:g} с", float(t))
        i = self.period.findText(cur)
        self.period.setCurrentIndex(i if i >= 0 else self.period.count() // 2)
        self.period.blockSignals(False)

    def refresh_profiles(self, select=None):
        self.prof_box.blockSignals(True)
        self.prof_box.clear()
        self.prof_box.addItem("—")
        for p in self.main.profiles:
            self.prof_box.addItem(p.name)
        if select:
            self.prof_box.setCurrentText(select)
        self.prof_box.blockSignals(False)

    def set_selection(self, names, emit=False):
        self.selected = set(names)
        self.redraw(keep_view=True)
        if emit:
            self.selectionChanged.emit(sorted(self.selected))

    def _on_profile_pick(self):
        name = self.prof_box.currentText()
        for p in self.main.profiles:
            if p.name == name:
                self.set_selection(p.stations, emit=True)
                return

    def make_profile(self):
        names = [n for n in self.main.stations if n in self.selected]
        if len(names) < 2:
            QtWidgets.QMessageBox.information(self, "Профиль", "Выделите хотя бы 2 станции "
                                              "(щелчками или рамкой на карте, или в списке слева).")
            return
        default = f"Профиль {len(self.main.profiles) + 1}"
        name, ok = QtWidgets.QInputDialog.getText(self, "Новый профиль", "Название:", text=default)
        if not ok or not name:
            return
        sts = [self.main.stations[n] for n in names]
        if any(not np.isfinite(s.lat) or not np.isfinite(s.lon) for s in sts):
            QtWidgets.QMessageBox.warning(self, "Профиль", "У части станций нет координат.")
            return
        p = Profile.fit(name, sts)
        self.main.profiles = [q for q in self.main.profiles if q.name != name] + [p]
        self.refresh_profiles(select=name)
        self.profilesChanged.emit()
        self.redraw(keep_view=True)

    def delete_profile(self):
        name = self.prof_box.currentText()
        self.main.profiles = [p for p in self.main.profiles if p.name != name]
        self.refresh_profiles()
        self.profilesChanged.emit()
        self.redraw(keep_view=True)

    # ------------------------------------------------------------------
    def _toggle_rect(self, on):
        if self.rs is not None:
            self.rs.set_active(on)

    def _on_rect(self, e1, e2):
        x0, x1 = sorted([e1.xdata, e2.xdata])
        y0, y1 = sorted([e1.ydata, e2.ydata])
        sel = set(self.selected)
        for n, st in self.main.stations.items():
            x, y = tiles.merc(st.lat, st.lon)
            if x0 <= x <= x1 and y0 <= y <= y1:
                sel.add(n)
        self.set_selection(sel, emit=True)

    def on_click(self, ev):
        if ev.inaxes is not self.ax or ev.button != 1 or self.plot.toolbar.mode:
            return
        if self.rect_mode.isChecked():
            return
        best, dmin = None, 1e300
        for n, st in self.main.stations.items():
            x, y = tiles.merc(st.lat, st.lon)
            px, py = self.ax.transData.transform((x, y))
            d = np.hypot(px - ev.x, py - ev.y)
            if d < dmin:
                best, dmin = n, d
        if best is None or dmin > 12:
            return
        if ev.key == "control":
            self.main.show_station(best)
            return
        sel = set(self.selected)
        sel.symmetric_difference_update({best})
        self.set_selection(sel, emit=True)

    # ------------------------------------------------------------------
    def redraw(self, keep_view=False):
        fig = self.plot.figure
        view = None
        if keep_view and self.ax is not None:
            view = (self.ax.get_xlim(), self.ax.get_ylim())
        fig.clear()
        self.ax = ax = fig.add_subplot(111)
        sts = [s for s in self.main.stations.values() if np.isfinite(s.lat) and np.isfinite(s.lon)]
        if not sts:
            ax.text(0.5, 0.5, "Загрузите EDI (Файл → Открыть)", ha="center", transform=ax.transAxes)
            self.plot.draw()
            return
        lat = np.array([s.lat for s in sts])
        lon = np.array([s.lon for s in sts])
        x, y = tiles.merc(lat, lon)
        span = max(x.max() - x.min(), y.max() - y.min(), 2000.0)
        pad = 0.15 * span
        xl = (x.min() - pad, x.max() + pad)
        yl = (y.min() - pad, y.max() + pad)
        if view is not None:
            xl, yl = view

        prov = self.base.currentText()
        if tiles.PROVIDERS.get(prov):
            la0, lo0 = tiles.inv_merc(xl[0], yl[0])
            la1, lo1 = tiles.inv_merc(xl[1], yl[1])
            key = (prov, round(float(la0), 3), round(float(la1), 3), round(float(lo0), 3), round(float(lo1), 3))
            if key not in self._tile_cache:
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                try:
                    self._tile_cache[key] = tiles.get_tiles(prov, la0, la1, lo0, lo1)
                finally:
                    QtWidgets.QApplication.restoreOverrideCursor()
                if not self._tile_cache[key]:
                    self.main.statusBar().showMessage("Подложка не загрузилась (нет интернета?)", 5000)
            for img, ext in self._tile_cache[key]:
                ax.imshow(img, extent=ext, origin="upper", zorder=0, interpolation="bilinear")

        # профили
        for p in self.main.profiles:
            px, py = tiles.merc([p.lat0, p.lat1], [p.lon0, p.lon1])
            ax.plot(px, py, "-", color="magenta", lw=2, zorder=2)
            ax.annotate(p.name, (px[0], py[0]), color="magenta", fontsize=9, zorder=5,
                        xytext=(-5, 5), textcoords="offset points", ha="right")

        names = [s.name for s in sts]
        sel = np.array([n in self.selected for n in names])
        ax.scatter(x[~sel], y[~sel], marker="v", s=50, c="white", edgecolors="black", zorder=4)
        if sel.any():
            ax.scatter(x[sel], y[sel], marker="v", s=70, c="red", edgecolors="black", zorder=4)
        if self.labels.isChecked():
            for xi, yi, n in zip(x, y, names):
                ax.annotate(n, (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points",
                            zorder=5)

        T0 = self.period.currentData()
        scale = 0.06 * span
        if T0 and self.ellipses.isChecked():
            w, h, ang, col, xs, ys = [], [], [], [], [], []
            for s, xi, yi in zip(sts, x, y):
                k = int(np.argmin(np.abs(np.log10(s.period) - np.log10(T0))))
                if abs(np.log10(s.period[k] / T0)) > 0.3:
                    continue
                pt = s.phase_tensor()
                pmax, pmin = pt["phimax"][k], pt["phimin"][k]
                if not np.isfinite(pmax) or pmax <= 0:
                    continue
                w.append(scale)
                h.append(scale * max(pmin, 1) / pmax)
                ang.append(90 - pt["strike"][k])   # от севера по часовой -> от оси x
                col.append(pmin)
                xs.append(xi)
                ys.append(yi)
            if xs:
                ec = EllipseCollection(w, h, ang, units="xy", offsets=np.c_[xs, ys],
                                       offset_transform=ax.transData, cmap="jet_r",
                                       edgecolors="k", linewidths=0.5, zorder=3)
                ec.set_array(np.array(col))
                ec.set_clim(0, 90)
                ax.add_collection(ec)
                fig.colorbar(ec, ax=ax, label="Φmin, °", shrink=0.6)
        if T0 and self.arrows.isChecked() and any(s.has_tipper for s in sts):
            for s, xi, yi in zip(sts, x, y):
                ia = s.induction_arrows()
                if ia is None:
                    continue
                k = int(np.argmin(np.abs(np.log10(s.period) - np.log10(T0))))
                if abs(np.log10(s.period[k] / T0)) > 0.3:
                    continue
                rn, re, im_n, im_e = (a[k] for a in ia)
                sc = 3 * scale
                if np.isfinite(rn):
                    ax.arrow(xi, yi, re * sc, rn * sc, color="black", width=scale * 0.05,
                             length_includes_head=True, zorder=6)
                    ax.arrow(xi, yi, im_e * sc, im_n * sc, color="tab:blue", width=scale * 0.03,
                             length_includes_head=True, zorder=6)
            ax.text(0.01, 0.01, "Стрелки Паркинсона: чёрные — Re, синие — Im; длина 1 = "
                    f"{3 * scale / 1000:.1f} км", transform=ax.transAxes, fontsize=8,
                    bbox=dict(fc="white", alpha=0.7))

        ax.set_xlim(xl)
        ax.set_ylim(yl)
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{float(tiles.inv_merc(v, 0)[1]):.2f}°"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{float(tiles.inv_merc(0, v)[0]):.2f}°"))
        ax.set_xlabel("Долгота")
        ax.set_ylabel("Широта")
        if prov in ("OpenStreetMap", "OpenTopoMap (рельеф)"):
            ax.text(0.99, 0.01, "© OpenStreetMap contributors", transform=ax.transAxes,
                    ha="right", fontsize=6, alpha=0.8)
        elif prov.startswith("Esri"):
            ax.text(0.99, 0.01, "© Esri", transform=ax.transAxes, ha="right", fontsize=6, alpha=0.8)
        self.rs = RectangleSelector(ax, self._on_rect, useblit=True, button=[1],
                                    interactive=False)
        self.rs.set_active(self.rect_mode.isChecked())
        self.plot.draw()
