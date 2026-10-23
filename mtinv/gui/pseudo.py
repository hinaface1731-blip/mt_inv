"""Вкладка «Псевдоразрезы» по профилю."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LogNorm
from PySide6 import QtWidgets

from .common import CMAPS, MplWidget, dspin


def profile_grid(stations, profile, comp, rot=0.0):
    """Данные станций профиля на общей сетке периодов (для изображений)."""
    xs, data = [], []
    for st in stations:
        a, _ = profile.project(st.lat, st.lon)
        s = st.rotated(rot) if rot else st
        rho, phs, _, _ = s.rho_phase(comp)
        rho = np.where(st.mask, np.nan, rho)
        phs = np.where(st.mask, np.nan, phs)
        xs.append(a)
        data.append((st.period, rho, phs))
    order = np.argsort(xs)
    xs = np.array(xs)[order] / 1000.0
    data = [data[i] for i in order]
    Tall = np.concatenate([d[0] for d in data])
    lT = np.linspace(np.log10(Tall.min()), np.log10(Tall.max()), 60)
    R = np.full((len(lT), len(xs)), np.nan)
    P = R.copy()
    for j, (T, rho, phs) in enumerate(data):
        o = np.argsort(T)
        lt = np.log10(T[o])
        ok = np.isfinite(rho[o]) & (rho[o] > 0)
        if ok.sum() < 2:
            continue
        inside = (lT >= lt[ok].min()) & (lT <= lt[ok].max())
        R[inside, j] = 10 ** np.interp(lT[inside], lt[ok], np.log10(rho[o][ok]))
        P[inside, j] = np.interp(lT[inside], lt[ok], phs[o][ok])
    return xs, lT, R, P, [stations[i].name for i in order]


def _edges(v):
    v = np.asarray(v, float)
    if len(v) == 1:
        return np.array([v[0] - 0.5, v[0] + 0.5])
    mid = 0.5 * (v[1:] + v[:-1])
    return np.r_[v[0] - (mid[0] - v[0]), mid, v[-1] + (v[-1] - mid[-1])]


class PseudoTab(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.plot = MplWidget(self)
        ctl = QtWidgets.QHBoxLayout()
        ctl.addWidget(QtWidgets.QLabel("Профиль:"))
        self.prof = QtWidgets.QComboBox()
        self.prof.setMinimumWidth(120)
        ctl.addWidget(self.prof)
        ctl.addWidget(QtWidgets.QLabel("Компоненты:"))
        self.comp = QtWidgets.QComboBox()
        self.comp.addItems(["xy + yx", "det", "ssq", "xy", "yx"])
        ctl.addWidget(self.comp)
        ctl.addWidget(QtWidgets.QLabel("Поворот:"))
        self.rot = dspin(-180, 180, 0, 5, 1, "°")
        ctl.addWidget(self.rot)
        ctl.addWidget(QtWidgets.QLabel("Палитра:"))
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(CMAPS)
        ctl.addWidget(self.cmap)
        b = QtWidgets.QPushButton("Построить")
        b.clicked.connect(self.redraw)
        ctl.addWidget(b)
        ctl.addStretch(1)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(ctl)
        lay.addWidget(self.plot, 1)
        for w in (self.prof, self.comp, self.cmap):
            w.currentIndexChanged.connect(self.redraw)

    def refresh_profiles(self):
        cur = self.prof.currentText()
        self.prof.blockSignals(True)
        self.prof.clear()
        self.prof.addItems([p.name for p in self.main.profiles])
        if cur:
            self.prof.setCurrentText(cur)
        self.prof.blockSignals(False)

    def redraw(self):
        fig = self.plot.figure
        fig.clear()
        p = self.main.profile_by_name(self.prof.currentText())
        if p is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Создайте профиль на вкладке «Карта»", ha="center",
                    transform=ax.transAxes)
            self.plot.draw()
            return
        sts = [self.main.stations[n] for n in p.stations if n in self.main.stations]
        sel = self.comp.currentText()
        comps = ["xy", "yx"] if sel == "xy + yx" else [sel]
        axes = fig.subplots(2, len(comps), sharex=True, sharey=True, squeeze=False)
        cmap = self.cmap.currentText()
        for j, c in enumerate(comps):
            xs, lT, R, P, names = profile_grid(sts, p, c, self.rot.value())
            xe, ye = _edges(xs), _edges(lT)
            ax = axes[0, j]
            fin = R[np.isfinite(R)]
            vmin, vmax = (np.percentile(fin, 2), np.percentile(fin, 98)) if len(fin) else (1, 1000)
            im = ax.pcolormesh(xe, ye, R, norm=LogNorm(vmin, vmax), cmap=cmap)
            fig.colorbar(im, ax=ax, label="ρa, Ом·м")
            ax.set_title(f"ρa {c.upper()}")
            ax2 = axes[1, j]
            im2 = ax2.pcolormesh(xe, ye, P, vmin=0, vmax=90, cmap=cmap.replace("_r", "") if
                                 cmap.endswith("_r") else cmap + "_r")
            fig.colorbar(im2, ax=ax2, label="Фаза, °")
            ax2.set_title(f"Фаза {c.upper()}")
            ax2.set_xlabel("Расстояние по профилю, км")
            for a in (ax, ax2):
                a.plot(xs, np.full(len(xs), ye[0]), "kv", ms=5, clip_on=False)
        for a in axes[:, 0]:
            a.set_ylabel("lg T, с")
        axes[0, 0].invert_yaxis()
        fig.suptitle(f"{p.name}: азимут {p.azimuth:.0f}°, длина {p.length / 1000:.1f} км")
        self.plot.draw()
