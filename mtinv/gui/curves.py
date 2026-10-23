"""Вкладка «Кривые»: rho_a, фаза, типпер, фазовый тензор одной станции."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from .common import COLORS, MplWidget, dspin


class CurvesTab(QtWidgets.QWidget):
    maskChanged = QtCore.Signal()

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.station = None
        self.plot = MplWidget(self)

        ctl = QtWidgets.QHBoxLayout()
        self.cb = {}
        for c, on in [("xy", True), ("yx", True), ("det", False), ("ssq", False),
                      ("xx", False), ("yy", False)]:
            b = QtWidgets.QCheckBox(c.upper() if c not in ("det", "ssq") else
                                    {"det": "Det (инв.)", "ssq": "SSQ"}[c])
            b.setChecked(on)
            b.toggled.connect(self.redraw)
            self.cb[c] = b
            ctl.addWidget(b)
        ctl.addSpacing(15)
        ctl.addWidget(QtWidgets.QLabel("Поворот:"))
        self.rot = dspin(-180, 180, 0.0, 5, 1, "°")
        self.rot.valueChanged.connect(self.redraw)
        ctl.addWidget(self.rot)
        self.show_masked = QtWidgets.QCheckBox("показывать исключённые")
        self.show_masked.setChecked(True)
        self.show_masked.toggled.connect(self.redraw)
        ctl.addWidget(self.show_masked)
        b = QtWidgets.QPushButton("Сбросить исключения")
        b.clicked.connect(self.reset_mask)
        ctl.addWidget(b)
        ctl.addStretch(1)

        hint = QtWidgets.QLabel(
            "Щелчок по точке на графике rho/фазы — исключить/вернуть эту частоту "
            "(для всех компонент). Исключённые точки не идут в инверсию.")
        hint.setStyleSheet("color: gray")
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(ctl)
        lay.addWidget(hint)
        lay.addWidget(self.plot, 1)
        self.plot.canvas.mpl_connect("button_press_event", self.on_click)

    def set_station(self, st):
        self.station = st
        self.redraw()

    def reset_mask(self):
        if self.station is not None:
            self.station.mask[:] = False
            self.maskChanged.emit()
            self.redraw()

    def on_click(self, ev):
        if self.station is None or ev.inaxes is None or ev.button != 1:
            return
        if self.plot.toolbar.mode:  # режим зума/сдвига
            return
        if ev.inaxes not in getattr(self, "_pick_axes", []):
            return
        T = self.station.period
        k = int(np.argmin(np.abs(np.log10(T) - np.log10(max(ev.xdata, 1e-12)))))
        if abs(np.log10(T[k]) - np.log10(max(ev.xdata, 1e-12))) > 0.15:
            return
        self.station.mask[k] = not self.station.mask[k]
        self.maskChanged.emit()
        self.redraw()

    def redraw(self):
        fig = self.plot.figure
        fig.clear()
        st = self.station
        if st is None:
            self.plot.draw()
            return
        st_r = st.rotated(self.rot.value()) if self.rot.value() != 0 else st
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.3, 1.1])
        ar = fig.add_subplot(gs[0, 0])
        ap = fig.add_subplot(gs[1, 0], sharex=ar)
        at = fig.add_subplot(gs[2, 0], sharex=ar)
        apt = fig.add_subplot(gs[0, 1], sharex=ar)
        ask = fig.add_subplot(gs[1, 1], sharex=ar)
        aell = fig.add_subplot(gs[2, 1], sharex=ar)
        self._pick_axes = [ar, ap]
        T = st.period
        m = st.mask
        for c, b in self.cb.items():
            if not b.isChecked():
                continue
            rho, phs, er, ep = st_r.rho_phase(c)
            col = COLORS[c]
            good = ~m & np.isfinite(rho)
            ar.errorbar(T[good], rho[good], yerr=np.minimum(er[good], 0.9 * rho[good]),
                        fmt="o", ms=4, color=col, label=c.upper(), capsize=0, lw=0.8)
            ap.errorbar(T[good], phs[good], yerr=ep[good], fmt="o", ms=4, color=col,
                        capsize=0, lw=0.8)
            if self.show_masked.isChecked() and m.any():
                ar.plot(T[m], rho[m], "x", color=col, alpha=0.5)
                ap.plot(T[m], phs[m], "x", color=col, alpha=0.5)
        ar.set_xscale("log")
        ar.set_yscale("log")
        ar.set_ylabel("ρa, Ом·м")
        ar.grid(True, which="both", alpha=0.3)
        ar.legend(fontsize=8, loc="best")
        rot_txt = f", поворот {self.rot.value():.0f}°" if self.rot.value() else ""
        ar.set_title(f"{st.name}  ({st.lat:.4f}, {st.lon:.4f}){rot_txt}", fontsize=10)
        ap.set_ylabel("Фаза, °")
        ap.set_ylim(-10, 100)
        ap.axhline(45, color="gray", lw=0.5)
        ap.grid(True, which="both", alpha=0.3)

        if st_r.has_tipper:
            T_ = st_r.T
            at.plot(T, T_.real[:, 0], "o-", ms=3, color="tab:red", label="Re Tzx")
            at.plot(T, T_.imag[:, 0], "o--", ms=3, color="tab:red", mfc="none", label="Im Tzx")
            at.plot(T, T_.real[:, 1], "s-", ms=3, color="tab:blue", label="Re Tzy")
            at.plot(T, T_.imag[:, 1], "s--", ms=3, color="tab:blue", mfc="none", label="Im Tzy")
            at.legend(fontsize=7, ncol=4, loc="best")
        else:
            at.text(0.5, 0.5, "нет типпера", transform=at.transAxes, ha="center", color="gray")
        at.set_ylabel("Типпер")
        at.set_xlabel("Период, с")
        at.grid(True, which="both", alpha=0.3)

        pt = st.phase_tensor()
        apt.plot(T, pt["phimax"], "^-", ms=3, label="Φmax")
        apt.plot(T, pt["phimin"], "v-", ms=3, label="Φmin")
        apt.set_ylabel("Фаз. тензор, °")
        apt.legend(fontsize=8)
        apt.grid(True, which="both", alpha=0.3)
        apt.set_title("Размерность и простирание", fontsize=10)
        ask.plot(T, pt["beta"], "o-", ms=3, color="tab:purple", label="β (skew)")
        ask.plot(T, st.swift_skew() * 10, ".-", ms=3, color="gray", label="Swift skew ×10")
        ask.axhspan(-3, 3, color="green", alpha=0.08)
        ask.set_ylabel("Skew, °")
        ask.legend(fontsize=7)
        ask.grid(True, which="both", alpha=0.3)
        strike = np.mod(pt["strike"], 180) - 90
        aell.plot(T, strike, "o", ms=3, color="tab:brown", label="простирание ФТ (±90°)")
        aell.set_ylim(-90, 90)
        aell.set_ylabel("Простир., °")
        aell.set_xlabel("Период, с")
        aell.grid(True, which="both", alpha=0.3)
        aell.legend(fontsize=7)
        self.plot.draw()
