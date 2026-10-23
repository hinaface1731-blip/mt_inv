"""Вкладка «1D инверсия»: Оккам по одной станции и «сшитый» разрез по профилю."""

from __future__ import annotations

import csv

import numpy as np
from matplotlib.colors import LogNorm
from PySide6 import QtWidgets

from ..inv1d import bostick, occam1d
from .common import CMAPS, MplWidget, dspin, error_box, ispin, start_worker

COMPS = [("det", "Det (инвариант)"), ("ssq", "SSQ"), ("xy", "XY"), ("yx", "YX"), ("av", "(XY−YX)/2")]


class Tab1D(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.result = None
        self.section = None
        self.plot = MplWidget(self)

        form = QtWidgets.QFormLayout()
        self.comp = QtWidgets.QComboBox()
        for k, t in COMPS:
            self.comp.addItem(t, k)
        form.addRow("Компонента:", self.comp)
        self.rot = dspin(-180, 180, 0, 5, 1, "°")
        form.addRow("Поворот:", self.rot)
        self.nlay = ispin(10, 120, 40)
        form.addRow("Число слоёв:", self.nlay)
        self.zmin = dspin(0, 1e5, 0, 10, 0, " м")
        self.zmin.setSpecialValueText("авто")
        form.addRow("Первый слой:", self.zmin)
        self.zmax = dspin(0, 1e6, 0, 1000, 0, " м")
        self.zmax.setSpecialValueText("авто")
        form.addRow("Макс. глубина:", self.zmax)
        self.frho = dspin(1, 100, 10, 1, 0, " %")
        form.addRow("Мин. ошибка ρa:", self.frho)
        self.fphs = dspin(0.1, 20, 2.9, 0.5, 1, "°")
        form.addRow("Мин. ошибка фазы:", self.fphs)
        self.target = dspin(0.3, 10, 1.0, 0.1, 2)
        form.addRow("Целевой RMS:", self.target)
        self.use = QtWidgets.QComboBox()
        self.use.addItems(["ρa + фаза", "только фаза", "только ρa"])
        form.addRow("Данные:", self.use)

        b1 = QtWidgets.QPushButton("▶ Инверсия текущей станции")
        b1.clicked.connect(self.run_one)
        self.prof = QtWidgets.QComboBox()
        b2 = QtWidgets.QPushButton("▶ Все станции профиля (сшитый разрез)")
        b2.clicked.connect(self.run_profile)
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(CMAPS)
        self.cmap.currentIndexChanged.connect(self.draw_section)
        self.sec_max = dspin(0, 1e6, 0, 1, 1, " км")
        self.sec_max.setSpecialValueText("авто")
        self.sec_max.valueChanged.connect(self.draw_section)
        self.sec_log = QtWidgets.QCheckBox("логарифмическая глубина")
        self.sec_log.toggled.connect(self.draw_section)
        b3 = QtWidgets.QPushButton("Экспорт модели(ей) в CSV…")
        b3.clicked.connect(self.export)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)

        left = QtWidgets.QVBoxLayout()
        left.addLayout(form)
        left.addWidget(b1)
        left.addSpacing(10)
        left.addWidget(QtWidgets.QLabel("Профиль:"))
        left.addWidget(self.prof)
        left.addWidget(b2)
        f2 = QtWidgets.QFormLayout()
        f2.addRow("Палитра:", self.cmap)
        f2.addRow("Глубина разреза:", self.sec_max)
        f2.addRow("", self.sec_log)
        left.addLayout(f2)
        left.addWidget(b3)
        left.addWidget(self.info)
        left.addStretch(1)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(330)

        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(lw)
        lay.addWidget(self.plot, 1)

    def refresh_profiles(self):
        cur = self.prof.currentText()
        self.prof.clear()
        self.prof.addItems([p.name for p in self.main.profiles])
        if cur:
            self.prof.setCurrentText(cur)

    # ------------------------------------------------------------------
    def _params(self):
        u = self.use.currentIndex()
        return dict(n_layers=self.nlay.value(), z_first=self.zmin.value() or None,
                    z_max=self.zmax.value() or None, floor_rho=self.frho.value() / 100,
                    floor_phs=self.fphs.value(), target_rms=self.target.value(),
                    use_rho=u != 1, use_phase=u != 2)

    def _invert(self, st, prm, comp, rot):
        s = st.rotated(rot) if rot else st
        rho, phs, er, ep = s.rho_phase(comp)
        ok = ~st.mask
        return occam1d(st.freq[ok], rho[ok], phs[ok], er[ok], ep[ok], **prm)

    def run_one(self):
        st = self.main.current_station()
        if st is None:
            QtWidgets.QMessageBox.information(self, "1D", "Выберите станцию в списке слева.")
            return
        try:
            self.result = (st.name, self._invert(st, self._params(), self.comp.currentData(),
                                                   self.rot.value()))
        except Exception as e:  # noqa: BLE001
            error_box(self, "Ошибка 1D-инверсии", str(e))
            return
        self.info.setText(f"{st.name}: RMS = {self.result[1].rms:.2f}, "
                          f"итераций {len(self.result[1].history) - 1}")
        self.draw_one()

    def run_profile(self):
        p = self.main.profile_by_name(self.prof.currentText())
        if p is None:
            QtWidgets.QMessageBox.information(self, "1D", "Сначала создайте профиль на вкладке «Карта».")
            return
        sts = [self.main.stations[n] for n in p.stations if n in self.main.stations]
        prm = self._params()
        comp, rot = self.comp.currentData(), self.rot.value()
        # общая сетка слоёв для всех станций
        if prm["z_first"] is None or prm["z_max"] is None:
            rmin, rmax, fmax, fmin = 1e9, 0, 0, 1e9
            for st in sts:
                r, _, _, _ = st.rho_phase(self.comp.currentData())
                r = r[np.isfinite(r) & ~st.mask]
                if len(r):
                    rmin, rmax = min(rmin, r.min()), max(rmax, r.max())
                fmax, fmin = max(fmax, st.freq.max()), min(fmin, st.freq.min())
            prm["z_first"] = prm["z_first"] or 0.3 * 503 * np.sqrt(rmin / fmax)
            prm["z_max"] = prm["z_max"] or 1.5 * 503 * np.sqrt(rmax / fmin)

        def job(progress, log, stop):
            out = []
            for i, st in enumerate(sts):
                if stop():
                    break
                try:
                    r = self._invert(st, prm, comp, rot)
                except Exception as e:  # noqa: BLE001
                    log(f"{st.name}: {e}")
                    continue
                a, _ = p.project(st.lat, st.lon)
                out.append((st.name, float(a), r))
                progress((i + 1, len(sts)))
            return out

        self.info.setText("Считаю…")
        start_worker(self, job, self._profile_done, lambda e: error_box(self, "Ошибка", e),
                     on_progress=lambda v: self.info.setText(f"Станция {v[0]} из {v[1]}…"),
                     on_log=lambda m: self.main.log(m))

    def _profile_done(self, out):
        self.section = (self.prof.currentText(), out)
        if out:
            rms = np.array([r.rms for _, _, r in out])
            self.info.setText(f"Готово: {len(out)} станций, средний RMS {rms.mean():.2f}")
        self.draw_section()

    # ------------------------------------------------------------------
    def draw_one(self):
        if not self.result:
            return
        name, r = self.result
        fig = self.plot.figure
        fig.clear()
        gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1])
        ar = fig.add_subplot(gs[0, 0])
        ap = fig.add_subplot(gs[1, 0], sharex=ar)
        am = fig.add_subplot(gs[:, 1])
        T = 1 / r.freq
        ar.errorbar(T, r.rho_obs, yerr=np.minimum(r.erho, 0.9 * r.rho_obs), fmt="o", ms=4,
                    color="tab:red", label="наблюд.")
        ar.plot(T, r.rho_pred, "-", color="black", label="модель")
        ar.set_xscale("log")
        ar.set_yscale("log")
        ar.set_ylabel("ρa, Ом·м")
        ar.legend(fontsize=8)
        ar.grid(True, which="both", alpha=0.3)
        ar.set_title(f"{name}: {self.comp.currentText()}, RMS {r.rms:.2f}")
        ap.errorbar(T, r.phs_obs, yerr=r.ephs, fmt="o", ms=4, color="tab:blue")
        ap.plot(T, r.phs_pred, "-", color="black")
        ap.set_ylabel("Фаза, °")
        ap.set_xlabel("Период, с")
        ap.set_ylim(0, 90)
        ap.grid(True, which="both", alpha=0.3)
        z = np.r_[r.tops, r.tops[-1] * 1.5]
        z[0] = max(r.tops[1] / 2, 1)
        am.step(np.r_[r.rho, r.rho[-1]], z, where="post", color="black", lw=2, label="Оккам")
        zb, rb = bostick(r.freq, r.rho_obs, r.phs_obs)
        am.plot(rb, zb, ".", color="gray", label="Бостик")
        am.set_xscale("log")
        am.set_yscale("log")
        am.invert_yaxis()
        am.set_xlabel("ρ, Ом·м")
        am.set_ylabel("Глубина, м")
        am.grid(True, which="both", alpha=0.3)
        am.legend(fontsize=8)
        self.plot.draw()

    def draw_section(self):
        if not self.section or not self.section[1]:
            return
        pname, out = self.section
        fig = self.plot.figure
        fig.clear()
        ax = fig.add_subplot(111)
        xs = np.array([a for _, a, _ in out]) / 1000
        o = np.argsort(xs)
        xs = xs[o]
        out = [out[i] for i in o]
        tops = out[0][2].tops
        # по умолчанию — медианная глубина Бостика на самом длинном периоде
        doi = [bostick(r.freq[-1:], r.rho_obs[-1:], r.phs_obs[-1:])[0][0] for _, _, r in out]
        zmax = self.sec_max.value() * 1000 or float(np.median(doi))
        zc = np.r_[tops, tops[-1] * 1.3]
        if self.sec_log.isChecked():
            zc[0] = max(tops[1] / 2, 1.0)
        M = np.array([r.rho for _, _, r in out]).T
        if len(xs) > 1:
            mid = 0.5 * (xs[1:] + xs[:-1])
            xe = np.r_[xs[0] - (mid[0] - xs[0]), mid, xs[-1] + (xs[-1] - mid[-1])]
        else:
            xe = np.array([xs[0] - 0.5, xs[0] + 0.5])
        fin = M[np.isfinite(M)]
        im = ax.pcolormesh(xe, zc / 1000, M, cmap=self.cmap.currentText(),
                           norm=LogNorm(np.percentile(fin, 2), np.percentile(fin, 98)))
        fig.colorbar(im, ax=ax, label="ρ, Ом·м")
        y_st = zc[0] / 1000 if self.sec_log.isChecked() else 0
        ax.plot(xs, np.full_like(xs, y_st), "kv", ms=7, clip_on=False)
        for x, (n, _, r) in zip(xs, out):
            ax.annotate(n, (x, y_st), xytext=(0, 8), textcoords="offset points", fontsize=7,
                        ha="center", rotation=90, annotation_clip=False)
        if self.sec_log.isChecked():
            ax.set_yscale("log")
            ax.set_ylim(zmax / 1000, zc[0] / 1000)
        else:
            ax.set_ylim(zmax / 1000, 0)
        ax.set_xlabel("Расстояние по профилю, км")
        ax.set_ylabel("Глубина, км")
        ax.set_title(f"Сшитый 1D-разрез (Оккам): {pname}, {self.comp.currentText()}", pad=40)
        self.plot.draw()

    def export(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Экспорт", "model1d.csv", "CSV (*.csv)")
        if not path:
            return
        rows = []
        if self.section and self.section[1]:
            for n, a, r in self.section[1]:
                for z, rho in zip(r.tops, r.rho):
                    rows.append([n, f"{a:.1f}", f"{z:.1f}", f"{rho:.4g}"])
        elif self.result:
            n, r = self.result
            for z, rho in zip(r.tops, r.rho):
                rows.append([n, "", f"{z:.1f}", f"{rho:.4g}"])
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["station", "distance_m", "top_depth_m", "rho_ohmm"])
            w.writerows(rows)
        self.main.statusBar().showMessage(f"Сохранено: {path}", 5000)
