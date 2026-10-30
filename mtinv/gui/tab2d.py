"""Вкладка «2D инверсия» (SimPEG)."""

from __future__ import annotations

import csv
import json

import numpy as np
from matplotlib.colors import LogNorm
from PySide6 import QtWidgets

from ..inv2d import Inv2DSettings, run_inversion
from ..mtdata import median_strike
from .common import CMAPS, MplWidget, dspin, error_box, ispin, start_worker


class Tab2D(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.result = None
        self.live = None   # (mesh, rho, core, x, names) во время счёта
        self.worker = None

        # ---- параметры
        form = QtWidgets.QFormLayout()
        self.prof = QtWidgets.QComboBox()
        self.prof.currentIndexChanged.connect(self.auto_strike_profile)
        form.addRow("Профиль:", self.prof)
        mrow = QtWidgets.QHBoxLayout()
        self.te = QtWidgets.QCheckBox("TE")
        self.te.setChecked(True)
        self.tm = QtWidgets.QCheckBox("TM")
        self.tm.setChecked(True)
        mrow.addWidget(self.te)
        mrow.addWidget(self.tm)
        mrow.addStretch(1)
        form.addRow("Моды:", mrow)
        srow = QtWidgets.QHBoxLayout()
        self.strike = dspin(-180, 180, 0, 5, 1, "°")
        srow.addWidget(self.strike)
        b = QtWidgets.QPushButton("⟂ профилю")
        b.setToolTip("Простирание = азимут профиля − 90°")
        b.clicked.connect(self.auto_strike_profile)
        srow.addWidget(b)
        b = QtWidgets.QPushButton("ФТ")
        b.setToolTip("Медианное простирание по фазовому тензору (неоднозначность 90°!)")
        b.clicked.connect(self.auto_strike_pt)
        srow.addWidget(b)
        form.addRow("Простирание:", srow)
        self.frho = dspin(1, 100, 10, 1, 0, " %")
        form.addRow("Мин. ошибка ρa:", self.frho)
        self.frho_tm = dspin(0, 1000, 0, 10, 0, " %")
        self.frho_tm.setSpecialValueText("как у TE")
        self.frho_tm.setToolTip("Большое значение (100 % и больше) — TM подбирается почти только по "
                                "фазе: так делают, когда ρa TM искажены статическим сдвигом")
        form.addRow("Мин. ошибка ρa TM:", self.frho_tm)
        self.fphs = dspin(0.1, 30, 2.9, 0.5, 1, "°")
        form.addRow("Мин. ошибка фазы:", self.fphs)
        self.fpd = ispin(2, 12, 5)
        form.addRow("Частот на декаду:", self.fpd)
        prow = QtWidgets.QHBoxLayout()
        self.tmin = dspin(0, 1e6, 0, 0.001, 4)
        self.tmin.setSpecialValueText("мин")
        self.tmax = dspin(0, 1e6, 0, 10, 2)
        self.tmax.setSpecialValueText("макс")
        prow.addWidget(self.tmin)
        prow.addWidget(QtWidgets.QLabel("–"))
        prow.addWidget(self.tmax)
        form.addRow("Периоды, с:", prow)
        self.rho0 = dspin(0, 1e5, 0, 10, 1, " Ом·м")
        self.rho0.setSpecialValueText("авто (медиана)")
        form.addRow("Старт. модель:", self.rho0)
        self.cells = ispin(1, 10, 3)
        form.addRow("Ячеек между ст.:", self.cells)
        self.zcore = dspin(0, 1000, 0, 1, 1, " км")
        self.zcore.setSpecialValueText("авто")
        form.addRow("Глубина обл.:", self.zcore)
        self.niter = ispin(1, 200, 15)
        form.addRow("Макс. итераций:", self.niter)
        self.smx = dspin(0.01, 100, 1.0, 0.5, 2)
        form.addRow("Гладкость по X:", self.smx)
        self.smz = dspin(0.01, 100, 1.0, 0.5, 2)
        form.addRow("Гладкость по Z:", self.smz)
        self.beta = dspin(0.001, 1000, 10.0, 1.0, 3)
        self.beta.setToolTip("Начальное отношение beta0 (больше — более гладкий старт)")
        form.addRow("β0 ratio:", self.beta)

        self.run_btn = QtWidgets.QPushButton("▶ Запустить 2D инверсию")
        self.run_btn.clicked.connect(self.run)
        self.stop_btn = QtWidgets.QPushButton("■ Стоп")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.logw = QtWidgets.QPlainTextEdit()
        self.logw.setReadOnly(True)
        self.logw.setMaximumBlockCount(2000)

        brow = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Сохранить результат…")
        b.clicked.connect(self.save_result)
        brow.addWidget(b)
        b = QtWidgets.QPushButton("Открыть…")
        b.clicked.connect(self.load_result)
        brow.addWidget(b)
        b2 = QtWidgets.QPushButton("Экспорт CSV…")
        b2.clicked.connect(self.export_csv)

        left = QtWidgets.QVBoxLayout()
        left.addLayout(form)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(self.run_btn, 1)
        r.addWidget(self.stop_btn)
        left.addLayout(r)
        left.addWidget(self.status)
        left.addWidget(self.logw, 1)
        left.addLayout(brow)
        left.addWidget(b2)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(360)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(lw)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(380)

        # ---- графики
        self.sec = MplWidget(self)
        sec_ctl = QtWidgets.QHBoxLayout()
        sec_ctl.addWidget(QtWidgets.QLabel("Палитра:"))
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(CMAPS)
        sec_ctl.addWidget(self.cmap)
        sec_ctl.addWidget(QtWidgets.QLabel("ρ от"))
        self.vmin = dspin(0, 1e6, 0, 1, 2)
        self.vmin.setSpecialValueText("авто")
        sec_ctl.addWidget(self.vmin)
        sec_ctl.addWidget(QtWidgets.QLabel("до"))
        self.vmax = dspin(0, 1e7, 0, 10, 1)
        self.vmax.setSpecialValueText("авто")
        sec_ctl.addWidget(self.vmax)
        sec_ctl.addWidget(QtWidgets.QLabel("Глубина:"))
        self.zplot = dspin(0, 1000, 0, 1, 1, " км")
        self.zplot.setSpecialValueText("авто")
        sec_ctl.addWidget(self.zplot)
        self.smooth = QtWidgets.QCheckBox("сглаживать")
        sec_ctl.addWidget(self.smooth)
        self.equal = QtWidgets.QCheckBox("1:1")
        sec_ctl.addWidget(self.equal)
        self.doi = QtWidgets.QCheckBox("затенять ниже глубинности")
        self.doi.setChecked(True)
        self.doi.setToolTip("Ячейки, чувствительность которых к данным меньше порога, затеняются: "
                            "там модель определяется регуляризацией, а не данными")
        sec_ctl.addWidget(self.doi)
        self.doi_thr = dspin(0.01, 50, 1.0, 0.5, 2, " %")
        self.doi_thr.setToolTip("Порог относительной чувствительности")
        sec_ctl.addWidget(self.doi_thr)
        sec_ctl.addStretch(1)
        for w in (self.cmap,):
            w.currentIndexChanged.connect(self.draw_section)
        for w in (self.vmin, self.vmax, self.zplot, self.doi_thr):
            w.valueChanged.connect(self.draw_section)
        for w in (self.smooth, self.equal, self.doi):
            w.toggled.connect(self.draw_section)
        secw = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(secw)
        sl.addLayout(sec_ctl)
        sl.addWidget(self.sec, 1)

        self.fit = MplWidget(self)
        fitw = QtWidgets.QWidget()
        fl = QtWidgets.QVBoxLayout(fitw)
        fc = QtWidgets.QHBoxLayout()
        fc.addWidget(QtWidgets.QLabel("Мода:"))
        self.fit_mode = QtWidgets.QComboBox()
        self.fit_mode.addItems(["TE", "TM"])
        self.fit_mode.currentIndexChanged.connect(self.draw_fit)
        fc.addWidget(self.fit_mode)
        fc.addStretch(1)
        fl.addLayout(fc)
        fl.addWidget(self.fit, 1)

        self.curv = MplWidget(self)
        curw = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(curw)
        cc = QtWidgets.QHBoxLayout()
        cc.addWidget(QtWidgets.QLabel("Станция:"))
        self.cur_st = QtWidgets.QComboBox()
        self.cur_st.currentIndexChanged.connect(self.draw_curves)
        cc.addWidget(self.cur_st)
        cc.addStretch(1)
        cl.addLayout(cc)
        cl.addWidget(self.curv, 1)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(secw, "Разрез")
        self.tabs.addTab(fitw, "Подбор (псевдоразрезы)")
        self.tabs.addTab(curw, "Подбор (кривые)")

        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(scroll)
        lay.addWidget(self.tabs, 1)

    # ------------------------------------------------------------------
    def refresh_profiles(self):
        cur = self.prof.currentText()
        self.prof.blockSignals(True)
        self.prof.clear()
        self.prof.addItems([p.name for p in self.main.profiles])
        if cur:
            self.prof.setCurrentText(cur)
        self.prof.blockSignals(False)
        self.auto_strike_profile()

    def auto_strike_profile(self):
        p = self.main.profile_by_name(self.prof.currentText())
        if p is not None:
            self.strike.setValue(((p.azimuth - 90) + 180) % 360 - 180)

    def auto_strike_pt(self):
        p = self.main.profile_by_name(self.prof.currentText())
        if p is None:
            return
        sts = [self.main.stations[n] for n in p.stations if n in self.main.stations]
        s = median_strike(sts)
        if not np.isfinite(s):
            return
        # из двух вариантов (s, s-90) берём ближайший к перпендикуляру профиля
        perp = p.azimuth - 90
        cands = [s, s - 90, s + 90, s - 180]
        best = min(cands, key=lambda c: abs(((c - perp) + 90) % 180 - 90))
        self.strike.setValue(((best + 180) % 360) - 180)
        self.main.statusBar().showMessage(
            f"Простирание по ФТ: {s:.1f}° (±90°). Выбран вариант ближе к ⟂ профилю: {best:.1f}°", 8000)

    def settings(self):
        modes = tuple(m for m, b in (("te", self.te), ("tm", self.tm)) if b.isChecked())
        return Inv2DSettings(
            modes=modes, strike=self.strike.value(), floor_rho=self.frho.value() / 100,
            floor_rho_tm=(self.frho_tm.value() / 100) or None,
            floor_phs=self.fphs.value(), freq_per_decade=self.fpd.value(),
            fmin=(1 / self.tmax.value()) if self.tmax.value() else None,
            fmax=(1 / self.tmin.value()) if self.tmin.value() else None,
            rho0=self.rho0.value() or None, cells_between=self.cells.value(),
            z_core=self.zcore.value() * 1000 or None, max_iter=self.niter.value(),
            smooth_x=self.smx.value(), smooth_z=self.smz.value(),
            beta0_ratio=self.beta.value())

    # ------------------------------------------------------------------
    def log(self, msg):
        self.logw.appendPlainText(msg)

    def run(self):
        p = self.main.profile_by_name(self.prof.currentText())
        if p is None:
            QtWidgets.QMessageBox.information(self, "2D", "Сначала создайте профиль на вкладке «Карта».")
            return
        s = self.settings()
        if not s.modes:
            QtWidgets.QMessageBox.information(self, "2D", "Выберите хотя бы одну моду (TE/TM).")
            return
        sts = [self.main.stations[n] for n in p.stations if n in self.main.stations]
        if len(sts) < 2:
            QtWidgets.QMessageBox.information(self, "2D", "В профиле меньше 2 станций.")
            return
        self.logw.clear()
        self.log(f"Профиль {p.name}: {len(sts)} станций, азимут {p.azimuth:.1f}°, "
                 f"простирание {s.strike:.1f}°")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("Идёт расчёт… (первая итерация самая долгая)")
        self._names = [st.name for st in sts]
        self._profile = p

        def job(progress, log, stop):
            return run_inversion(sts, p, s, progress=lambda it, rms, rho: progress((it, rms, rho)),
                                 stop_flag=stop, log=log, on_setup=lambda r: progress(r))

        self.worker = start_worker(self, job, self.done, self.failed, on_progress=self.on_progress,
                                   on_log=self.log)

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.status.setText("Останавливаю после текущей итерации…")

    def on_progress(self, v):
        if not isinstance(v, tuple):  # сетка построена — заготовка результата
            self.result = v
            self.tabs.setCurrentIndex(0)
            self.draw_section()
            return
        it, rms, rho = v
        self.status.setText(f"Итерация {it}: RMS = {rms:.3f}")
        if self.result is not None and self.result.mesh.n_cells == len(rho):
            self.result.rho = rho
            self.result.rms = rms
            self.draw_section()

    def done(self, res):
        self.result = res
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText(f"Готово: RMS = {res.rms:.3f}")
        self.cur_st.blockSignals(True)
        self.cur_st.clear()
        self.cur_st.addItems(res.data.names)
        self.cur_st.blockSignals(False)
        self.fit_mode.clear()
        self.fit_mode.addItems([m.upper() for m in res.pred])
        self.draw_all()

    def failed(self, err):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText("Ошибка")
        self.log(err)
        error_box(self, "Ошибка 2D-инверсии", err)

    # ------------------------------------------------------------------
    def draw_all(self):
        self.draw_section()
        self.draw_fit()
        self.draw_curves()

    def draw_section(self):
        res = self.result
        fig = self.sec.figure
        fig.clear()
        if res is None:
            self.sec.draw()
            return
        ax = fig.add_subplot(111)
        m = res.mesh
        R = res.rho.reshape(m.shape_cells, order="F")
        x0, x1, zc, _ = res.core
        zmax = self.zplot.value() * 1000 or -zc
        if not self.zplot.value() and self.doi.isChecked() and getattr(res, "sens", None) is not None:
            dd = res.doi_depth(self.doi_thr.value() / 100)
            if dd is not None and dd.max() > 0:
                zmax = min(-zc, 1.3 * float(np.percentile(dd[dd > 0], 90)))
        ix = np.where((m.cell_centers_x >= x0) & (m.cell_centers_x <= x1))[0]
        iz = np.where((m.cell_centers_y < 0) & (m.cell_centers_y >= -zmax * 1.05))[0]
        sub = R[np.ix_(ix, iz)].T
        fin = sub[np.isfinite(sub)]
        vmin = self.vmin.value() or (np.percentile(fin, 1) if len(fin) else 1)
        vmax = self.vmax.value() or (np.percentile(fin, 99) if len(fin) else 1000)
        norm = LogNorm(vmin, max(vmax, vmin * 1.01))
        cmap = self.cmap.currentText()
        if self.smooth.isChecked():
            im = ax.pcolormesh(m.cell_centers_x[ix] / 1000, -m.cell_centers_y[iz] / 1000, sub,
                               norm=norm, cmap=cmap, shading="gouraud")
        else:
            xe = m.nodes_x[ix[0]:ix[-1] + 2] / 1000
            ze = -m.nodes_y[iz[0]:iz[-1] + 2] / 1000
            im = ax.pcolormesh(xe, ze, sub, norm=norm, cmap=cmap)
        fig.colorbar(im, ax=ax, label="ρ, Ом·м", shrink=0.9)
        if self.doi.isChecked() and getattr(res, "sens", None) is not None:
            thr = self.doi_thr.value() / 100
            Sm = res.sens.reshape(m.shape_cells, order="F")[np.ix_(ix, iz)].T
            shade = np.where(np.nan_to_num(Sm) < thr, 1.0, np.nan)
            xe = m.nodes_x[ix[0]:ix[-1] + 2] / 1000
            ze = -m.nodes_y[iz[0]:iz[-1] + 2] / 1000
            ax.pcolormesh(xe, ze, shade, cmap="Greys", vmin=0, vmax=1.4, alpha=0.75, zorder=3)
            dd = res.doi_depth(thr)
            if dd is not None:
                ax.plot(m.cell_centers_x[ix] / 1000, dd[ix] / 1000, "k--", lw=1, zorder=4,
                        label=f"глубинность (чувств. {self.doi_thr.value():g} %)")
                ax.legend(loc="lower right", fontsize=8)
        xs = res.data.x / 1000
        ax.plot(xs, np.zeros_like(xs), "kv", ms=7, clip_on=False, zorder=5)
        for x, n in zip(xs, res.data.names):
            ax.annotate(n, (x, 0), xytext=(0, 8), textcoords="offset points", fontsize=7,
                        ha="center", rotation=90, annotation_clip=False)
        ax.set_ylim(zmax / 1000, 0)
        ax.set_xlim(x0 / 1000, x1 / 1000)
        ax.set_xlabel("Расстояние по профилю, км")
        ax.set_ylabel("Глубина, км")
        if self.equal.isChecked():
            ax.set_aspect("equal")
        modes = "+".join(mo.upper() for mo in res.settings.modes)
        ax.set_title(f"2D-инверсия {modes}, простирание {res.settings.strike:.0f}°, "
                     f"RMS {res.rms:.2f}", pad=45)
        self.sec.draw()

    def draw_fit(self):
        res = self.result
        fig = self.fit.figure
        fig.clear()
        if res is None or not self.fit_mode.currentText():
            self.fit.draw()
            return
        mode = self.fit_mode.currentText().lower()
        if mode not in res.pred:
            return
        ob, pr = res.data.obs[mode], res.pred[mode]
        xs = res.data.x / 1000
        lT = np.log10(1 / res.data.freqs)

        def edges(v):
            if len(v) == 1:
                return np.array([v[0] - 0.5, v[0] + 0.5])
            mid = 0.5 * (v[1:] + v[:-1])
            return np.r_[v[0] - (mid[0] - v[0]), mid, v[-1] + (v[-1] - mid[-1])]

        xe, ye = edges(xs), edges(lT)
        axes = fig.subplots(2, 3, sharex=True, sharey=True)
        fin = ob["rho"][np.isfinite(ob["rho"])]
        norm = LogNorm(np.percentile(fin, 2), np.percentile(fin, 98))
        cmap = self.cmap.currentText()
        pcm = cmap[:-2] if cmap.endswith("_r") else cmap + "_r"
        for j, (d, t) in enumerate([(ob, "наблюд."), (pr, "модель")]):
            im = axes[0, j].pcolormesh(xe, ye, d["rho"], norm=norm, cmap=cmap)
            axes[0, j].set_title(f"ρa {mode.upper()} {t}")
            im2 = axes[1, j].pcolormesh(xe, ye, d["phs"], vmin=0, vmax=90, cmap=pcm)
            axes[1, j].set_title(f"Фаза {mode.upper()} {t}")
        fig.colorbar(im, ax=axes[0, 1], label="Ом·м")
        fig.colorbar(im2, ax=axes[1, 1], label="°")
        nr = (np.log(pr["rho"]) - np.log(ob["rho"])) / (ob["erho"] / ob["rho"])
        npf = (pr["phs"] - ob["phs"]) / ob["ephs"]
        im3 = axes[0, 2].pcolormesh(xe, ye, nr, vmin=-3, vmax=3, cmap="RdBu_r")
        axes[0, 2].set_title("Невязка ρa (в ошибках)")
        im4 = axes[1, 2].pcolormesh(xe, ye, npf, vmin=-3, vmax=3, cmap="RdBu_r")
        axes[1, 2].set_title("Невязка фазы (в ошибках)")
        fig.colorbar(im3, ax=axes[0, 2])
        fig.colorbar(im4, ax=axes[1, 2])
        axes[0, 0].invert_yaxis()
        for a in axes[:, 0]:
            a.set_ylabel("lg T, с")
        for a in axes[1]:
            a.set_xlabel("км")
        self.fit.draw()

    def draw_curves(self):
        res = self.result
        fig = self.curv.figure
        fig.clear()
        if res is None or self.cur_st.currentIndex() < 0:
            self.curv.draw()
            return
        j = self.cur_st.currentIndex()
        T = 1 / res.data.freqs
        ar, ap = fig.subplots(2, 1, sharex=True)
        cols = {"te": "tab:red", "tm": "tab:blue"}
        for mode in res.pred:
            ob, pr = res.data.obs[mode], res.pred[mode]
            c = cols[mode]
            ar.errorbar(T, ob["rho"][:, j], yerr=np.minimum(ob["erho"][:, j], 0.9 * ob["rho"][:, j]),
                        fmt="o", ms=4, color=c, label=f"{mode.upper()} набл.")
            ar.plot(T, pr["rho"][:, j], "-", color=c, label=f"{mode.upper()} модель")
            ap.errorbar(T, ob["phs"][:, j], yerr=ob["ephs"][:, j], fmt="o", ms=4, color=c)
            ap.plot(T, pr["phs"][:, j], "-", color=c)
        ar.set_xscale("log")
        ar.set_yscale("log")
        ar.set_ylabel("ρa, Ом·м")
        ar.legend(fontsize=8)
        ar.grid(True, which="both", alpha=0.3)
        ar.set_title(res.data.names[j])
        ap.set_ylabel("Фаза, °")
        ap.set_xlabel("Период, с")
        ap.grid(True, which="both", alpha=0.3)
        self.curv.draw()

    # ------------------------------------------------------------------
    def export_csv(self):
        res = self.result
        if res is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Экспорт модели", "model2d.csv",
                                                        "CSV (*.csv)")
        if not path:
            return
        m = res.mesh
        x0, x1, zc, _ = res.core
        cc = m.cell_centers
        sel = (cc[:, 0] >= x0) & (cc[:, 0] <= x1) & (cc[:, 1] < 0) & (cc[:, 1] >= zc)
        p = getattr(self, "_profile", None)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            hdr = ["x_m", "depth_m", "rho_ohmm"]
            if p is not None:
                hdr += ["lat", "lon"]
            w.writerow(hdr)
            from ..profile import from_local, to_local
            if p is not None:
                ex, ey = to_local(p.lat1, p.lon1, p.lat0, p.lon0)
                L = max(np.hypot(ex, ey), 1e-9)
            for (x, y), r in zip(cc[sel], res.rho[sel]):
                row = [f"{x:.1f}", f"{-y:.1f}", f"{r:.5g}"]
                if p is not None:
                    la, lo = from_local(x * ex / L, x * ey / L, p.lat0, p.lon0)
                    row += [f"{float(la):.6f}", f"{float(lo):.6f}"]
                w.writerow(row)
        self.main.statusBar().showMessage(f"Сохранено: {path}", 5000)

    def save_result(self):
        res = self.result
        if res is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить результат", "inv2d.npz",
                                                        "NumPy (*.npz)")
        if not path:
            return
        d = dict(hx=res.mesh.h[0], hz=res.mesh.h[1], origin=np.array(res.mesh.origin), rho=res.rho,
                 x=res.data.x, freqs=res.data.freqs, names=np.array(res.data.names),
                 core=np.array(res.core), rms=res.rms, rms_hist=np.array(res.rms_history),
                 settings=json.dumps(res.settings.to_dict()))
        if getattr(res, "sens", None) is not None:
            d["sens"] = res.sens
        for mode in res.pred:
            for k, v in res.data.obs[mode].items():
                d[f"obs_{mode}_{k}"] = v
            for k, v in res.pred[mode].items():
                d[f"pred_{mode}_{k}"] = v
        np.savez_compressed(path, **d)
        self.main.statusBar().showMessage(f"Сохранено: {path}", 5000)

    def load_result(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Открыть результат", "", "NumPy (*.npz)")
        if not path:
            return
        try:
            self.result = load_npz(path)
        except Exception as e:  # noqa: BLE001
            error_box(self, "Ошибка", str(e))
            return
        self.done(self.result)


def load_npz(path):
    import discretize

    from ..inv2d import Inv2DData, Inv2DResult

    z = np.load(path, allow_pickle=False)
    mesh = discretize.TensorMesh([z["hx"], z["hz"]], origin=z["origin"])
    s = Inv2DSettings(**{k: (tuple(v) if isinstance(v, list) else v)
                         for k, v in json.loads(str(z["settings"])).items()})
    data = Inv2DData([str(n) for n in z["names"]], z["x"], z["freqs"])
    pred = {}
    for mode in s.modes:
        if f"obs_{mode}_rho" in z:
            data.obs[mode] = {k: z[f"obs_{mode}_{k}"] for k in ("rho", "phs", "erho", "ephs")}
            pred[mode] = {k: z[f"pred_{mode}_{k}"] for k in ("rho", "phs")}
    return Inv2DResult(mesh, z["rho"], data, pred, list(z["rms_hist"]), float(z["rms"]),
                       tuple(z["core"]), s, sens=z["sens"] if "sens" in z else None)

