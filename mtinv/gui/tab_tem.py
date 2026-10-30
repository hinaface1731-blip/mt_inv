"""Вкладка «АМТЗ + ЗСБ»: загрузка ЗСБ, привязка к станциям, инверсия ЗСБ,
статический сдвиг по ЗСБ, совместная инверсия."""

from __future__ import annotations

import csv
import os

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..joint1d import SHIFT_DEPTH_FRAC, joint_inversion, mt_data, static_shift_from_model
from ..profile import to_local
from ..tem import CONFIGS, SIGNAL_UNITS, read_tem_header
from ..tsikl import RX_MODES
from .common import MplWidget, dspin, error_box, ispin, start_worker

COMPS = [("det", "Det (инвариант)"), ("xy", "XY"), ("yx", "YX"), ("ssq", "SSQ"),
         ("av", "(XY−YX)/2")]
STYLE = {"АМТЗ": ("tab:red", "--"), "ЗСБ": ("tab:green", "-."), "Совместная": ("black", "-")}


class TEMImportDialog(QtWidgets.QDialog):
    """Параметры, которых нет в шапке файлов ЗСБ."""

    def __init__(self, parent, hdr):
        super().__init__(parent)
        self.setWindowTitle("Импорт ЗСБ")
        f = QtWidgets.QFormLayout(self)
        f.addRow(QtWidgets.QLabel(
            "Формат файла: колонки «время  сигнал  [ошибка]» (пробел, табуляция или ;).\n"
            "Строки с # — комментарии; ключи вида «# TX_SIDE = 200» в шапке\n"
            "имеют приоритет над значениями ниже. См. README."))
        self.tu = QtWidgets.QComboBox()
        for k, t in [("ms", "мс"), ("us", "мкс"), ("s", "с")]:
            self.tu.addItem(t, k)
        self.du = QtWidgets.QComboBox()
        for k, t in SIGNAL_UNITS.items():
            self.du.addItem(t, k)
        self.cfg = QtWidgets.QComboBox()
        for k, t in CONFIGS.items():
            self.cfg.addItem(t, k)
        self.side = dspin(1, 5000, 100, 10, 1, " м")
        self.area = dspin(0.001, 1e7, 1.0, 10, 3, " м²")
        self.area.setToolTip("Эффективная площадь приёмника (число витков × площадь). "
                             "Для совмещённых петель — площадь петли × витки.")
        self.ramp = dspin(0, 10000, 0, 1, 1, " мкс")
        f.addRow("Единицы времени:", self.tu)
        f.addRow("Сигнал:", self.du)
        f.addRow("Установка:", self.cfg)
        f.addRow("Сторона генераторной петли:", self.side)
        f.addRow("Эфф. площадь приёмника:", self.area)
        f.addRow("Время выключения тока:", self.ramp)
        self._set(hdr)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def _set(self, h):
        for w, key in ((self.tu, "time_units"), (self.du, "data_units"), (self.cfg, "config")):
            if key in h:
                i = w.findData(h[key])
                if i >= 0:
                    w.setCurrentIndex(i)
        for w, key in ((self.side, "tx_side"), (self.area, "rx_area"), (self.ramp, "ramp_us")):
            try:
                w.setValue(float(h[key]))
            except (KeyError, ValueError):
                pass

    def values(self):
        return dict(time_units=self.tu.currentData(), data_units=self.du.currentData(),
                    config=self.cfg.currentData(), tx_side=self.side.value(),
                    rx_area=self.area.value(), ramp_us=self.ramp.value())


class PodImportDialog(QtWidgets.QDialog):
    """Параметры обработки файлов «Цикла» (*.pod)."""

    def __init__(self, parent, n_files, prev=None):
        super().__init__(parent)
        self.setWindowTitle("Импорт ЗСБ «Цикл» (*.pod)")
        f = QtWidgets.QFormLayout(self)
        f.addRow(QtWidgets.QLabel(
            f"Файлов: {n_files}. Дубли одного пикета и тока складываются (медиана),\n"
            "кривые малого и большого тока нормируются на ток и сшиваются:\n"
            "ранние времена — малый ток, поздние — большой."))
        self.units = QtWidgets.QComboBox()
        for k, t in [("mV", "мВ (формат ПОДБОР)"), ("uV", "мкВ"), ("V", "В")]:
            self.units.addItem(t, k)
        self.rx_mode = QtWidgets.QComboBox()
        for k, t in RX_MODES.items():
            self.rx_mode.addItem(t, k)
        self.turns = dspin(0.01, 1e5, 1, 1, 2)
        self.t_min = dspin(0, 10000, 50, 1, 1, " мкс")
        self.t_min.setSpecialValueText("авто (2 × задержка из файла)")
        self.t_min.setToolTip("Начальное время датчика: ПДИ-20 — 10 мкс, ПДИ-50 — 50 мкс, "
                              "ПДИ-100 — 100 мкс, ПДИ-400 — 1 мс")
        self.t_cross = dspin(0, 100000, 0, 10, 1, " мкс")
        self.t_cross.setSpecialValueText("авто")
        self.snr = dspin(0.5, 100, 2, 0.5, 1)
        self.ramp = dspin(0, 10000, 0, 1, 1, " мкс")
        self.ramp_file = QtWidgets.QCheckBox("фронт выключения из файла, отдельно для каждого тока "
                                             "(экспериментально)")
        self.ramp_file.toggled.connect(lambda on: self.ramp.setEnabled(not on))
        self.scale = QtWidgets.QCheckBox("приводить малый ток к большому (коэффициент сшивки, как «ПРОБА»)")
        self.scale.setChecked(True)
        f.addRow("Единицы ЭДС в файле:", self.units)
        f.addRow("Приёмник (Lrc):", self.rx_mode)
        f.addRow("Витков приёмника:", self.turns)
        f.addRow("Отбросить времена до:", self.t_min)
        f.addRow("Стык малый/большой ток:", self.t_cross)
        f.addRow("Мин. сигнал/шум:", self.snr)
        f.addRow("Время выключения тока:", self.ramp)
        f.addRow("", self.ramp_file)
        f.addRow("", self.scale)
        if prev:
            for w, key in ((self.units, "units"), (self.rx_mode, "rx_mode")):
                i = w.findData(prev.get(key))
                if i >= 0:
                    w.setCurrentIndex(i)
            self.turns.setValue(prev.get("rx_turns", 1.0))
            if prev.get("t_min") is not None:
                self.t_min.setValue(prev["t_min"] * 1e6)
            self.snr.setValue(prev.get("snr_min", 2.0))
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def values(self):
        return dict(units=self.units.currentData(), rx_mode=self.rx_mode.currentData(),
                    rx_turns=self.turns.value(),
                    t_min=self.t_min.value() * 1e-6 or None,
                    t_cross=self.t_cross.value() * 1e-6 or None,
                    snr_min=self.snr.value(),
                    ramp=None if self.ramp_file.isChecked() else self.ramp.value() * 1e-6,
                    scale_small=self.scale.isChecked())


class TabTEM(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.results = {}   # имя ЗСБ -> {метка: JointResult}
        self.ss_preview = None
        self.plot = MplWidget(self)

        left = QtWidgets.QVBoxLayout()
        b = QtWidgets.QPushButton("📂 Загрузить ЗСБ (файлы)…")
        b.clicked.connect(self.load_files)
        left.addWidget(b)
        b = QtWidgets.QPushButton("📁 Папка «Цикл» (*.pod)…")
        b.clicked.connect(self.load_pod_folder)
        left.addWidget(b)
        left.addWidget(QtWidgets.QLabel("Зондирования ЗСБ → МТ-станция:"))
        self.list = QtWidgets.QListWidget()
        self.list.currentRowChanged.connect(self._on_current)
        left.addWidget(self.list, 1)
        lr = QtWidgets.QHBoxLayout()
        lr.addWidget(QtWidgets.QLabel("Привязка:"))
        self.link = QtWidgets.QComboBox()
        self.link.activated.connect(self._set_link)
        lr.addWidget(self.link, 1)
        left.addLayout(lr)
        ar = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Авто-привязка ≤")
        b.clicked.connect(self.auto_link)
        ar.addWidget(b)
        self.link_dist = dspin(1, 100000, 1000, 100, 0, " м")
        ar.addWidget(self.link_dist)
        left.addLayout(ar)

        box = QtWidgets.QGroupBox("Параметры")
        f = QtWidgets.QFormLayout(box)
        self.comp = QtWidgets.QComboBox()
        for k, t in COMPS:
            self.comp.addItem(t, k)
        f.addRow("Компонента АМТЗ:", self.comp)
        self.nlay = ispin(10, 120, 45)
        f.addRow("Число слоёв:", self.nlay)
        self.ftem = dspin(0.5, 100, 10, 1, 1, " %")
        f.addRow("Мин. ошибка ЗСБ:", self.ftem)
        self.frho = dspin(1, 100, 10, 1, 0, " %")
        f.addRow("Мин. ошибка ρa:", self.frho)
        self.fphs = dspin(0.1, 20, 2.9, 0.5, 1, "°")
        f.addRow("Мин. ошибка фазы:", self.fphs)
        self.target = dspin(0.3, 10, 1.0, 0.1, 2)
        f.addRow("Целевой RMS:", self.target)
        left.addWidget(box)

        b1 = QtWidgets.QPushButton("1. Инверсия ЗСБ")
        b1.clicked.connect(lambda: self.run(["ЗСБ"]))
        left.addWidget(b1)

        sbox = QtWidgets.QGroupBox("2. Статический сдвиг АМТЗ по ЗСБ")
        sl = QtWidgets.QVBoxLayout(sbox)
        b = QtWidgets.QPushButton("Рассчитать для привязанной станции")
        b.clicked.connect(self.calc_shift)
        sl.addWidget(b)
        self.apply_btn = QtWidgets.QPushButton("Применить")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.apply_shift)
        sl.addWidget(self.apply_btn)
        b = QtWidgets.QPushButton("Рассчитать и применить для всех привязанных")
        b.clicked.connect(self.shift_all)
        sl.addWidget(b)
        b = QtWidgets.QPushButton("Сбросить сдвиг станции")
        b.clicked.connect(self.reset_shift)
        sl.addWidget(b)
        left.addWidget(sbox)

        jbox = QtWidgets.QGroupBox("3. Совместная инверсия АМТЗ + ЗСБ")
        jl = QtWidgets.QVBoxLayout(jbox)
        self.inv_ss = QtWidgets.QCheckBox("подбирать статический сдвиг АМТЗ")
        self.inv_ss.setChecked(True)
        jl.addWidget(self.inv_ss)
        b = QtWidgets.QPushButton("▶ Совместная инверсия")
        b.clicked.connect(lambda: self.run(["Совместная"]))
        jl.addWidget(b)
        b = QtWidgets.QPushButton("▶ Сравнить: АМТЗ / ЗСБ / совместная")
        b.setToolTip("Три инверсии на одной сетке — видно, как совместная инверсия "
                     "снимает эквивалентность")
        b.clicked.connect(lambda: self.run(["АМТЗ", "ЗСБ", "Совместная"]))
        jl.addWidget(b)
        left.addWidget(jbox)
        b = QtWidgets.QPushButton("Экспорт моделей (CSV)…")
        b.clicked.connect(self.export)
        left.addWidget(b)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        left.addWidget(self.info)

        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(lw)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(380)
        scroll.setMinimumWidth(330)
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(scroll)
        lay.addWidget(self.plot, 1)

    # ------------------------------------------------------------------
    def current(self):
        it = self.list.currentItem()
        return self.main.tems.get(it.data(QtCore.Qt.UserRole)) if it else None

    def station_for(self, snd):
        return self.main.stations.get(snd.link) if snd and snd.link else None

    def refresh(self):
        cur = self.current()
        self.list.blockSignals(True)
        self.list.clear()
        for n, s in self.main.tems.items():
            it = QtWidgets.QListWidgetItem(f"{n} → {s.link or '—'}")
            it.setData(QtCore.Qt.UserRole, n)
            self.list.addItem(it)
        self.list.blockSignals(False)
        names = list(self.main.tems)
        if names:
            self.list.setCurrentRow(names.index(cur.name) if cur and cur.name in names else 0)
        else:
            self._on_current(-1)

    def refresh_stations(self):
        self.link.blockSignals(True)
        self.link.clear()
        self.link.addItem("—", "")
        for n in self.main.stations:
            self.link.addItem(n, n)
        self.link.blockSignals(False)
        self._sync_link_box()

    def _sync_link_box(self):
        snd = self.current()
        i = self.link.findData(snd.link if snd else "")
        self.link.blockSignals(True)
        self.link.setCurrentIndex(max(i, 0))
        self.link.blockSignals(False)

    def _on_current(self, _row):
        self._sync_link_box()
        self.ss_preview = None
        self.apply_btn.setEnabled(False)
        self.redraw()

    def _set_link(self):
        snd = self.current()
        if snd is None:
            return
        snd.link = self.link.currentData() or ""
        self.refresh()
        self.main.map.redraw(keep_view=True)

    def auto_link(self):
        n = auto_link(self.main.tems, self.main.stations, self.link_dist.value())
        self.refresh()
        self.main.map.redraw(keep_view=True)
        self.main.statusBar().showMessage(f"Привязано зондирований ЗСБ: {n}", 6000)

    def load_files(self):
        start = self.main.settings.value("last_dir_tem", self.main.settings.value("last_dir", ""))
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Открыть данные ЗСБ", start,
            "ЗСБ (*.pod *.POD *.txt *.dat *.csv *.tem *.zsb);;«Цикл» (*.pod *.POD);;Все файлы (*)")
        if not paths:
            return
        self.main.settings.setValue("last_dir_tem", os.path.dirname(paths[0]))
        pods = [p for p in paths if p.lower().endswith(".pod")]
        if pods:
            self._import_pods(pods)
            paths = [p for p in paths if p not in pods]
            if not paths:
                return
        dlg = TEMImportDialog(self, read_tem_header(paths[0]))
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        self.main.load_tem_paths(paths, dlg.values())

    def load_pod_folder(self):
        start = self.main.settings.value("last_dir_tem", self.main.settings.value("last_dir", ""))
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Папка с файлами «Цикла»", start)
        if not d:
            return
        self.main.settings.setValue("last_dir_tem", d)
        import glob

        paths = sorted(set(glob.glob(os.path.join(d, "**", "*.pod"), recursive=True)
                           + glob.glob(os.path.join(d, "**", "*.POD"), recursive=True)))
        if not paths:
            QtWidgets.QMessageBox.information(self, "ЗСБ", "В папке нет файлов *.pod")
            return
        self._import_pods(paths)

    def _import_pods(self, paths):
        import json

        try:
            prev = json.loads(self.main.settings.value("pod_params", "") or "null")
        except (TypeError, ValueError):
            prev = None
        dlg = PodImportDialog(self, len(paths), prev if isinstance(prev, dict) else None)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        prm = dlg.values()
        self.main.settings.setValue("pod_params", json.dumps(prm))
        self.main.load_pod_paths(paths, prm)

    # ------------------------------------------------------------------
    def _params(self):
        return dict(n_layers=self.nlay.value(), floor_rho=self.frho.value() / 100,
                    floor_phs=self.fphs.value(), floor_tem=self.ftem.value() / 100,
                    target_rms=self.target.value())

    def run(self, kinds):
        snd = self.current()
        if snd is None:
            QtWidgets.QMessageBox.information(self, "ЗСБ", "Загрузите и выберите зондирование ЗСБ.")
            return
        st = self.station_for(snd)
        if any(k != "ЗСБ" for k in kinds) and st is None:
            QtWidgets.QMessageBox.information(self, "ЗСБ", "Привяжите зондирование к МТ-станции.")
            return
        prm = self._params()
        comp = self.comp.currentData()
        inv_ss = self.inv_ss.isChecked()
        mt = mt_data(st, comp) if st is not None else None
        # общая сетка слоёв для сравнения
        if len(kinds) > 1:
            probe = joint_inversion(mt=mt, tem=snd, max_iter=0, **prm)
            prm.update(z_first=probe.tops[1], z_max=probe.tops[-1])

        def job(progress, log, stop):
            out = {}
            for k in kinds:
                if stop():
                    break
                progress(f"{k}…")
                if k == "ЗСБ":
                    out[k] = joint_inversion(tem=snd, label=k, **prm)
                elif k == "АМТЗ":
                    out[k] = joint_inversion(mt=mt, label=k, **prm)
                else:
                    out[k] = joint_inversion(mt=mt, tem=snd, invert_ss=inv_ss, label=k, **prm)
            return out

        self.info.setText("Считаю…")
        name = snd.name

        def done(out):
            self.results.setdefault(name, {}).update(out)
            txt = []
            for k, r in out.items():
                s = f"{k}: RMS {r.rms:.2f}"
                if r.ss is not None:
                    s += f", поправка ρa k = {r.ss:.2f}"
                txt.append(s)
            self.info.setText("\n".join(txt))
            self.redraw()

        start_worker(self, job, done, lambda e: error_box(self, "Ошибка инверсии", e),
                     on_progress=lambda m: self.info.setText(f"Считаю: {m}"))

    # ------------------------------------------------------------------
    def _tem_model(self, snd):
        r = self.results.get(snd.name, {}).get("ЗСБ")
        if r is None:
            r = joint_inversion(tem=snd, label="ЗСБ", **self._params())
            self.results.setdefault(snd.name, {})["ЗСБ"] = r
        return r

    def calc_shift(self):
        snd = self.current()
        st = self.station_for(snd)
        if st is None:
            QtWidgets.QMessageBox.information(self, "ЗСБ", "Привяжите зондирование к МТ-станции.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            r = self._tem_model(snd)
            doi = snd.depth_of_investigation()
            out = {c: static_shift_from_model(st, r.tops, r.rho, c,
                                              max_depth=SHIFT_DEPTH_FRAC * doi)
                   for c in ("xy", "yx")}
        except Exception as e:  # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            error_box(self, "Ошибка", str(e))
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.ss_preview = (st.name, out, doi)
        self.apply_btn.setEnabled(True)
        warn = "" if all(o["overlap"] for o in out.values()) else \
            "\n⚠ Нет общих частот: АМТЗ «видит» глубже ЗСБ, взяты 3 самые высокие частоты."
        if any(o["spread"] >= 0.25 for o in out.values()):
            warn += ("\n⚠ Формы кривых АМТЗ и модели ЗСБ на общих частотах не совпадают "
                     "(разброс отношения > 25 %): сдвиг ненадёжен — возможны 3D-искажения "
                     "или неточная модель ЗСБ.")
        self.info.setText(
            f"{st.name}: глубинность ЗСБ ≈ {doi:.0f} м\n"
            f"поправка ρxy × {out['xy']['k']:.3f} (по {int(out['xy']['use'].sum())} частотам, "
            f"разброс {100 * out['xy']['spread']:.0f} %)\n"
            f"поправка ρyx × {out['yx']['k']:.3f} (по {int(out['yx']['use'].sum())} частотам, "
            f"разброс {100 * out['yx']['spread']:.0f} %)"
            f"{warn}\nНажмите «Применить».")
        self.redraw()

    def apply_shift(self):
        if not self.ss_preview:
            return
        name, out, _ = self.ss_preview
        st = self.main.stations.get(name)
        if st is None:
            return
        st.ss = np.array([out["xy"]["k"], out["yx"]["k"]])
        self.ss_preview = None
        self.apply_btn.setEnabled(False)
        self.main.station_data_changed(name)
        self.info.setText(f"{name}: сдвиг применён (ρxy × {st.ss[0]:.3f}, ρyx × {st.ss[1]:.3f}). "
                          "Он учитывается в кривых, 1D и 2D инверсии.")
        self.redraw()

    def reset_shift(self):
        st = self.station_for(self.current())
        if st is not None:
            st.ss = np.ones(2)
            self.main.station_data_changed(st.name)
            self.redraw()

    def shift_all(self):
        # центральные зондирования обрабатываются последними — их сдвиг и остаётся
        pairs = sorted([(s, self.main.stations[s.link]) for s in self.main.tems.values()
                        if s.link in self.main.stations], key=lambda p: p[0].is_central)
        if not pairs:
            QtWidgets.QMessageBox.information(self, "ЗСБ", "Нет привязанных зондирований.")
            return
        prm = self._params()
        cached = {n: r.get("ЗСБ") for n, r in self.results.items()}

        def job(progress, log, stop):
            out = []
            for i, (snd, st) in enumerate(pairs):
                if stop():
                    break
                progress(f"{snd.name} ({i + 1}/{len(pairs)})")
                r = cached.get(snd.name) or joint_inversion(tem=snd, label="ЗСБ", **prm)
                doi = snd.depth_of_investigation()
                k = [static_shift_from_model(st, r.tops, r.rho, c, max_depth=SHIFT_DEPTH_FRAC * doi)
                     for c in ("xy", "yx")]
                out.append((snd.name, st.name, r, k))
            return out

        def done(out):
            lines = []
            for sname, stname, r, k in out:
                self.results.setdefault(sname, {})["ЗСБ"] = r
                st = self.main.stations[stname]
                st.ss = np.array([k[0]["k"], k[1]["k"]])
                flag = "" if k[0]["reliable"] and k[1]["reliable"] else " ⚠"
                lines.append(f"{stname}: ×{st.ss[0]:.2f} / ×{st.ss[1]:.2f}{flag}")
                self.main.station_data_changed(stname)
            self.info.setText("Сдвиг применён (ρxy / ρyx):\n" + "\n".join(lines) +
                              ("\n⚠ — сдвиг ненадёжен: нет общих частот или формы кривых не совпадают"
                               if any("⚠" in s for s in lines) else ""))
            self.redraw()

        self.info.setText("Считаю…")
        start_worker(self, job, done, lambda e: error_box(self, "Ошибка", e),
                     on_progress=lambda m: self.info.setText(f"Инверсия ЗСБ: {m}"))

    # ------------------------------------------------------------------
    def redraw(self):
        fig = self.plot.figure
        fig.clear()
        snd = self.current()
        if snd is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Загрузите данные ЗСБ (кнопка слева)", ha="center",
                    transform=ax.transAxes)
            self.plot.draw()
            return
        st = self.station_for(snd)
        res = self.results.get(snd.name, {})
        gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 1])
        a_sig = fig.add_subplot(gs[0, 0])
        a_rt = fig.add_subplot(gs[1, 0], sharex=a_sig)
        a_rho = fig.add_subplot(gs[0, 1])
        a_ph = fig.add_subplot(gs[1, 1], sharex=a_rho)
        a_m = fig.add_subplot(gs[:, 2])

        ok = ~snd.mask
        t = snd.times * 1e3
        v = snd.signal
        e = np.where(np.isfinite(snd.err), snd.err, 0)
        # исходные кривые малого/большого тока («Цикл»)
        for key, col, lab in (("small", "tab:blue", "малый ток"), ("big", "tab:red", "большой ток")):
            c = snd.raw.get(key) if snd.raw else None
            if c is not None:
                a_sig.plot(c[0] * 1e3, np.abs(c[1]), "-", color=col, lw=0.6, alpha=0.45, label=lab)
        pos, neg = ok & (v > 0), ok & (v < 0)
        a_sig.errorbar(t[pos], v[pos], yerr=np.minimum(e[pos], 0.9 * v[pos]), fmt="s", ms=3,
                       color="tab:green", label="ЗСБ набл. (>0)")
        if neg.any():
            a_sig.errorbar(t[neg], -v[neg], yerr=np.minimum(e[neg], -0.9 * v[neg]), fmt="s", ms=3,
                           mfc="white", color="tab:green", label="ЗСБ набл. (<0)")
        a_sig.plot(t[~ok], np.abs(v[~ok]), "x", ms=3, color="gray", alpha=0.6, label="исключено")
        tc = (snd.raw or {}).get("info", {}).get("t_cross")
        if tc:
            a_sig.axvline(tc * 1e3, color="gray", ls=":", lw=0.8)
        central = snd.is_central   # ρτ «ПРОБЫ»: размер приёмной петли не учитывается
        if central:
            a_rt.plot(t[pos], snd.rho_tau()[pos], "s", ms=3, color="tab:green", label="ρτ набл.")
        for lab, r in res.items():
            if r.tem is None:
                continue
            c, ls = STYLE.get(lab, ("gray", "-"))
            p = r.tem["pred"]
            a_sig.plot(r.tem["times"] * 1e3, np.where(p > 0, p, np.nan), ls, color=c,
                       label=f"{lab} (RMS ЗСБ {r.rms_tem:.2f})")
            if (p < 0).any():
                a_sig.plot(r.tem["times"] * 1e3, np.where(p < 0, -p, np.nan), ls, color=c, lw=0.8,
                           alpha=0.6)
            if central:
                a_rt.plot(r.tem["times"] * 1e3, r.tem["rho_tau_pred"], ls, color=c)
            else:
                res_n = (r.tem["obs"] - p) / np.maximum(np.where(np.isfinite(r.tem["err"]),
                                                                 r.tem["err"], 0),
                                                        self.ftem.value() / 100 * np.abs(r.tem["obs"]))
                a_rt.plot(r.tem["times"] * 1e3, res_n, "o" + ls[0], color=c, ms=3, lw=0.8)
        a_sig.set_xscale("log")
        a_sig.set_yscale("log")
        a_sig.set_ylabel("|dBz/dt|/I, Тл/(с·А)")
        a_sig.set_title(f"ЗСБ {snd.name}: {snd.geometry_text}", fontsize=9)
        a_sig.legend(fontsize=6)
        a_sig.grid(True, which="both", alpha=0.3)
        a_rt.set_xlabel("Время, мс")
        if central:
            a_rt.set_yscale("log")
            a_rt.set_ylabel("ρτ, Ом·м")
        else:
            a_rt.axhspan(-1, 1, color="green", alpha=0.08)
            a_rt.set_ylabel("Невязка, в ошибках")
            a_rt.set_title("ρτ для этой установки не считается", fontsize=8, color="gray")
        a_rt.grid(True, which="both", alpha=0.3)

        if st is not None:
            comp = self.comp.currentData()
            T = st.period
            m = ~st.mask
            rho_c, phs, _, _ = st.rho_phase(comp)
            if st.has_static_shift:
                raw = type(st)(st.name, st.lat, st.lon, st.elev, st.freq, st.Z, st.Zerr, mask=st.mask)
                rho_raw, _, _, _ = raw.rho_phase(comp)
                a_rho.plot(T[m], rho_raw[m], "o", ms=4, mfc="none", color="gray", label="ρa исходная")
                a_rho.plot(T[m], rho_c[m], "o", ms=4, color="tab:red", label="ρa исправленная")
            else:
                a_rho.plot(T[m], rho_c[m], "o", ms=4, color="tab:red", label=f"ρa {comp}")
            a_ph.plot(T[m], phs[m], "o", ms=4, color="tab:blue")
            if self.ss_preview and self.ss_preview[0] == st.name:
                _, out, doi = self.ss_preview
                for c, col in (("xy", "tab:orange"), ("yx", "tab:purple")):
                    o = out[c]
                    a_rho.plot(T, o["rho_model"], "-", color=col, lw=1,
                               label=f"модель ЗСБ → МТ ({c}: ×{o['k']:.2f})")
                    a_rho.plot(T[o["use"]], o["rho_model"][o["use"]], "|", color=col, ms=14)
            for lab, r in res.items():
                if r.mt is None:
                    continue
                cc, ls = STYLE.get(lab, ("gray", "-"))
                a_rho.plot(1 / r.mt["freq"], r.mt["rho_pred"], ls, color=cc, label=f"{lab} (RMS АМТЗ {r.rms_mt:.2f})")
                a_ph.plot(1 / r.mt["freq"], r.mt["phs_pred"], ls, color=cc)
                if r.ss is not None:
                    a_rho.plot(1 / r.mt["freq"], r.mt["rho_true_pred"], ":", color=cc,
                               label=f"{lab}: без сдвига (k={r.ss:.2f})")
            a_rho.set_xscale("log")
            a_rho.set_yscale("log")
            a_rho.set_ylabel("ρa, Ом·м")
            a_rho.set_title(f"АМТЗ {st.name} ({comp})" + (f", сдвиг ×{st.ss[0]:.2f}/×{st.ss[1]:.2f}"
                                                          if st.has_static_shift else ""), fontsize=10)
            a_rho.legend(fontsize=7)
            a_ph.set_ylim(0, 90)
        else:
            a_rho.text(0.5, 0.5, "нет привязанной МТ-станции", ha="center", transform=a_rho.transAxes,
                       color="gray")
        for a in (a_rho, a_ph):
            a.grid(True, which="both", alpha=0.3)
        a_ph.set_xlabel("Период, с")
        a_ph.set_ylabel("Фаза, °")

        zmax, zmin = 0, 1e9
        for lab, r in res.items():
            c, ls = STYLE.get(lab, ("gray", "-"))
            z = np.r_[r.tops, r.tops[-1] * 1.5]
            z[0] = max(r.tops[1] / 2, 0.5)
            a_m.step(np.r_[r.rho, r.rho[-1]], z, where="post", color=c, ls=ls, lw=2, label=lab)
            zmax = max(zmax, r.tops[-1])
            zmin = min(zmin, z[0])
        doi = snd.depth_of_investigation()
        a_m.axhline(doi, color="tab:green", lw=0.8, ls=":")
        a_m.annotate("глубинность ЗСБ", (1, doi), xycoords=("axes fraction", "data"), ha="right",
                     va="bottom", fontsize=7, color="tab:green")
        a_m.set_xscale("log")
        a_m.set_yscale("log")
        if zmax:
            a_m.set_ylim(zmax, zmin)
        else:
            a_m.set_ylim(doi * 10, 1)
        a_m.set_xlabel("ρ, Ом·м")
        a_m.set_ylabel("Глубина, м")
        a_m.set_title("Модели", fontsize=10)
        a_m.grid(True, which="both", alpha=0.3)
        if res:
            a_m.legend(fontsize=8)
        self.plot.draw()

    def export(self):
        if not self.results:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Экспорт", "models_tem_mt.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["tem", "station", "inversion", "top_depth_m", "rho_ohmm", "rms", "ss_k"])
            for n, rr in self.results.items():
                snd = self.main.tems.get(n)
                for lab, r in rr.items():
                    for z, rho in zip(r.tops, r.rho):
                        w.writerow([n, snd.link if snd else "", lab, f"{z:.1f}", f"{rho:.4g}",
                                    f"{r.rms:.3f}", "" if r.ss is None else f"{r.ss:.4f}"])
        self.main.statusBar().showMessage(f"Сохранено: {path}", 5000)


def auto_link(tems, stations, max_dist):
    """Привязка ЗСБ к ближайшей МТ-станции (или к станции с тем же именем)."""
    n = 0
    for snd in tems.values():
        if snd.link in stations:
            n += 1
            continue
        if snd.name in stations:
            snd.link = snd.name
            n += 1
            continue
        if not (np.isfinite(snd.lat) and np.isfinite(snd.lon)):
            continue
        best, dmin = "", 1e300
        for st in stations.values():
            if not (np.isfinite(st.lat) and np.isfinite(st.lon)):
                continue
            x, y = to_local(st.lat, st.lon, snd.lat, snd.lon)
            d = float(np.hypot(x, y))
            if d < dmin:
                best, dmin = st.name, d
        if best and dmin <= max_dist:
            snd.link = best
            n += 1
    return n

