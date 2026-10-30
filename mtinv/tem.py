"""ЗСБ (TEM): данные зондирования, чтение файлов, 1D прямая задача, ρτ.

Внутреннее представление сигнала — |dBz/dt| / I  [Тл/(с·А)] = [В/(м²·А)].
Прямая задача — SimPEG Simulation1DLayered (горизонтальная петля-генератор,
круглая петля эквивалентной площади, ступенчатое или линейное выключение тока).

Установки:
  central    — приёмная рамка (или датчик) в центре генераторной петли;
  coincident — совмещённые петли (ЭДС нормируется на площадь петли).
               SimPEG считает только центральную установку, поэтому для
               совмещённых петель используется она же: на поздней стадии
               нормированные отклики совпадают, на ранних временах
               (t < ~μ0σa²/10) это приближение.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

MU0 = 4e-7 * np.pi

# Единицы сигнала во входных файлах
SIGNAL_UNITS = {
    "emf_uV_A": "ЭДС/I, мкВ/А (нужна эффективная площадь приёмника)",
    "emf_V_A": "ЭДС/I, В/А (нужна эффективная площадь приёмника)",
    "dbdt_nT_s_A": "dB/dt / I, нТл/(с·А)",
    "dbdt_T_s_A": "dB/dt / I, Тл/(с·А)",
    "rho_tau": "ρτ, Ом·м (поздняя стадия)",
}
TIME_UNITS = {"ms": 1e-3, "us": 1e-6, "s": 1.0}
CONFIGS = {"central": "центральная (приёмник в центре петли)",
           "coincident": "совмещённые петли (приближённо)"}


@dataclass
class TEMSounding:
    name: str
    lat: float
    lon: float
    times: np.ndarray        # с, после выключения тока
    signal: np.ndarray       # |dBz/dt|/I, Тл/(с·А)
    err: np.ndarray          # стд. ошибка в тех же единицах (nan = нет)
    tx_side: float = 100.0   # сторона квадратной генераторной петли, м
    config: str = "central"
    ramp: float = 0.0        # длительность линейного выключения тока, с
    elev: float = 0.0
    path: str = ""
    link: str = ""           # имя привязанной МТ-станции
    mask: np.ndarray = field(default=None)
    opts: dict = field(default_factory=dict)  # параметры импорта (для проекта)

    def __post_init__(self):
        self.times = np.asarray(self.times, float)
        self.signal = np.abs(np.asarray(self.signal, float))
        self.err = np.asarray(self.err, float)
        o = np.argsort(self.times)
        self.times, self.signal, self.err = self.times[o], self.signal[o], self.err[o]
        if self.mask is None:
            self.mask = np.zeros(len(self.times), bool)
        else:
            self.mask = np.asarray(self.mask, bool)[o]

    @property
    def tx_radius(self):
        """Радиус круглой петли той же площади."""
        return self.tx_side / np.sqrt(np.pi)

    def rho_tau(self, signal=None):
        return rho_tau(self.times, self.signal if signal is None else signal, self.tx_radius)

    def depth_of_investigation(self, rho=None):
        """Глубина диффузии на самом позднем времени (оценка), м."""
        rt = self.rho_tau()
        rho = float(np.nanmedian(rt[~self.mask])) if rho is None else rho
        return float(np.sqrt(2 * self.times[~self.mask].max() * rho / MU0))


def rho_tau(times, signal, radius):
    """Кажущееся сопротивление поздней стадии (центральная установка):
    |dBz/dt|/I = a² σ^{3/2} μ0^{5/2} / (20 √π t^{5/2})."""
    t = np.asarray(times, float)
    s = np.abs(np.asarray(signal, float))
    sig32 = 20 * np.sqrt(np.pi) * t ** 2.5 * s / (radius ** 2 * MU0 ** 2.5)
    return 1.0 / np.maximum(sig32, 1e-300) ** (2.0 / 3.0)


def signal_from_rho_tau(times, rho, radius):
    t = np.asarray(times, float)
    sig = 1.0 / np.asarray(rho, float)
    return radius ** 2 * sig ** 1.5 * MU0 ** 2.5 / (20 * np.sqrt(np.pi) * t ** 2.5)


# ----------------------------------------------------------------------
class TEMForward:
    """Прямая задача 1D для заданной геометрии и времён (с кэшем симуляции)."""

    def __init__(self, times, tx_side, config="central", ramp=0.0):
        import simpeg.electromagnetics.time_domain as tdem

        self.times = np.asarray(times, float)
        self.radius = tx_side / np.sqrt(np.pi)
        self.config = config
        self.radii = np.array([0.0])
        self.weights = np.array([1.0])
        self.ramp = float(ramp)
        if ramp > 0:
            wave = tdem.sources.RampOffWaveform(off_time=ramp)
        else:
            wave = tdem.sources.StepOffWaveform()
        t_abs = self.times + self.ramp
        rxs = [tdem.receivers.PointMagneticFluxTimeDerivative(
            np.array([[r, 0.0, 0.0]]), t_abs, "z") for r in self.radii]
        src = tdem.sources.CircularLoop(rxs, location=np.array([0.0, 0.0, 0.0]),
                                        radius=self.radius, current=1.0, waveform=wave)
        self._tdem = tdem
        self.survey = tdem.Survey([src])

    def __call__(self, rho, thick):
        from simpeg import maps

        rho = np.asarray(rho, float)
        sim = self._tdem.Simulation1DLayered(
            survey=self.survey, thicknesses=np.asarray(thick, float),
            sigmaMap=maps.IdentityMap(nP=len(rho)))
        d = sim.dpred(1.0 / rho).reshape(len(self.radii), len(self.times))
        return np.abs(self.weights @ d)


# ----------------------------------------------------------------------
_KEYS = {
    "NAME": "name", "LAT": "lat", "LON": "lon", "LONG": "lon", "ELEV": "elev",
    "TX_SIDE": "tx_side", "RX_AREA": "rx_area", "CONFIG": "config",
    "TIME_UNITS": "time_units", "DATA": "data_units", "UNITS": "data_units",
    "RAMP_US": "ramp_us", "LINK": "link",
}


def read_tem_header(path):
    """Ключи из строк-комментариев вида  '# KEY = value'."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] not in "#;!/":
                continue
            m = re.match(r"^[#;!/]+\s*([A-Za-z_]+)\s*[=:]\s*([^#(]*)", s)
            if m and m.group(1).upper() in _KEYS:
                out[_KEYS[m.group(1).upper()]] = m.group(2).strip()
    return out


def _read_columns(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#;!/":
                continue
            if ";" in s:
                parts = s.replace(",", ".").split(";")
            elif re.search(r"\s", s):
                parts = re.split(r"\s+", s.replace(",", "."))
            else:
                parts = s.split(",")
            parts = [p.strip() for p in parts]
            try:
                vals = [float(p.replace("D", "E")) for p in parts if p != ""]
            except ValueError:
                continue  # строка заголовка таблицы
            if len(vals) >= 2:
                rows.append(vals[:3] if len(vals) >= 3 else vals[:2] + [np.nan])
    if len(rows) < 3:
        raise ValueError("в файле меньше 3 строк с числами (время, сигнал[, ошибка])")
    return np.array(rows, float)


def read_tem(path, defaults=None) -> TEMSounding:
    """Читает текстовый файл ЗСБ: колонки время, сигнал[, ошибка].

    Параметры (единицы, петля, координаты) берутся из шапки файла, а
    недостающие — из defaults (диалог импорта).
    """
    opts = dict(name=os.path.splitext(os.path.basename(path))[0], lat="nan", lon="nan",
                elev="0", tx_side="100", rx_area="1", config="central", time_units="ms",
                data_units="emf_uV_A", ramp_us="0", link="")
    if defaults:
        opts.update({k: str(v) for k, v in defaults.items() if v is not None})
    opts.update(read_tem_header(path))
    a = _read_columns(path)
    tu = TIME_UNITS.get(opts["time_units"].lower().replace("µ", "u").replace("мкс", "us")
                        .replace("мс", "ms").replace("с", "s"))
    if tu is None:
        raise ValueError(f"неизвестные единицы времени: {opts['time_units']}")
    t = a[:, 0] * tu
    v = a[:, 1]
    e = a[:, 2]
    units = opts["data_units"]
    side = float(opts["tx_side"])
    radius = side / np.sqrt(np.pi)
    rx_area = float(opts["rx_area"])
    if units == "emf_uV_A":
        sig, err = v * 1e-6 / rx_area, e * 1e-6 / rx_area
    elif units == "emf_V_A":
        sig, err = v / rx_area, e / rx_area
    elif units == "dbdt_nT_s_A":
        sig, err = v * 1e-9, e * 1e-9
    elif units == "dbdt_T_s_A":
        sig, err = v, e
    elif units == "rho_tau":
        sig = signal_from_rho_tau(t, v, radius)
        err = 1.5 * sig * np.abs(e / v)  # δS/S = 3/2 δρ/ρ
    else:
        raise ValueError(f"неизвестные единицы сигнала: {units}")

    def num(x, d=np.nan):
        try:
            return float(x)
        except ValueError:
            from .edi import _parse_angle
            v_ = _parse_angle(x)
            return d if v_ is None else v_

    return TEMSounding(name=opts["name"], lat=num(opts["lat"]), lon=num(opts["lon"]),
                       times=t, signal=sig, err=err, tx_side=side, config=opts["config"],
                       ramp=float(opts["ramp_us"]) * 1e-6, elev=num(opts["elev"], 0.0),
                       path=path, link=opts["link"], opts=opts)


def write_tem(snd: TEMSounding, path, rx_area=1.0):
    """Запись в формате, который понимает read_tem (сигнал в мкВ/А)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# NAME = {snd.name}\n# LAT = {snd.lat:.6f}\n# LON = {snd.lon:.6f}\n"
                f"# ELEV = {snd.elev:.1f}\n# TX_SIDE = {snd.tx_side:g}\n"
                f"# RX_AREA = {rx_area:g}\n# CONFIG = {snd.config}\n# TIME_UNITS = ms\n"
                f"# DATA = emf_uV_A\n# RAMP_US = {snd.ramp * 1e6:g}\n")
        if snd.link:
            f.write(f"# LINK = {snd.link}\n")
        f.write("# time_ms  emf_uV_A  err_uV_A\n")
        for t, s, e in zip(snd.times, snd.signal, snd.err):
            f.write(f"{t * 1e3:.6g}  {s * rx_area * 1e6:.6e}  {e * rx_area * 1e6:.6e}\n")
