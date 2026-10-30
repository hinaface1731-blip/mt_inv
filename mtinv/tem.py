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
    signal: np.ndarray       # dBz/dt / I, Тл/(с·А); со знаком (вне петли бывает < 0)
    err: np.ndarray          # стд. ошибка в тех же единицах (nan = нет)
    tx_side: float = 100.0   # сторона генераторной петли (X), м
    config: str = "central"
    ramp: float = 0.0        # длительность линейного выключения тока, с
    elev: float = 0.0
    path: str = ""
    link: str = ""           # имя привязанной МТ-станции
    mask: np.ndarray = field(default=None)
    opts: dict = field(default_factory=dict)  # параметры импорта (для проекта)
    tx_b: float = 0.0        # вторая сторона петли (0 = квадрат)
    rx_side: float = 0.0     # сторона приёмной рамки (0 = точечный датчик)
    offset: float = 0.0      # разнос: центр приёмника от центра петли, м
    offset_angle: float = 0.0  # направление разноса от оси X петли, град
    raw: dict = field(default_factory=dict)  # исходные кривые (для показа)
    # длительность фронта выключения для каждого отсчёта (с); если задано,
    # время отсчёта считается от середины фронта (как в аппаратуре «Цикл»)
    ramps: np.ndarray | None = None

    def __post_init__(self):
        self.times = np.asarray(self.times, float)
        self.signal = np.asarray(self.signal, float)
        self.err = np.asarray(self.err, float)
        o = np.argsort(self.times)
        self.times, self.signal, self.err = self.times[o], self.signal[o], self.err[o]
        if self.mask is None:
            self.mask = np.zeros(len(self.times), bool)
        else:
            self.mask = np.asarray(self.mask, bool)[o]
        if self.ramps is not None:
            self.ramps = np.asarray(self.ramps, float)[o]

    @property
    def tx_radius(self):
        """Радиус круглой петли той же площади."""
        return np.sqrt(self.tx_side * (self.tx_b or self.tx_side) / np.pi)

    @property
    def is_central(self):
        return abs(self.offset) < 1e-6

    @property
    def geometry_text(self):
        tx = f"{self.tx_side:g}×{self.tx_b or self.tx_side:g} м"
        rx = f", рамка {self.rx_side:g} м" if self.rx_side else ""
        pos = "центр" if self.is_central else f"разнос {self.offset:g} м"
        return f"петля {tx}{rx}, {pos}"

    def rho_tau(self, signal=None):
        return rho_tau(self.times, self.signal if signal is None else signal, self.tx_radius)

    def halfspace_rho(self):
        """ρ однородного полупространства, лучше всего описывающего кривую
        (для любой геометрии, в т.ч. разноса). Кэшируется."""
        key = (tuple(self.mask), self.tx_side, self.tx_b, self.rx_side, self.offset, self.ramp,
               None if self.ramps is None else tuple(self.ramps))
        if getattr(self, "_hs", (None,))[0] == key:
            return self._hs[1]
        ok = ~self.mask & np.isfinite(self.signal) & (self.signal != 0)
        if ok.sum() < 2:
            return 100.0
        fwd = make_forward(self, self.times[ok])
        obs = self.signal[ok]
        s0 = np.median(np.abs(obs)) * 1e-3

        def misfit(lr):
            p = fwd([10 ** lr], [])
            return float(np.sum((np.arcsinh(p / s0) - np.arcsinh(obs / s0)) ** 2))

        grid = np.linspace(-1, 5, 25)
        vals = [misfit(g) for g in grid]
        i = int(np.argmin(vals))
        fine = np.linspace(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)], 9)
        best = fine[int(np.argmin([misfit(g) for g in fine]))]
        self._hs = (key, float(10 ** best))
        return self._hs[1]

    def depth_of_investigation(self, rho=None):
        """Глубина диффузии на самом позднем времени (оценка), м."""
        if rho is None:
            if self.is_central:
                rt = self.rho_tau()
                ok = ~self.mask & (self.signal > 0)
                late = ok & (self.times >= np.median(self.times[ok])) if ok.sum() > 3 else ok
                rho = float(np.nanmedian(rt[late if late.any() else ~self.mask]))
            else:
                rho = self.halfspace_rho()
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


def _gauss(n, a, b):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


class LoopTEMForward:
    """Прямоугольная генераторная петля, приёмная рамка конечного размера,
    любой разнос (в т.ч. за петлёй).

    Петля представлена равномерным слоем вертикальных магнитных диполей
    (квадратуры Гаусса по площади), рамка — усреднением dBz/dt по её площади.
    Отклик диполя считается быстрым ядром (tem_fast.VMDKernel) на сетке
    разносов и интерполируется. Результат — dBz/dt на 1 А, со знаком
    (внутри петли на поздних временах > 0). Есть аналитический якобиан.
    """

    def __init__(self, times, tx_a, tx_b=None, rx_side=0.0, offset=0.0, angle=0.0,
                 ramp=0.0, n_tx=8, n_rx=4, ramp_mid=False):
        """ramp — длительность линейного выключения (число или массив по временам);
        ramp_mid=True — время отсчитывается от середины фронта, иначе от конца."""
        from .tem_fast import VMDKernel

        self.times = np.asarray(times, float)
        tx_b = tx_b or tx_a
        xt, wxt = _gauss(n_tx, -tx_a / 2, tx_a / 2)
        yt, wyt = _gauss(n_tx, -tx_b / 2, tx_b / 2)
        XT, YT = np.meshgrid(xt, yt)
        WT = np.outer(wyt, wxt)                       # сумма = площадь петли
        ox = offset * np.cos(np.deg2rad(angle))
        oy = offset * np.sin(np.deg2rad(angle))
        if rx_side > 0:
            xr, wxr = _gauss(n_rx, ox - rx_side / 2, ox + rx_side / 2)
            yr, wyr = _gauss(n_rx, oy - rx_side / 2, oy + rx_side / 2)
            XR, YR = np.meshgrid(xr, yr)
            WR = np.outer(wyr, wxr) / rx_side ** 2   # среднее по рамке
        else:
            XR, YR, WR = np.array([[ox]]), np.array([[oy]]), np.array([[1.0]])
        d = np.hypot(XT.ravel()[:, None] - XR.ravel()[None, :],
                     YT.ravel()[:, None] - YR.ravel()[None, :]).ravel()
        w = (WT.ravel()[:, None] * WR.ravel()[None, :]).ravel()
        d = np.maximum(d, 0.5)

        # линейное выключение тока: dB/dt = среднее ступенчатого отклика по
        # [t_end, t_end + ramp], где t_end — время от конца фронта
        ramps = np.broadcast_to(np.asarray(ramp, float), self.times.shape)
        self.ramp = ramps
        if np.any(ramps > 0):
            x, wx = np.polynomial.legendre.leggauss(6)
            u, wu = 0.5 * (x + 1), 0.5 * wx                 # узлы на [0, 1]
            t_end = self.times - (ramps / 2 if ramp_mid else 0.0)
            t_end = np.maximum(t_end, 1e-7)
            t_eval = (t_end[:, None] + u[None, :] * ramps[:, None]).ravel()
            R = np.zeros((len(self.times), len(t_eval)))
            for i in range(len(self.times)):
                R[i, i * 6:(i + 1) * 6] = wu
        else:
            t_eval = self.times
            R = None
        self.R = R
        self.kernel = VMDKernel(d.min() * 0.99, d.max() * 1.01, t_eval)
        lg = np.log(self.kernel.r)
        ld = np.log(d)
        j = np.clip(np.searchsorted(lg, ld) - 1, 0, len(lg) - 2)
        f = (ld - lg[j]) / (lg[j + 1] - lg[j])
        P = np.zeros((len(ld), len(lg)))
        P[np.arange(len(j)), j] = 1 - f
        P[np.arange(len(j)), j + 1] = f
        self.v = w @ P                                 # веса на сетке разносов

    def __call__(self, rho, thick):
        out = self.kernel.dbdt(rho, thick) @ self.v
        return self.R @ out if self.R is not None else out

    def jac(self, rho, thick):
        """(отклик, d отклик / d log10 ρ слоёв)."""
        d, J = self.kernel.dbdt_jac(rho, thick)
        out = d @ self.v
        Jo = np.einsum("trj,r->tj", J, self.v)
        if self.R is not None:
            out, Jo = self.R @ out, self.R @ Jo
        return out, Jo


def make_forward(snd: "TEMSounding", times=None):
    """Подходящая прямая задача для геометрии зондирования."""
    times = snd.times if times is None else times
    simple = (snd.rx_side == 0 and snd.is_central and snd.tx_b in (0, snd.tx_side)
              and snd.ramps is None)
    if simple:
        return TEMForward(times, snd.tx_side, snd.config, snd.ramp)
    if snd.ramps is not None:
        ramp = np.interp(times, snd.times, snd.ramps)
        return LoopTEMForward(times, snd.tx_side, snd.tx_b or snd.tx_side, snd.rx_side,
                              snd.offset, snd.offset_angle, ramp, ramp_mid=True)
    return LoopTEMForward(times, snd.tx_side, snd.tx_b or snd.tx_side, snd.rx_side, snd.offset,
                          snd.offset_angle, snd.ramp)


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
