"""2D-инверсия МТ (TE/TM) на основе SimPEG.

Ось x' (ROTATION = strike) направлена вдоль простирания, профиль — вдоль y'.
TE = Zx'y' (E вдоль простирания), TM = Zy'x'.
Данные: ln(кажущееся сопротивление) и фаза, интерполированные на общий
набор частот. Параметр модели — ln(sigma) в ячейках ниже поверхности.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class Inv2DSettings:
    modes: tuple = ("te", "tm")
    strike: float = 0.0          # град от севера по часовой
    floor_rho: float = 0.10      # относительная ошибка rho_a
    floor_rho_tm: float | None = None  # отдельно для TM (None = как floor_rho); большое
                                       # значение = TM почти только по фазе
    floor_phs: float = 2.9       # град
    freq_per_decade: int = 5
    fmin: float | None = None
    fmax: float | None = None
    rho0: float | None = None    # стартовая модель (Ом·м), None = медиана данных
    cells_between: int = 3       # ячеек между соседними станциями
    dx: float | None = None      # ширина ячейки, м (None = авто)
    z_first: float | None = None  # мощность первого слоя, м
    z_core: float | None = None   # глубина «рабочей» области, м
    z_growth: float = 1.12
    max_iter: int = 15
    beta0_ratio: float = 10.0
    cooling_factor: float = 2.0
    alpha_s: float = 1e-4
    smooth_x: float = 1.0        # относительный вес горизонтальной гладкости
    smooth_z: float = 1.0
    rho_min: float = 0.1
    rho_max: float = 1e5
    target_chifact: float = 1.0
    cg_maxiter: int = 20
    reg_x_scale: float = 1.0      # служебные множители (для подбора)
    reg_z_scale: float = 4.0

    def to_dict(self):
        return asdict(self)


@dataclass
class Inv2DData:
    names: list
    x: np.ndarray                 # положение станций вдоль профиля, м
    freqs: np.ndarray             # частоты инверсии
    obs: dict = field(default_factory=dict)  # mode -> dict(rho, phs, erho, ephs) (nf, nst)


@dataclass
class Inv2DResult:
    mesh: object
    rho: np.ndarray               # (n_cells,), воздух = nan
    data: Inv2DData
    pred: dict                    # mode -> dict(rho, phs)
    rms_history: list
    rms: float
    core: tuple                   # xmin, xmax, zmin(depth max), 0
    settings: Inv2DSettings
    log: list = field(default_factory=list)


class StopInversion(Exception):
    pass


# ----------------------------------------------------------------------
def _interp_station(st, comp, freqs):
    rho, phs, erho, ephs = st.rho_phase(comp)
    ok = (~st.mask) & np.isfinite(rho) & np.isfinite(phs) & (rho > 0)
    out = [np.full(len(freqs), np.nan) for _ in range(4)]
    if ok.sum() < 2:
        return out
    lf = np.log10(st.freq[ok])
    order = np.argsort(lf)
    lf = lf[order]
    q = np.log10(freqs)
    inside = (q >= lf[0] - 1e-6) & (q <= lf[-1] + 1e-6)
    # не интерполируем через большие пропуски (> 0.5 декады)
    gaps = np.diff(lf)
    for k in np.where(gaps > 0.5)[0]:
        inside &= ~((q > lf[k] + 1e-6) & (q < lf[k + 1] - 1e-6))
    vals = [np.log10(rho[ok][order]), phs[ok][order], (erho / rho)[ok][order], ephs[ok][order]]
    for i, v in enumerate(vals):
        v = np.where(np.isfinite(v), v, np.nanmedian(v) if np.any(np.isfinite(v)) else 0.0)
        out[i][inside] = np.interp(q[inside], lf, v)
    out[0] = 10 ** out[0]
    out[2] = out[2] * out[0]
    return out


def prepare_data(stations, profile, s: Inv2DSettings) -> Inv2DData:
    """stations — список Station в порядке профиля."""
    along = []
    for st in stations:
        a, _ = profile.project(st.lat, st.lon)
        along.append(float(a))
    along = np.array(along)
    order = np.argsort(along)
    stations = [stations[i] for i in order]
    along = along[order]
    # совпадающие точки слегка разнесём
    for i in range(1, len(along)):
        if along[i] - along[i - 1] < 1.0:
            along[i] = along[i - 1] + 1.0

    fall = np.concatenate([st.freq[~st.mask] for st in stations])
    fmax = s.fmax or fall.max()
    fmin = s.fmin or fall.min()
    nd = max(np.log10(fmax / fmin), 0.1)
    nf = max(int(round(nd * s.freq_per_decade)) + 1, 2)
    freqs = np.logspace(np.log10(fmax), np.log10(fmin), nf)

    data = Inv2DData([st.name for st in stations], along, freqs)
    for mode in s.modes:
        comp = "xy" if mode == "te" else "yx"
        R = np.full((nf, len(stations)), np.nan)
        P, ER, EP = R.copy(), R.copy(), R.copy()
        for j, st in enumerate(stations):
            sr = st.rotated(s.strike)
            r, p, er, ep = _interp_station(sr, comp, freqs)
            R[:, j], P[:, j] = r, p
            fl = s.floor_rho_tm if (mode == "tm" and s.floor_rho_tm is not None) else s.floor_rho
            ER[:, j] = np.maximum(np.nan_to_num(er), fl * r)
            EP[:, j] = np.maximum(np.nan_to_num(ep), s.floor_phs)
        data.obs[mode] = dict(rho=R, phs=P, erho=ER, ephs=EP)
    return data


# ----------------------------------------------------------------------
def _pad(h0, factor, total):
    out = []
    acc = 0.0
    h = h0
    while acc < total:
        h *= factor
        out.append(h)
        acc += h
    return np.array(out)


def build_mesh(data: Inv2DData, s: Inv2DSettings, rho_bg: float):
    import discretize

    x = data.x
    fmax, fmin = data.freqs.max(), data.freqs.min()
    allr = np.concatenate([d["rho"][np.isfinite(d["rho"])] for d in data.obs.values()])
    rlo = np.percentile(allr, 5) if len(allr) else rho_bg
    rhi = max(np.percentile(allr, 95) if len(allr) else rho_bg, rho_bg)
    skin_min = 503.0 * np.sqrt(rlo / fmax)
    skin_core = 503.0 * np.sqrt(rho_bg / fmin)
    skin_max = 503.0 * np.sqrt(rhi / fmin)

    if s.dx:
        dx = s.dx
    else:
        sp = np.diff(x)
        dmin = np.median(sp) if len(sp) else 1000.0
        dx = dmin / max(s.cells_between, 1)
        width = max(x.max() - x.min(), dx)
        dx = max(dx, width / 300.0)
        dx = max(dx, skin_min / 4.0)
        dx = min(dx, max(width / 10.0, skin_min))
    width = x.max() - x.min()
    ncore = int(np.ceil(width / dx)) + 4
    hx_core = np.full(ncore, dx)
    pad_x = _pad(dx, 1.3, 2.5 * skin_max)
    hx = np.r_[pad_x[::-1], hx_core, pad_x]
    x0 = x.min() - 2 * dx - pad_x.sum()

    z1 = s.z_first or max(skin_min / 20.0, 0.5)
    z1 = float(np.round(z1, 1 if z1 < 10 else 0))
    zc = s.z_core or (min(skin_core, max(2 * width, 10 * skin_min)) if width > 0 else skin_core)
    hz = [z1]
    while sum(hz) < zc:
        hz.append(hz[-1] * s.z_growth)
    hz = np.array(hz)
    pad_z = _pad(hz[-1], 1.3, 2.5 * skin_max)
    hz_g = np.r_[hz, pad_z][::-1]  # снизу вверх
    air = _pad(z1, 1.6, max(3 * skin_max, 1e5))
    hz_all = np.r_[hz_g, air]
    mesh = discretize.TensorMesh([hx, hz_all], origin=[x0, -hz_g.sum()])
    core = (x.min() - 2 * dx, x.max() + 2 * dx, -float(hz.sum()), 0.0)
    return mesh, core


# ----------------------------------------------------------------------
def run_inversion(stations, profile, s: Inv2DSettings, progress=None, stop_flag=None,
                  log=None, on_setup=None) -> Inv2DResult:
    """progress(iter, rms, rho_cells) — колбэк после каждой итерации;
    stop_flag() -> bool — запрос остановки;
    on_setup(Inv2DResult) — заготовка результата со стартовой моделью."""
    from simpeg import (maps, data as sdata, data_misfit, regularization, optimization,
                        inverse_problem, inversion, directives)
    from simpeg.electromagnetics import natural_source as nsem
    from simpeg.utils import get_default_solver

    from ._simpeg_fast import FastLogTE, FastLogTM

    logs = []

    def say(msg):
        logs.append(msg)
        if log:
            log(msg)

    data = prepare_data(stations, profile, s)
    allr = np.concatenate([d["rho"][np.isfinite(d["rho"])] for d in data.obs.values()])
    if len(allr) == 0:
        raise ValueError("нет данных для инверсии в выбранном диапазоне")
    rho_bg = s.rho0 or float(10 ** np.median(np.log10(allr)))
    mesh, core = build_mesh(data, s, rho_bg)
    say(f"Сетка: {mesh.shape_cells[0]} x {mesh.shape_cells[1]} ячеек, "
        f"частот: {len(data.freqs)}, стартовая модель {rho_bg:.1f} Ом·м")

    active = mesh.cell_centers[:, 1] < 0
    n_act = int(active.sum())
    sigma_map = maps.ExpMap(mesh) * maps.InjectActiveCells(mesh, active, np.log(1e-8))
    solver = get_default_solver()
    say(f"Решатель: {solver.__name__}")

    misfits = []
    order_info = {}
    for mode in s.modes:
        ob = data.obs[mode]
        ori = "xy" if mode == "te" else "yx"
        srcs, dobs, std, idx = [], [], [], []
        for i, f in enumerate(data.freqs):
            ok = np.isfinite(ob["rho"][i]) & np.isfinite(ob["phs"][i])
            if not ok.any():
                continue
            loc = np.c_[data.x[ok], np.zeros(ok.sum())]
            rxs = [nsem.receivers.Impedance(loc, orientation=ori, component="apparent_resistivity",
                                           storeProjections=True),
                   nsem.receivers.Impedance(loc, orientation=ori, component="phase",
                                            storeProjections=True)]
            srcs.append(nsem.sources.Planewave(rxs, frequency=f))
            ph = ob["phs"][i][ok] - (180.0 if mode == "te" else 0.0)
            # ρa подгоняется в логарифмах: ln ρa, ошибка = относительная ошибка ρa
            dobs += [np.log(ob["rho"][i][ok]), ph]
            std += [ob["erho"][i][ok] / ob["rho"][i][ok], ob["ephs"][i][ok]]
            idx.append((i, np.where(ok)[0]))
        if not srcs:
            continue
        survey = nsem.survey.Survey(srcs)
        Sim = FastLogTE if mode == "te" else FastLogTM
        sim = Sim(mesh, survey=survey, sigmaMap=sigma_map, solver=solver)
        dat = sdata.Data(survey, dobs=np.concatenate(dobs), standard_deviation=np.concatenate(std))
        dm = data_misfit.L2DataMisfit(data=dat, simulation=sim)
        misfits.append(dm)
        order_info[mode] = (idx, dat)
        say(f"{mode.upper()}: {dat.nD} отсчётов")
    if not misfits:
        raise ValueError("нет данных для выбранных мод")
    nD = sum(m.data.nD for m in misfits)
    dmis = misfits[0]
    for m in misfits[1:]:
        dmis = dmis + m

    m0 = np.full(n_act, np.log(1.0 / rho_bg))
    if on_setup:
        rho_start = np.full(mesh.n_cells, np.nan)
        rho_start[active] = rho_bg
        on_setup(Inv2DResult(mesh, rho_start, data, {}, [], float("nan"), core, s))
    reg_mesh_h = np.min(mesh.h[1][mesh.h[1] > 0])
    # веса гладкости: alpha ~ L^2, где L — характерная длина сглаживания
    reg = regularization.WeightedLeastSquares(
        mesh, active_cells=active, reference_model=m0, alpha_s=s.alpha_s,
        alpha_x=s.smooth_x * s.reg_x_scale * float(np.min(mesh.h[0])) ** 2,
        alpha_y=s.smooth_z * s.reg_z_scale * float(reg_mesh_h) ** 2)

    opt = optimization.ProjectedGNCG(maxIter=s.max_iter, lower=np.log(1 / s.rho_max),
                                     upper=np.log(1 / s.rho_min), cg_maxiter=s.cg_maxiter, maxIterLS=15)
    opt.remember("xc")
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    rms_hist = []
    t0 = time.time()

    def rms_of(model):
        tot = 0.0
        for mf in misfits:
            r = mf.W * (mf.simulation.dpred(model) - mf.data.dobs)
            tot += float(r @ r)
        return float(np.sqrt(tot / nD))

    class Progress(directives.InversionDirective):
        def endIter(self):
            phi_d = self.invProb.phi_d
            # phi_d в SimPEG = ||W r||^2
            rms = float(np.sqrt(phi_d / nD))
            rms_hist.append(rms)
            say(f"Итерация {self.opt.iter}: RMS = {rms:.3f}, beta = {self.invProb.beta:.3g}, "
                f"{time.time() - t0:.0f} с")
            if progress:
                rho = np.full(mesh.n_cells, np.nan)
                rho[active] = np.exp(-self.opt.xc)
                progress(self.opt.iter, rms, rho)
            if stop_flag and stop_flag():
                raise StopInversion()

    dirs = [
        directives.BetaEstimate_ByEig(beta0_ratio=s.beta0_ratio, random_seed=1),
        directives.BetaSchedule(coolingFactor=s.cooling_factor, coolingRate=1),
        directives.TargetMisfit(chifact=s.target_chifact),
        Progress(),
    ]
    inv = inversion.BaseInversion(inv_prob, directiveList=dirs)
    say("Запуск инверсии…")
    try:
        m_rec = inv.run(m0)
    except StopInversion:
        say("Остановлено пользователем")
        m_rec = opt.xc if getattr(opt, "xc", None) is not None else m0

    rho = np.full(mesh.n_cells, np.nan)
    rho[active] = np.exp(-m_rec)

    pred = {}
    for mf, mode in zip(misfits, order_info):
        idx, dat = order_info[mode]
        dp = mf.simulation.dpred(m_rec)
        R = np.full_like(data.obs[mode]["rho"], np.nan)
        P = R.copy()
        k = 0
        for i, cols in idx:
            n = len(cols)
            R[i, cols] = np.exp(dp[k:k + n])
            P[i, cols] = dp[k + n:k + 2 * n] + (180.0 if mode == "te" else 0.0)
            k += 2 * n
        P = np.where(P > 180, P - 360, P)
        pred[mode] = dict(rho=R, phs=P)
    final_rms = rms_of(m_rec)
    say(f"Готово. Итоговый RMS = {final_rms:.3f}")
    return Inv2DResult(mesh, rho, data, pred, rms_hist, final_rms, core, s, logs)


def forward_2d(mesh, rho_cells, x_st, freqs, modes=("te", "tm")):
    """Прямая задача для произвольной модели (для синтетики/проверки)."""
    from simpeg import maps
    from simpeg.electromagnetics import natural_source as nsem
    from simpeg.utils import get_default_solver

    out = {}
    loc = np.c_[x_st, np.zeros(len(x_st))]
    for mode in modes:
        ori = "xy" if mode == "te" else "yx"
        srcs = [nsem.sources.Planewave(
            [nsem.receivers.Impedance(loc, orientation=ori, component="apparent_resistivity"),
             nsem.receivers.Impedance(loc, orientation=ori, component="phase")], frequency=f)
            for f in freqs]
        Sim = (nsem.simulation.Simulation2DElectricField if mode == "te"
               else nsem.simulation.Simulation2DMagneticField)
        sim = Sim(mesh, survey=nsem.survey.Survey(srcs), sigmaMap=maps.IdentityMap(),
                  solver=get_default_solver())
        d = sim.dpred(1.0 / rho_cells).reshape(len(freqs), 2, len(x_st))
        ph = d[:, 1] + (180.0 if mode == "te" else 0.0)
        ph = np.where(ph > 180, ph - 360, ph)
        out[mode] = dict(rho=d[:, 0], phs=ph)
    return out
