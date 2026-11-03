"""Латерально-связанная инверсия ЗСБ (LCI, quasi-2D).

Под каждым пикетом — 1D-модель на общей сетке слоёв, но все пикеты профиля
подбираются одновременно: кроме гладкости по глубине, штрафуются различия
lg ρ одного и того же слоя у соседних по профилю пикетов (Auken et al.,
2005, Exploration Geophysics 36). Получается непрерывный разрез без
«столбиков»; шумные пикеты подтягиваются соседями, а резкие границы
остаются, только если их требуют данные.

Прямая задача — быстрое ядро (LoopTEMForward) с аналитическим якобианом,
поэтому профиль из ~100 пикетов считается за минуты.

К пикету можно приложить данные АМТЗ ближайшей станции (mt_data из joint1d):
* только фазы (use_rho=False) — совместная инверсия в духе Meju (1996): фаза
  не искажается статическим сдвигом, уровень сопротивлений задаёт ЗСБ;
* ρa и фазы с подбором множителя статического сдвига k для каждой станции
  (invert_ss=True): наблюдённая ρa = истинная / k.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .inv1d import layer_mesh, occam_core, rho_phase_1d
from .joint1d import tem_data, tem_transform
from .tem import LoopTEMForward, TEMSounding, rho_tau


@dataclass
class LCIResult:
    names: list
    x: np.ndarray            # положение пикетов вдоль профиля, м
    tops: np.ndarray         # кровли слоёв (общие), м
    rho: np.ndarray          # (n_snd, n_layers), Ом·м
    rms: float
    rms_each: np.ndarray     # RMS каждого пикета
    history: list
    doi: np.ndarray          # глубинность по чувствительности, м
    fits: list = field(default_factory=list)   # dict(times, obs, pred) по пикетам
    rms_tem: np.ndarray | None = None   # RMS ЗСБ по пикетам
    rms_mt: np.ndarray | None = None    # RMS АМТЗ по пикетам (nan — без АМТЗ)
    ss: np.ndarray | None = None        # подобранные множители k (nan — нет)


def _forward(snd: TEMSounding, times):
    ramp = np.interp(times, snd.times, snd.ramps) if snd.ramps is not None else snd.ramp
    return LoopTEMForward(times, snd.tx_side, snd.tx_b or snd.tx_side, snd.rx_side,
                          snd.offset, snd.offset_angle, ramp,
                          ramp_mid=snd.ramps is not None)


def lateral_roughness(x, n_layers, alpha_lat=1.0, d_ref=100.0, w_max=3.0):
    """Строки первых разностей по глубине (внутри пикета) и по профилю
    (тот же слой у соседних пикетов). Вес латеральной связи
    alpha_lat·sqrt(d_ref/расстояние), не больше w_max."""
    ns = len(x)
    rows = []
    for i in range(ns):
        for k in range(n_layers - 1):
            r = np.zeros(ns * n_layers)
            r[i * n_layers + k], r[i * n_layers + k + 1] = -1.0, 1.0
            rows.append(r)
    order = np.argsort(x)
    for a, b in zip(order[:-1], order[1:]):
        dist = max(abs(x[b] - x[a]), 1.0)
        w = min(alpha_lat * np.sqrt(d_ref / dist), w_max)
        for k in range(n_layers):
            r = np.zeros(ns * n_layers)
            r[a * n_layers + k], r[b * n_layers + k] = -w, w
            rows.append(r)
    return np.array(rows)


def lci_inversion(soundings, x, *, n_layers=35, z_first=10.0, z_max=4000.0, alpha_lat=1.0,
                  floor=0.10, rho0=None, target_rms=1.0, max_iter=20,
                  progress=None, n_threads=None, mt=None, use_rho=False, invert_ss=False,
                  floor_rho=0.10, floor_phs=2.0, w_ss=0.05) -> LCIResult:
    """soundings — список TEMSounding; x — их положения вдоль профиля (м);
    mt — None или список той же длины: dict из joint1d.mt_data() или None."""
    x = np.asarray(x, float)
    ns = len(soundings)
    mt = list(mt) if mt is not None else [None] * ns
    if not use_rho:
        invert_ss = False
    tops, thick = layer_mesh(n_layers, z_first, z_max)
    nl = n_layers
    npar = ns * nl + (ns if invert_ss else 0)
    blocks = []
    for i, s in enumerate(soundings):
        td = tem_data(s)
        if len(td["times"]) < 3:
            raise ValueError(f"{s.name}: мало точек")
        tr, dt, st, s0 = tem_transform(td["obs"], td["err"], floor)
        b = dict(snd=s, td=td, tr=tr, s0=s0, fwd=_forward(s, td["times"]), nt=len(dt), mt=mt[i])
        dd, sd = [dt], [st]
        if mt[i] is not None:
            if use_rho:
                dd.append(np.log10(mt[i]["rho"]))
                sd.append(np.maximum(mt[i]["erho"] / mt[i]["rho"], floor_rho) / np.log(10))
            dd.append(mt[i]["phs"])
            sd.append(np.maximum(mt[i]["ephs"], floor_phs))
        b["d"], b["sd"] = np.concatenate(dd), np.concatenate(sd)
        blocks.append(b)
    nd = [len(b["d"]) for b in blocks]
    off = np.r_[0, np.cumsum(nd)]
    d = np.concatenate([b["d"] for b in blocks])
    Wd = 1.0 / np.concatenate([b["sd"] for b in blocks])

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=n_threads or min(8, os.cpu_count() or 1))

    def mt_pred(i, lr, lss):
        r, p = rho_phase_1d(10 ** lr, thick, blocks[i]["mt"]["freq"])
        out = [np.log10(r) - lss] if use_rho else []
        out.append(p)
        return np.concatenate(out)

    def one_fwd(i, m):
        b = blocks[i]
        lr = m[i * nl:(i + 1) * nl]
        out = [b["tr"](b["fwd"](10 ** lr, thick))]
        if b["mt"] is not None:
            out.append(mt_pred(i, lr, m[ns * nl + i] if invert_ss else 0.0))
        return np.concatenate(out)

    def fwd(m):
        return np.concatenate(list(pool.map(lambda i: one_fwd(i, m), range(ns))))

    def jac_fn(m, f0):
        J = np.zeros((len(d), len(m)))

        def one(i):
            b = blocks[i]
            lr = m[i * nl:(i + 1) * nl]
            val, Jt = b["fwd"].jac(10 ** lr, thick)
            dtr = 1.0 / np.sqrt(b["s0"] ** 2 + val ** 2)
            r0 = off[i]
            J[r0:r0 + b["nt"], i * nl:(i + 1) * nl] = Jt * dtr[:, None]
            if b["mt"] is not None:
                lss = m[ns * nl + i] if invert_ss else 0.0
                base = mt_pred(i, lr, lss)
                h = 0.01
                for k in range(nl):
                    lk = lr.copy()
                    lk[k] += h
                    J[r0 + b["nt"]:off[i + 1], i * nl + k] = (mt_pred(i, lk, lss) - base) / h
                if invert_ss:
                    nf = len(b["mt"]["freq"])
                    J[r0 + b["nt"]:r0 + b["nt"] + nf, ns * nl + i] = -1.0

        list(pool.map(one, range(ns)))
        return J

    if rho0 is None:
        rho0 = float(np.exp(np.median([np.log(s.halfspace_rho()) for s in soundings])))
    m0 = np.full(npar, np.log10(rho0))
    lo = np.full(npar, -1.0)
    hi = np.full(npar, 5.5)
    R = lateral_roughness(x, nl, alpha_lat)
    if invert_ss:
        m0[ns * nl:] = 0.0
        lo[ns * nl:], hi[ns * nl:] = -1.5, 1.5
        R = np.hstack([R, np.zeros((R.shape[0], ns))])
        D = np.zeros((ns, npar))
        D[np.arange(ns), ns * nl + np.arange(ns)] = w_ss
        R = np.vstack([R, D])
    m, rms, hist = occam_core(fwd, d, Wd, m0, R, max_iter=max_iter, target_rms=target_rms,
                              lo=lo, hi=hi, progress=progress, jac_fn=jac_fn,
                              lambdas=np.logspace(-2, 3, 11))
    f = fwd(m)
    r = (d - f) * Wd
    M = m[:ns * nl].reshape(ns, nl)
    rms_each = np.array([np.sqrt(np.mean(r[off[i]:off[i + 1]] ** 2)) for i in range(ns)])
    rms_tem = np.array([np.sqrt(np.mean(r[off[i]:off[i] + blocks[i]["nt"]] ** 2))
                        for i in range(ns)])
    rms_mt = np.array([np.sqrt(np.mean(r[off[i] + blocks[i]["nt"]:off[i + 1]] ** 2))
                       if blocks[i]["mt"] is not None else np.nan for i in range(ns)])
    ss = None
    if invert_ss:
        ss = np.array([10 ** m[ns * nl + i] if blocks[i]["mt"] is not None else np.nan
                       for i in range(ns)])

    # глубинность: кровля самого глубокого слоя, ниже которого остаётся >= 1 %
    # суммарной чувствительности Σ|W·J| пикета (ЗСБ и АМТЗ вместе)
    J = jac_fn(m, f) * Wd[:, None]
    doi = np.zeros(ns)
    for i in range(ns):
        sens = np.abs(J[off[i]:off[i + 1], i * nl:(i + 1) * nl]).sum(axis=0)
        c = np.cumsum(sens[::-1])[::-1]               # чувствительность слоёв глубже k
        k = np.where(c >= 0.01 * c[0])[0]
        doi[i] = tops[k[-1]] if len(k) else tops[-1]
    fits = []
    for i, b in enumerate(blocks):
        pred = b["fwd"](10 ** M[i], thick)
        R_ = b["snd"].tx_radius
        fit = dict(times=b["td"]["times"], obs=b["td"]["obs"], pred=pred,
                   rho_tau_obs=rho_tau(b["td"]["times"], b["td"]["obs"], R_),
                   rho_tau_pred=rho_tau(b["td"]["times"], pred, R_))
        if b["mt"] is not None:
            rp, pp = rho_phase_1d(10 ** M[i], thick, b["mt"]["freq"])
            fit["mt"] = dict(b["mt"], rho_pred=rp, phs_pred=pp)
        fits.append(fit)
    pool.shutdown()
    return LCIResult([s.name for s in soundings], x, tops, 10 ** M, rms, rms_each, hist,
                     doi, fits, rms_tem, rms_mt, ss)
