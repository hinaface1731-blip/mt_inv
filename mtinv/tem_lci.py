"""Латерально-связанная инверсия ЗСБ (LCI, quasi-2D).

Под каждым пикетом — 1D-модель на общей сетке слоёв, но все пикеты профиля
подбираются одновременно: кроме гладкости по глубине, штрафуются различия
lg ρ одного и того же слоя у соседних по профилю пикетов (Auken et al.,
2005, Exploration Geophysics 36). Получается непрерывный разрез без
«столбиков»; шумные пикеты подтягиваются соседями, а резкие границы
остаются, только если их требуют данные.

Прямая задача — быстрое ядро (LoopTEMForward) с аналитическим якобианом,
поэтому профиль из ~100 пикетов считается за минуты.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .inv1d import layer_mesh, occam_core
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
                  progress=None, n_threads=None) -> LCIResult:
    """soundings — список TEMSounding; x — их положения вдоль профиля (м)."""
    x = np.asarray(x, float)
    ns = len(soundings)
    tops, thick = layer_mesh(n_layers, z_first, z_max)
    blocks = []
    for s in soundings:
        td = tem_data(s)
        if len(td["times"]) < 3:
            raise ValueError(f"{s.name}: мало точек")
        tr, dt, st, s0 = tem_transform(td["obs"], td["err"], floor)
        blocks.append(dict(snd=s, td=td, tr=tr, d=dt, sd=st, s0=s0,
                           fwd=_forward(s, td["times"])))
    nd = [len(b["d"]) for b in blocks]
    off = np.r_[0, np.cumsum(nd)]
    d = np.concatenate([b["d"] for b in blocks])
    Wd = 1.0 / np.concatenate([b["sd"] for b in blocks])

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=n_threads or min(8, os.cpu_count() or 1))

    def fwd(m):
        M = m.reshape(ns, n_layers)
        out = pool.map(lambda i: blocks[i]["tr"](blocks[i]["fwd"](10 ** M[i], thick)), range(ns))
        return np.concatenate(list(out))

    def jac_fn(m, f0):
        M = m.reshape(ns, n_layers)
        J = np.zeros((len(d), len(m)))

        def one(i):
            b = blocks[i]
            val, Jt = b["fwd"].jac(10 ** M[i], thick)
            dtr = 1.0 / np.sqrt(b["s0"] ** 2 + val ** 2)
            J[off[i]:off[i + 1], i * n_layers:(i + 1) * n_layers] = Jt * dtr[:, None]

        list(pool.map(one, range(ns)))
        return J

    if rho0 is None:
        rho0 = float(np.exp(np.median([np.log(s.halfspace_rho()) for s in soundings])))
    m0 = np.full(ns * n_layers, np.log10(rho0))
    R = lateral_roughness(x, n_layers, alpha_lat)
    m, rms, hist = occam_core(fwd, d, Wd, m0, R, max_iter=max_iter, target_rms=target_rms,
                              lo=-1.0, hi=5.5, progress=progress, jac_fn=jac_fn,
                              lambdas=np.logspace(-2, 3, 11))
    f = fwd(m)
    r = (d - f) * Wd
    M = m.reshape(ns, n_layers)
    rms_each = np.array([np.sqrt(np.mean(r[off[i]:off[i + 1]] ** 2)) for i in range(ns)])

    # глубинность: кровля самого глубокого слоя, ниже которого остаётся >= 1 %
    # суммарной чувствительности Σ|W·J| пикета
    J = jac_fn(m, f) * Wd[:, None]
    doi = np.zeros(ns)
    for i in range(ns):
        sens = np.abs(J[off[i]:off[i + 1], i * n_layers:(i + 1) * n_layers]).sum(axis=0)
        c = np.cumsum(sens[::-1])[::-1]               # чувствительность слоёв глубже k
        k = np.where(c >= 0.01 * c[0])[0]
        doi[i] = tops[k[-1]] if len(k) else tops[-1]
    fits = []
    for i, b in enumerate(blocks):
        pred = b["fwd"](10 ** M[i], thick)
        R_ = b["snd"].tx_radius
        fits.append(dict(times=b["td"]["times"], obs=b["td"]["obs"], pred=pred,
                         rho_tau_obs=rho_tau(b["td"]["times"], b["td"]["obs"], R_),
                         rho_tau_pred=rho_tau(b["td"]["times"], pred, R_)))
    pool.shutdown()
    return LCIResult([s.name for s in soundings], x, tops, 10 ** M, rms, rms_each, hist,
                     doi, fits)
