"""Комплексирование АМТЗ и ЗСБ в 1D.

* инверсия ЗСБ (Оккам);
* статический сдвиг кривых АМТЗ по модели ЗСБ: для модели ЗСБ решается
  1D прямая задача МТ, и на высоких частотах (где МТ «видит» только ту
  часть разреза, которую надёжно определяет ЗСБ) кривая ρa сдвигается
  по вертикали до совпадения. Фаза статическим сдвигом не искажается;
* совместная инверсия: одна слоистая модель подбирается одновременно под
  ρa и фазу АМТЗ и под сигнал ЗСБ; дополнительно можно подбирать
  поправку статического сдвига ρa АМТЗ (k) как свободный параметр.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .inv1d import MU0, bostick, layer_mesh, occam_core, rho_phase_1d, roughness
from .tem import TEMForward, TEMSounding


@dataclass
class JointResult:
    tops: np.ndarray
    rho: np.ndarray
    rms: float
    history: list
    ss: float | None = None             # подобранная поправка k: ρa_испр = k·ρa_набл
    mt: dict | None = None              # freq, rho, phs, erho, ephs, rho_pred, phs_pred
    tem: dict | None = None             # times, obs, err, pred, rho_tau_obs, rho_tau_pred
    rms_mt: float = float("nan")
    rms_tem: float = float("nan")
    label: str = ""
    extra: dict = field(default_factory=dict)


def mt_data(station, comp="det", rot=0.0):
    """Неисключённые данные станции (с учётом статического сдвига)."""
    s = station.rotated(rot) if rot else station
    rho, phs, er, ep = s.rho_phase(comp)
    ok = ~station.mask & np.isfinite(rho) & np.isfinite(phs) & (rho > 0)
    return dict(freq=station.freq[ok], rho=rho[ok], phs=phs[ok],
                erho=np.nan_to_num(er[ok]), ephs=np.nan_to_num(ep[ok]))


def tem_data(snd: TEMSounding):
    ok = ~snd.mask & np.isfinite(snd.signal) & (snd.signal > 0)
    return dict(times=snd.times[ok], obs=snd.signal[ok], err=snd.err[ok])


def _auto_depths(mt, tem_d, tem_snd):
    zf, zm = [], []
    if tem_d is not None:
        rt = tem_snd.rho_tau()[~tem_snd.mask]
        rlo = max(float(np.nanpercentile(rt, 5)), 0.1)
        rhi = float(np.nanpercentile(rt, 95))
        zf.append(0.4 * np.sqrt(2 * tem_d["times"].min() * rlo / MU0))
        zm.append(4.0 * np.sqrt(2 * tem_d["times"].max() * rhi / MU0))
    if mt is not None:
        zf.append(0.3 * 503.0 * np.sqrt(np.min(mt["rho"]) / mt["freq"].max()))
        zm.append(1.5 * 503.0 * np.sqrt(np.max(mt["rho"]) / mt["freq"].min()))
    return max(min(zf), 0.5), max(zm)


def joint_inversion(mt=None, tem: TEMSounding | None = None, *, n_layers=45, z_first=None,
                    z_max=None, rho0=None, floor_rho=0.10, floor_phs=2.9, floor_tem=0.05,
                    invert_ss=False, use_rho=True, use_phase=True, target_rms=1.0,
                    max_iter=30, progress=None, label="") -> JointResult:
    """mt — словарь из mt_data() или None; tem — TEMSounding или None."""
    if mt is None and tem is None:
        raise ValueError("нет данных")
    if mt is not None and len(mt["freq"]) < 2:
        raise ValueError("мало точек АМТЗ")
    td = tem_data(tem) if tem is not None else None
    if td is not None and len(td["times"]) < 3:
        raise ValueError("мало точек ЗСБ")
    if invert_ss and (mt is None or td is None):
        invert_ss = False

    zf, zm = _auto_depths(mt, td, tem)
    tops, thick = layer_mesh(n_layers, z_first or zf, z_max or zm)

    d, sd = [], []
    if mt is not None:
        if use_rho:
            d.append(np.log10(mt["rho"]))
            sd.append(np.maximum(mt["erho"] / mt["rho"], floor_rho) / np.log(10))
        if use_phase:
            d.append(mt["phs"])
            sd.append(np.maximum(mt["ephs"], floor_phs))
    fwd_tem = None
    if td is not None:
        rel = np.where(np.isfinite(td["err"]), td["err"] / td["obs"], 0.0)
        d.append(np.log10(td["obs"]))
        sd.append(np.maximum(rel, floor_tem) / np.log(10))
        fwd_tem = TEMForward(td["times"], tem.tx_side, tem.config, tem.ramp)
    d = np.concatenate(d)
    Wd = 1.0 / np.concatenate(sd)
    n_mt = (len(mt["freq"]) * (int(use_rho) + int(use_phase))) if mt is not None else 0

    def split(m):
        return (m[:n_layers], m[n_layers]) if invert_ss else (m, 0.0)

    def fwd(m):
        lr, lss = split(m)
        rho = 10 ** lr
        out = []
        if mt is not None:
            r, p = rho_phase_1d(rho, thick, mt["freq"])
            if use_rho:
                out.append(np.log10(r) - lss)   # наблюдённая = истинная / k
            if use_phase:
                out.append(p)
        if fwd_tem is not None:
            out.append(np.log10(np.maximum(fwd_tem(rho, thick), 1e-300)))
        return np.concatenate(out)

    if rho0 is None:
        vals = []
        if mt is not None:
            vals.append(np.median(mt["rho"]))
        if td is not None:
            vals.append(np.median(tem.rho_tau()[~tem.mask]))
        rho0 = float(np.exp(np.mean(np.log(vals))))
    m0 = np.full(n_layers, np.log10(rho0))
    lo, hi = np.full(n_layers, -3.0), np.full(n_layers, 6.0)
    if invert_ss:
        m0 = np.r_[m0, 0.0]
        lo, hi = np.r_[lo, -1.5], np.r_[hi, 1.5]
    R = roughness(n_layers, 1 if invert_ss else 0, w_extra=0.05)
    m, rms, hist = occam_core(fwd, d, Wd, m0, R, max_iter=max_iter, target_rms=target_rms,
                              lo=lo, hi=hi, progress=progress)

    lr, lss = split(m)
    rho = 10 ** lr
    res = JointResult(tops, rho, rms, hist, ss=(10 ** lss) if invert_ss else None, label=label)
    f = fwd(m)
    r = (d - f) * Wd
    if mt is not None:
        rp, pp = rho_phase_1d(rho, thick, mt["freq"])
        res.mt = dict(mt, rho_pred=rp / (10 ** lss), phs_pred=pp, rho_true_pred=rp)
        res.rms_mt = float(np.sqrt(np.mean(r[:n_mt] ** 2))) if n_mt else float("nan")
    if td is not None:
        pred = fwd_tem(rho, thick)
        res.tem = dict(td, pred=pred, rho_tau_obs=_rt(tem, td["times"], td["obs"]),
                       rho_tau_pred=_rt(tem, td["times"], pred))
        res.rms_tem = float(np.sqrt(np.mean(r[n_mt:] ** 2)))
    return res


def _rt(snd, times, sig):
    from .tem import rho_tau

    return rho_tau(times, sig, snd.tx_radius)


def static_shift_from_model(station, tops, rho, comp="xy", max_depth=None, min_points=2):
    """Множитель статического сдвига по 1D-модели (например, по ЗСБ).

    Сравниваются наблюдённая ρa (без уже введённой поправки) и ρa,
    рассчитанная для модели, на частотах, где глубина Бостика модельной
    кривой меньше max_depth (часть разреза, надёжно освещённая ЗСБ).
    Возвращает dict: k (множитель ρa — то, что пишется в Station.ss),
    use (маска частот), rho_model, overlap (False — общих частот нет,
    взяты самые высокие частоты АМТЗ).
    """
    r_obs, _, _, _ = station.rho_phase(comp)
    if comp in ("xy", "xx"):
        r_obs = r_obs / station.ss[0]
    elif comp in ("yx", "yy"):
        r_obs = r_obs / station.ss[1]
    else:
        r_obs = r_obs / np.sqrt(np.prod(station.ss))
    r_mod, p_mod = rho_phase_1d(rho, np.diff(tops), station.freq)
    zb, _ = bostick(station.freq, r_mod, p_mod)
    ok = ~station.mask & np.isfinite(r_obs) & (r_obs > 0)
    use = ok & (zb <= max_depth) if max_depth else ok.copy()
    overlap = int(use.sum()) >= min_points
    if not overlap:
        idx = np.where(ok)[0][np.argsort(-station.freq[ok])][:3]
        use = np.zeros_like(ok)
        use[idx] = True
    if not use.any():
        raise ValueError("нет данных АМТЗ для сравнения")
    k = float(10 ** np.median(np.log10(r_mod[use]) - np.log10(r_obs[use])))
    return dict(k=k, use=use, rho_model=r_mod, overlap=overlap)
