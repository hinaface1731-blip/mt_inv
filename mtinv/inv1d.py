"""1D МТ: прямая задача, трансформация Ниблетта–Бостика и инверсия Оккама.

Модель: N слоёв фиксированной мощности (последний — полупространство),
параметры — log10(rho). Данные — log10(rho_a) и фаза (град).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MU0 = 4e-7 * np.pi


def forward_1d(rho, thick, freq):
    """Импеданс слоистой среды (СИ, Ом). rho: (N,), thick: (N-1,)."""
    rho = np.asarray(rho, float)
    w = 2 * np.pi * np.asarray(freq, float)[:, None]
    k = np.sqrt(1j * w * MU0 / rho[None, :])
    Zi = 1j * w * MU0 / k  # собственный импеданс слоя
    Z = Zi[:, -1]
    for n in range(len(rho) - 2, -1, -1):
        th = np.tanh(k[:, n] * thick[n])
        Z = Zi[:, n] * (Z + Zi[:, n] * th) / (Zi[:, n] + Z * th)
    return Z


def rho_phase_1d(rho, thick, freq):
    Z = forward_1d(rho, thick, freq)
    w = 2 * np.pi * np.asarray(freq)
    return np.abs(Z) ** 2 / (w * MU0), np.rad2deg(np.angle(Z))


def bostick(freq, rho_a, phase):
    """Трансформация Ниблетта–Бостика: (глубина, rho)."""
    T = 1.0 / np.asarray(freq)
    depth = np.sqrt(rho_a * T / (2 * np.pi * MU0))
    ph = np.clip(np.deg2rad(phase), 1e-3, np.pi / 2 - 1e-3)
    rho_b = rho_a * (np.pi / (2 * ph) - 1)
    return depth, rho_b


def layer_mesh(n_layers=40, z_first=10.0, z_max=50000.0):
    """Глубины кровель слоёв (логарифмически), мощности слоёв."""
    tops = np.r_[0.0, np.logspace(np.log10(z_first), np.log10(z_max), n_layers - 1)]
    return tops, np.diff(tops)


@dataclass
class Occam1DResult:
    tops: np.ndarray
    rho: np.ndarray
    freq: np.ndarray
    rho_obs: np.ndarray
    phs_obs: np.ndarray
    erho: np.ndarray
    ephs: np.ndarray
    rho_pred: np.ndarray
    phs_pred: np.ndarray
    rms: float
    history: list


def occam1d(freq, rho_a, phase, erho=None, ephs=None, *, n_layers=40, z_first=None,
            z_max=None, rho0=None, floor_rho=0.1, floor_phs=2.86, max_iter=30,
            target_rms=1.0, use_rho=True, use_phase=True, progress=None):
    """Инверсия Оккама (Constable et al., 1987).

    floor_rho — относительная минимальная ошибка rho_a (0.1 = 10 %),
    floor_phs — минимальная ошибка фазы в градусах.
    """
    freq = np.asarray(freq, float)
    ok = np.isfinite(rho_a) & np.isfinite(phase) & (rho_a > 0)
    freq, rho_a, phase = freq[ok], np.asarray(rho_a)[ok], np.asarray(phase)[ok]
    erho = np.zeros_like(rho_a) if erho is None else np.nan_to_num(np.asarray(erho)[ok])
    ephs = np.zeros_like(phase) if ephs is None else np.nan_to_num(np.asarray(ephs)[ok])
    if len(freq) < 2:
        raise ValueError("слишком мало точек для инверсии")

    rmed = float(np.median(rho_a)) if rho0 is None else float(rho0)
    if z_first is None:
        z_first = 0.3 * 503.0 * np.sqrt(np.min(rho_a) / freq.max())
    if z_max is None:
        z_max = 1.5 * 503.0 * np.sqrt(np.max(rho_a) / freq.min())
    tops, thick = layer_mesh(n_layers, z_first, z_max)

    # данные: log10 rho_a, фаза
    s_lr = np.maximum(erho / rho_a, floor_rho) / np.log(10)
    s_ph = np.maximum(ephs, floor_phs)
    d_parts, s_parts = [], []
    if use_rho:
        d_parts.append(np.log10(rho_a))
        s_parts.append(s_lr)
    if use_phase:
        d_parts.append(phase)
        s_parts.append(s_ph)
    d = np.concatenate(d_parts)
    Wd = 1.0 / np.concatenate(s_parts)
    N = len(d)

    def fwd(m):
        r, p = rho_phase_1d(10 ** m, thick, freq)
        out = []
        if use_rho:
            out.append(np.log10(r))
        if use_phase:
            out.append(p)
        return np.concatenate(out)

    def jac(m, f0):
        J = np.empty((N, len(m)))
        h = 0.01
        for i in range(len(m)):
            mm = m.copy()
            mm[i] += h
            J[:, i] = (fwd(mm) - f0) / h
        return J

    def rms_of(f):
        return float(np.sqrt(np.mean(((d - f) * Wd) ** 2)))

    M = n_layers
    R = np.diff(np.eye(M), axis=0)
    RtR = R.T @ R
    m = np.full(M, np.log10(rmed))
    f = fwd(m)
    rms = rms_of(f)
    history = [rms]
    lambdas = np.logspace(-3, 4, 22)
    for it in range(max_iter):
        J = jac(m, f)
        JW = J * Wd[:, None]
        dhat = (d - f + J @ m) * Wd
        JtJ = JW.T @ JW
        Jtd = JW.T @ dhat
        cands = []
        for lam in lambdas:
            try:
                mn = np.linalg.solve(JtJ + lam * RtR, Jtd)
            except np.linalg.LinAlgError:
                continue
            mn = np.clip(mn, -3, 6)
            fn = fwd(mn)
            cands.append((rms_of(fn), lam, mn, fn))
        if not cands:
            break
        best = min(cands, key=lambda c: c[0])
        if best[0] < target_rms:
            # фаза 2: самая гладкая модель, достигающая целевого RMS
            ok_c = [c for c in cands if c[0] <= target_rms]
            best = max(ok_c, key=lambda c: c[1])
        new_rms, lam, mn, fn = best
        if new_rms > rms * 0.999 and rms > target_rms:
            # нет прогресса — делаем полшага
            mh = 0.5 * (m + mn)
            fh = fwd(mh)
            if rms_of(fh) < rms:
                mn, fn, new_rms = mh, fh, rms_of(fh)
            else:
                break
        rough_old = np.sum((R @ m) ** 2)
        rough_new = np.sum((R @ mn) ** 2)
        m, f, prev = mn, fn, rms
        rms = new_rms
        history.append(rms)
        if progress:
            progress(it + 1, rms)
        if rms <= target_rms and prev <= target_rms and abs(rough_new - rough_old) < 1e-3 * max(rough_old, 1e-9):
            break

    r_pred, p_pred = rho_phase_1d(10 ** m, thick, freq)
    return Occam1DResult(tops, 10 ** m, freq, rho_a, phase, erho, ephs, r_pred, p_pred,
                         rms, history)
