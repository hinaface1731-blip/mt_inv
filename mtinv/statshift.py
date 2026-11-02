"""Статистическая нормализация уровня кривых (коррекция статического сдвига
без ЗСБ).

Так корректировали приповерхностные искажения на Горном Таймыре
(Пальшин, Алексанова, Яковлев и др., Геофизические исследования, 2017,
т. 18, № 2): модуль импеданса на выбранном периоде сглаживается вдоль
профиля (пространственный фильтр), и для каждого пункта вычисляется
коэффициент нормализации, на который умножаются амплитудные кривые во
всём диапазоне частот. Авторы рекомендуют такой подход при шаге между
пунктами 200–300 м; при шаге от 1 км — ЗСБ в каждом пункте.
"""

from __future__ import annotations

import numpy as np


def level_at_period(st, period, comp="det", band=0.25):
    """lg ρa компоненты на периоде period (с): медиана неисключённых точек
    в окне ±band декады (nan, если таких нет)."""
    rho, _, _, _ = st.rho_phase(comp)
    lt = np.log10(st.period)
    ok = ~st.mask & np.isfinite(rho) & (rho > 0) & (np.abs(lt - np.log10(period)) <= band)
    if not ok.any():
        return np.nan
    # уровень без уже введённой поправки
    if comp in ("xy", "xx"):
        r = rho[ok] / st.ss[0]
    elif comp in ("yx", "yy"):
        r = rho[ok] / st.ss[1]
    else:
        r = rho[ok] / np.sqrt(np.prod(st.ss))
    return float(np.median(np.log10(r)))


def statistical_normalization(stations, positions, period=0.1, window=2000.0, comp="det",
                              per_component=False, filt="median"):
    """Коэффициенты нормализации (множители ρa) для станций профиля.

    stations — список Station, positions — их положение вдоль профиля (м).
    Для каждой станции уровень lg ρa на периоде period сравнивается с
    медианой (или средним) соседей в окне ±window/2 по профилю.
    per_component=False — один множитель для XY и YX (по инварианту det, как
    нормализация модуля тензора); True — отдельно для XY и YX.
    Возвращает список (k_xy, k_yx, число соседей) по станциям; nan — уровень
    не определён (нет данных на этом периоде).
    """
    x = np.asarray(positions, float)
    comps = ("xy", "yx") if per_component else (comp,)
    levels = {c: np.array([level_at_period(s, period, c) for s in stations]) for c in comps}
    out = []
    for i in range(len(stations)):
        near = np.abs(x - x[i]) <= window / 2
        ks = []
        n = 0
        for c in comps:
            lv = levels[c]
            sel = near & np.isfinite(lv)
            n = int(sel.sum())
            if not np.isfinite(lv[i]) or n < 3:
                ks.append(np.nan)
                continue
            ref = np.median(lv[sel]) if filt == "median" else np.mean(lv[sel])
            ks.append(float(10 ** (ref - lv[i])))
        if not per_component:
            ks = [ks[0], ks[0]]
        out.append((ks[0], ks[1], n))
    return out


def apply_normalization(stations, coefs):
    """Записать множители в Station.ss (nan — без поправки)."""
    for st, (kx, ky, _) in zip(stations, coefs):
        st.ss = np.array([kx if np.isfinite(kx) else 1.0, ky if np.isfinite(ky) else 1.0])
