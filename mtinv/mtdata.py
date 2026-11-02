"""Модель данных одной МТ-станции и производные величины.

Соглашения: x — север, y — восток, импеданс в полевых единицах
мВ/км/нТл (как в EDI), rho_a = 0.2 * T * |Z|^2 [Ом·м].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

COMPONENTS = ("xy", "yx", "det", "ssq", "xx", "yy")
_IDX = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}


@dataclass
class Station:
    name: str
    lat: float
    lon: float
    elev: float
    freq: np.ndarray
    Z: np.ndarray            # (nf, 2, 2) complex
    Zerr: np.ndarray         # (nf, 2, 2) std |Z|
    T: np.ndarray | None = None      # (nf, 2) complex (Tzx, Tzy)
    Terr: np.ndarray | None = None   # (nf, 2)
    path: str = ""
    zrot: float = 0.0
    mask: np.ndarray = field(default=None)  # True = частота исключена
    # множители статического сдвига ρa для строк X (xx, xy) и Y (yx, yy)
    ss: np.ndarray = field(default=None)
    # True = значение восстановлено по соседним частотам (fill_gaps)
    filled: np.ndarray = field(default=None)

    def __post_init__(self):
        self.freq = np.asarray(self.freq, dtype=float)
        if self.mask is None:
            self.mask = np.zeros(len(self.freq), dtype=bool)
        if self.filled is None:
            self.filled = np.zeros(len(self.freq), dtype=bool)
        self._orig = None  # (Z, Zerr, T, Terr, mask) до заполнения пропусков
        self.ss = np.ones(2) if self.ss is None else np.asarray(self.ss, float)
        # пустые ошибки -> 5 % от |Z| строки
        bad = ~np.isfinite(self.Zerr) | (self.Zerr <= 0)
        if np.any(bad):
            ref = np.sqrt(np.abs(self.Z[:, 0, 1] * self.Z[:, 1, 0]))[:, None, None]
            ref = np.broadcast_to(ref, self.Z.shape)
            self.Zerr = np.where(bad, 0.05 * ref, self.Zerr)

    # ------------------------------------------------------------------
    @property
    def period(self):
        return 1.0 / self.freq

    @property
    def Zc(self):
        """Импеданс с учётом статического сдвига (ρa строки умножается на ss)."""
        return self.Z * np.sqrt(self.ss)[None, :, None]

    @property
    def Zerrc(self):
        return self.Zerr * np.sqrt(self.ss)[None, :, None]

    @property
    def has_static_shift(self):
        return bool(np.any(np.abs(self.ss - 1.0) > 1e-9))

    @property
    def has_tipper(self):
        return self.T is not None and np.any(np.isfinite(self.T))

    def sort(self):
        """Сортировка по возрастанию периода."""
        idx = np.argsort(-self.freq)
        self.freq = self.freq[idx]
        self.Z = self.Z[idx]
        self.Zerr = self.Zerr[idx]
        self.mask = self.mask[idx]
        self.filled = self.filled[idx]
        if self.T is not None:
            self.T = self.T[idx]
            self.Terr = self.Terr[idx]

    # ------------------------------------------------------------------
    def zcomp(self, comp: str):
        """Комплексная компонента импеданса и её ошибка."""
        comp = comp.lower()
        Zc, Ec = self.Zc, self.Zerrc
        if comp in _IDX:
            i, j = _IDX[comp]
            return Zc[:, i, j], Ec[:, i, j]
        Z = Zc.copy()
        # отсутствующие диагональные компоненты считаем нулевыми
        for i in (0, 1):
            Z[:, i, i] = np.where(np.isfinite(Z[:, i, i]), Z[:, i, i], 0)
        eo = np.sqrt(0.5 * (Ec[:, 0, 1] ** 2 + Ec[:, 1, 0] ** 2))
        if comp == "det":
            d = Z[:, 0, 0] * Z[:, 1, 1] - Z[:, 0, 1] * Z[:, 1, 0]
            return np.sqrt(d), eo
        if comp == "ssq":
            s = np.sum(Z.reshape(-1, 4) ** 2, axis=1) / 2.0
            return np.sqrt(s), eo
        if comp in ("av", "avg"):
            return 0.5 * (Z[:, 0, 1] - Z[:, 1, 0]), eo
        raise ValueError(comp)

    def rho_phase(self, comp: str, fold=True):
        """rho_a, phase(deg), err_rho, err_phase(deg).

        fold=True переносит фазу yx в первый квадрант (+180).
        """
        z, ez = self.zcomp(comp)
        rho = 0.2 / self.freq * np.abs(z) ** 2
        phs = np.rad2deg(np.angle(z))
        if fold:
            phs = np.where(phs < -90, phs + 180, phs)
            phs = np.where(phs > 180 - 1e-9, phs - 180, phs)
        rel = ez / np.maximum(np.abs(z), 1e-30)
        erho = 2.0 * rel * rho
        ephs = np.rad2deg(np.arcsin(np.clip(rel, 0, 1)))
        return rho, phs, erho, ephs

    # ------------------------------------------------------------------
    def rotated(self, angle_deg: float) -> "Station":
        """Поворот системы координат на angle по часовой стрелке
        (ось x' направлена по азимуту angle). Статический сдвиг применяется
        до поворота."""
        a = np.deg2rad(angle_deg)
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, s], [-s, c]])
        Z = np.einsum("ik,fkl,jl->fij", R, self.Zc, R)
        E2 = np.einsum("ik,fkl,jl->fij", R ** 2, self.Zerrc ** 2, R ** 2)
        T = Terr = None
        if self.T is not None:
            T = np.einsum("fk,jk->fj", self.T, R)
            Terr = np.sqrt(np.einsum("fk,jk->fj", self.Terr ** 2, R ** 2))
        return Station(self.name, self.lat, self.lon, self.elev, self.freq.copy(), Z,
                       np.sqrt(E2), T, Terr, self.path, self.zrot + angle_deg,
                       self.mask.copy(), filled=self.filled.copy())

    # ------------------------------------------------------------------
    def phase_tensor(self):
        """Параметры фазового тензора (Caldwell et al., 2004).

        Возвращает dict: phimax, phimin (град), alpha, beta (skew), strike (град),
        ellipticity.
        """
        X = self.Zc.real
        Y = self.Zc.imag
        det = X[:, 0, 0] * X[:, 1, 1] - X[:, 0, 1] * X[:, 1, 0]
        det = np.where(np.abs(det) < 1e-30, np.nan, det)
        Xi = np.empty_like(X)
        Xi[:, 0, 0] = X[:, 1, 1] / det
        Xi[:, 1, 1] = X[:, 0, 0] / det
        Xi[:, 0, 1] = -X[:, 0, 1] / det
        Xi[:, 1, 0] = -X[:, 1, 0] / det
        P = np.einsum("fij,fjk->fik", Xi, Y)
        p11, p12, p21, p22 = P[:, 0, 0], P[:, 0, 1], P[:, 1, 0], P[:, 1, 1]
        tr = p11 + p22
        sk = p12 - p21
        det_p = p11 * p22 - p12 * p21
        pi1 = 0.5 * np.sqrt((p11 - p22) ** 2 + (p12 + p21) ** 2)
        pi2 = 0.5 * np.sqrt(tr ** 2 + sk ** 2)
        phimax = np.rad2deg(np.arctan(pi2 + pi1))
        phimin = np.rad2deg(np.arctan(np.sign(det_p) * np.abs(pi2 - pi1)))
        alpha = 0.5 * np.rad2deg(np.arctan2(p12 + p21, p11 - p22))
        beta = 0.5 * np.rad2deg(np.arctan2(sk, tr))
        ell = (phimax - phimin) / np.maximum(phimax + phimin, 1e-9)
        return dict(phimax=phimax, phimin=phimin, alpha=alpha, beta=beta,
                    strike=alpha - beta, ellipticity=ell)

    def swift_skew(self):
        Z = self.Zc
        return np.abs(Z[:, 0, 0] + Z[:, 1, 1]) / np.maximum(np.abs(Z[:, 0, 1] - Z[:, 1, 0]), 1e-30)

    def induction_arrows(self, convention="parkinson"):
        """Реальные и мнимые индукционные стрелки (северная, восточная компоненты)."""
        if self.T is None:
            return None
        s = -1.0 if convention == "parkinson" else 1.0
        return (s * self.T.real[:, 0], s * self.T.real[:, 1],
                s * self.T.imag[:, 0], s * self.T.imag[:, 1])


def median_strike(stations, fmin=None, fmax=None):
    """Медианное простирание по фазовому тензору (по модулю 90°)."""
    vals = []
    for st in stations:
        pt = st.phase_tensor()
        sel = ~st.mask & np.isfinite(pt["strike"])
        if fmin is not None:
            sel &= st.freq >= fmin
        if fmax is not None:
            sel &= st.freq <= fmax
        vals.append(np.mod(pt["strike"][sel], 90.0))
    if not vals:
        return np.nan
    v = np.concatenate(vals)
    if len(v) == 0:
        return np.nan
    # круговая медиана по модулю 90°
    ang = np.deg2rad(v * 4)
    mean = np.rad2deg(np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))) / 4.0
    return float(np.mod(mean, 90.0))


def bad_points(st: Station, max_rel_err=0.5, jump=0.4, comps=("xy", "yx")):
    """Плохие точки по компонентам: {comp: bool-массив}.

    Точка плохая, если относительная ошибка |Z| больше max_rel_err
    (0.5 = ошибка ρa ~100 %), фаза вне [-5°, 95°] (YX перенесена в 1-й
    квадрант), либо lg ρa отличается от медианы соседей (окно 5 точек)
    больше чем на jump декады, или фаза — больше чем на 20°.
    """
    out = {}
    for c in comps:
        rho, phs, er, ep = st.rho_phase(c)
        z, ez = st.zcomp(c)
        rel = ez / np.maximum(np.abs(z), 1e-30)
        lr = np.log10(np.where(rho > 0, rho, np.nan))
        hard = ~np.isfinite(lr) | ~np.isfinite(phs) | (rel > max_rel_err) | (phs < -5) | (phs > 95)
        n = len(lr)
        lf = np.log10(st.freq)
        # 1) узкое окно (5 точек)
        narrow = np.zeros(n, bool)
        for i in range(n):
            lo, hi = max(0, i - 2), min(n, i + 3)
            nb = [j for j in range(lo, hi) if j != i and not hard[j]]
            if len(nb) >= 2:
                if abs(lr[i] - np.nanmedian(lr[nb])) > jump or \
                        abs(phs[i] - np.nanmedian(phs[nb])) > 20:
                    narrow[i] = True
        b = hard | narrow

        # 2) широкое окно (±0.45 декады): ловит группы выбросов в «мёртвой
        # зоне», где узкое окно состоит из одних плохих точек; и возвращает
        # хорошие точки, которые узкое окно забраковало из-за соседства с ними
        def wide(i, ref):
            nb = (np.abs(lf - lf[i]) <= 0.45) & ~ref
            nb[i] = False
            if nb.sum() < 3:
                return None
            return abs(lr[i] - np.median(lr[nb])), abs(phs[i] - np.median(phs[nb]))

        ref = b.copy()
        for i in range(n):
            d = wide(i, ref)
            if d is None:
                continue
            if not b[i] and (d[0] > 2 * jump or d[1] > 30):
                b[i] = True
            elif narrow[i] and not hard[i] and d[0] < jump / 2 and d[1] < 10:
                b[i] = False
        out[c] = b
    return out


def auto_mask(st: Station, max_rel_err=0.5, jump=0.4, comps=("xy", "yx"), keep=True):
    """Автоматическая отбраковка явно плохих частот станции (см. bad_points).

    Частота исключается, если плоха хотя бы одна из компонент comps.
    keep=True — уже исключённые вручную точки остаются исключёнными.
    Возвращает число новых исключённых частот.
    """
    bad = np.zeros(len(st.freq), bool)
    for b in bad_points(st, max_rel_err, jump, comps).values():
        bad |= b
    before = int(st.mask.sum())
    st.mask = (st.mask | bad) if keep else bad
    return int(st.mask.sum()) - before


# ----------------------------------------------------------------------
# Заполнение коротких пропусков вместо исключения точек
# ----------------------------------------------------------------------
def _fill_line(x, y, good, gap, k=3):
    """Прямая через пропуск по k ближайшим хорошим точкам с каждой стороны.

    Взвешенный МНК, одна точка с наибольшей невязкой отбрасывается, если она
    резко выбивается. Значения ограничены диапазоном опорных точек (без
    «вылетов» на крутых участках). Возвращает (значения, разброс опорных точек
    относительно прямой).
    """
    gi = np.flatnonzero(good)
    idx = np.r_[gi[gi < gap[0]][-k:], gi[gi > gap[-1]][:k]]
    xs, ys = x[idx], y[idx]
    c = np.polyfit(xs, ys, 1)
    res = ys - np.polyval(c, xs)
    if len(idx) >= 4:
        j = int(np.argmax(np.abs(res)))
        if abs(res[j]) > 2.5 * np.median(np.abs(res)) + 1e-12:
            keep = np.arange(len(idx)) != j
            xs, ys = xs[keep], ys[keep]
            c = np.polyfit(xs, ys, 1)
            res = ys - np.polyval(c, xs)
    v = np.clip(np.polyval(c, x[gap]), ys.min(), ys.max())
    return v, float(np.sqrt(np.mean(res ** 2)))


def fill_gaps(st: Station, max_gap=0.6, rel_err=0.1, max_scatter=0.15):
    """Заменить исключённые точки внутри коротких пропусков отрезком прямой.

    Для каждой компоненты импеданса lg|Z| (то есть lg ρa) и фаза
    заменяются прямой по lg f, проведённой по 3 ближайшим хорошим точкам с
    каждой стороны (с отсевом одного выброса); если эти точки разбросаны
    относительно прямой больше чем на max_scatter декады по ρa (или
    ~max_scatter·90° по фазе), участок считается шумом и не заполняется. Пропуск заполняется, только если
    между ограничивающими хорошими точками не больше max_gap декады; концы
    кривой (экстраполяция) и длинные пропуски остаются исключёнными.

    Внутри исключённой частоты заменяются только плохие компоненты
    (по bad_points); если частота исключена вручную, — все. Диагональные
    компоненты и типпер заменяются, если заменены обе XY и YX.

    Ошибка восстановленных точек: rel_err от |Z| (0.1 → ρa ±20 %, фаза ±6°),
    но не меньше ошибки соседних хороших точек. Восстановленные частоты
    снимаются с исключения и помечаются в st.filled; отменить — unfill_gaps.
    Возвращает число заполненных частот.
    """
    if getattr(st, "_orig", None) is not None:
        unfill_gaps(st)
    st.sort()
    nf = len(st.freq)
    if nf < 4 or not st.mask.any():
        return 0
    x = np.log10(st.freq)
    orig = (st.Z.copy(), st.Zerr.copy(),
            None if st.T is None else st.T.copy(),
            None if st.Terr is None else st.Terr.copy(), st.mask.copy())
    bad = bad_points(st)
    Z, Zerr = st.Z.copy(), st.Zerr.copy()
    T = None if st.T is None else st.T.copy()
    Terr = None if st.Terr is None else st.Terr.copy()
    mask = st.mask.copy()
    filled = np.zeros(nf, bool)
    good_f = ~st.mask
    i = 0
    while i < nf:
        if not st.mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < nf and st.mask[j + 1]:
            j += 1
        gap = np.arange(i, j + 1)
        i = j + 1
        if gap[0] == 0 or gap[-1] == nf - 1:
            continue  # без экстраполяции
        xl, xr = x[gap[0] - 1], x[gap[-1] + 1]
        if abs(xr - xl) > max_gap:
            continue
        if good_f[:gap[0]].sum() < 1 or good_f[gap[-1] + 1:].sum() < 1:
            continue
        # какие компоненты на каждой частоте пропуска заменять
        rep = {}
        for c in ("xy", "yx"):
            r = bad[c][gap].copy()
            manual = ~(bad["xy"][gap] | bad["yx"][gap])
            rep[c] = r | manual
        both = rep["xy"] & rep["yx"]
        ok = True
        newZ = {}
        for c in ("xx", "xy", "yx", "yy"):
            a, b = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}[c]
            z = st.Z[:, a, b]
            fin = np.isfinite(z) & (np.abs(z) > 0)
            gd = good_f & fin
            if c in ("xx", "yy") and (~fin[gap]).all():
                continue  # диагональных компонент нет
            if not (gd[:gap[0]].any() and gd[gap[-1] + 1:].any()):
                if c in ("xy", "yx"):
                    ok = False
                continue
            la = np.full(nf, np.nan)
            la[fin] = np.log(np.abs(z[fin]))
            ph_g = np.full(nf, np.nan)
            ph_g[gd] = np.unwrap(np.angle(z[gd]))  # unwrap только по хорошим точкам
            lz, slz = _fill_line(x, la, gd, gap)
            pz, spz = _fill_line(x, ph_g, gd, gap)
            if c in ("xy", "yx"):
                # «хорошую» компоненту на исключённой частоте оставляем, только
                # если она согласуется с прямой
                dev = (np.abs(la[gap] - lz) > 0.2 * np.log(10) / 2) | \
                    (np.abs(np.angle(np.exp(1j * (np.angle(z[gap]) - pz)))) > np.deg2rad(10)) | \
                    ~fin[gap]
                rep[c] = rep[c] | dev
                both = rep["xy"] & rep["yx"]
            # опорные точки слишком шумные — пропуск не заполняем
            if c in ("xy", "yx") and (slz > max_scatter * np.log(10) / 2 or
                                      np.rad2deg(spz) > 2 * max_scatter * 45):
                ok = False
                continue
            zn = np.exp(lz + 1j * pz)
            gi = np.flatnonzero(gd)
            nb = np.r_[gi[gi < gap[0]][-2:], gi[gi > gap[-1]][:2]]
            e_nb = np.median(st.Zerr[nb, a, b] / np.abs(z[nb]))
            er = max(rel_err, e_nb if np.isfinite(e_nb) else 0.0) * np.abs(zn)
            newZ[c] = (a, b, zn, er)
        if not ok:
            continue
        for c, (a, b, zn, er) in newZ.items():
            sel = rep[c] if c in ("xy", "yx") else both
            Z[gap[sel], a, b] = zn[sel]
            Zerr[gap[sel], a, b] = er[sel]
        if T is not None and both.any():
            for k in range(2):
                t = st.T[:, k]
                gd = good_f & np.isfinite(t)
                if gd[:gap[0]].any() and gd[gap[-1] + 1:].any():
                    re_ = _fill_line(x, t.real, gd, gap)[0]
                    im_ = _fill_line(x, t.imag, gd, gap)[0]
                    T[gap[both], k] = (re_ + 1j * im_)[both]
                    te = np.nanmedian(st.Terr[gd, k]) if Terr is not None else 0.05
                    if Terr is not None:
                        Terr[gap[both], k] = max(2 * te, 0.05)
        mask[gap] = False
        filled[gap] = True
    n = int(filled.sum())
    if n:
        st.Z, st.Zerr, st.T, st.Terr = Z, Zerr, T, Terr
        st.mask, st.filled = mask, filled
        st._orig = orig
        st.fill_opts = dict(max_gap=max_gap, rel_err=rel_err, max_scatter=max_scatter)
    return n


def unfill_gaps(st: Station):
    """Вернуть исходные значения и исключения после fill_gaps."""
    if getattr(st, "_orig", None) is None:
        return 0
    n = int(st.filled.sum())
    st.Z, st.Zerr, st.T, st.Terr, st.mask = st._orig
    st.filled = np.zeros(len(st.freq), bool)
    st._orig = None
    return n
