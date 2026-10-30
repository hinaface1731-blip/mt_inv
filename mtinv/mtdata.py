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

    def __post_init__(self):
        self.freq = np.asarray(self.freq, dtype=float)
        if self.mask is None:
            self.mask = np.zeros(len(self.freq), dtype=bool)
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
                       self.mask.copy())

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
