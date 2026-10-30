"""Быстрая 1D прямая задача ЗСБ: вертикальный магнитный диполь на поверхности.

dBz/dt (ступенчатое выключение) для набора разносов r и времён t.
  * частотная область: Hz_втор(ω, r) = 1/(4π) ∫ r_TE(λ, ω) λ² J0(λr) dλ,
    r_TE — рекурсия по слоям (квазистатика, e^{iωt});
  * интеграл Ганкеля — цифровой фильтр Key (201 точка); разносы берутся на
    сетке с шагом фильтра («лаговая свёртка»), поэтому r_TE считается всего
    для (201 + число разносов) значений λ;
  * переход во время — синус-фильтр Key (81 точка), Im Hz интерполируется
    сплайном по log ω.
Проверено по SimPEG (Simulation1DLayered) — см. tests/test_tem.py.
"""

from __future__ import annotations

import numpy as np

MU0 = 4e-7 * np.pi


class VMDKernel:
    def __init__(self, r_min, r_max, times, n_per_decade=12):
        import libdlf

        base, j0, _ = libdlf.hankel.key_201_2009()
        dlt = float(np.log(base[1] / base[0]))
        r_min = max(float(r_min), 0.1)
        n_r = int(np.ceil(np.log(max(r_max, r_min * 1.01) / r_min) / dlt)) + 1
        self.r = r_min * np.exp(dlt * np.arange(n_r))
        # λ_jk = base_k / r_j = base_{k-j} / r_min  ->  общий набор λ
        m = np.arange(-(n_r - 1), len(base))
        self.lam = base[0] / r_min * np.exp(dlt * m)
        W = np.zeros((n_r, len(self.lam)))
        for j in range(n_r):
            idx = np.arange(len(base)) - j + (n_r - 1)
            W[j, idx] = self.lam[idx] ** 2 * j0 / (4 * np.pi * self.r[j])
        self.W = W

        self.times = np.asarray(times, float)
        from scipy.interpolate import CubicSpline

        fb, fsin, _ = libdlf.fourier.key_81_2009()
        wmin = fb.min() / self.times.max()
        wmax = fb.max() / self.times.min()
        nw = int(np.ceil(np.log10(wmax / wmin) * n_per_decade)) + 1
        self.omega = np.logspace(np.log10(wmin), np.log10(wmax), nw)
        # сплайн по log ω и синус-фильтр линейны -> матрица A: Im H(ω) -> dB/dt(t)
        w_eval = fb[None, :] / self.times[:, None]               # (nt, nf)
        cs = CubicSpline(np.log(self.omega), np.eye(nw), axis=0)
        M = cs(np.log(w_eval))                                   # (nt, nf, nω)
        self.A = -2.0 * MU0 / np.pi * np.einsum("tfw,f->tw", M, fsin) / self.times[:, None]

    def _recursion(self, sig, thick, with_jac):
        lam = self.lam[None, :]
        lam2 = lam ** 2
        iwm = 1j * self.omega[:, None] * MU0
        n = len(sig)
        G = np.sqrt(lam2 + iwm * sig[-1])
        dG = None
        if with_jac:
            dG = np.zeros((n,) + G.shape, complex)   # dG/dσ_j
            dG[-1] = iwm / (2 * G)
        for k in range(n - 2, -1, -1):
            g = np.sqrt(lam2 + iwm * sig[k])
            T = np.tanh(g * thick[k])
            num = G + g * T
            den = g + G * T
            Gn = g * num / den
            if with_jac:
                dGn_dG = g * g * (1 - T * T) / den ** 2
                dT = thick[k] * (1 - T * T)
                dGn_dg = num / den + g * ((T + g * dT) * den - num * (1 + G * dT)) / den ** 2
                dG[k + 1:] *= dGn_dG[None]
                dG[k] = dGn_dg * iwm / (2 * g)
            G = Gn
        rte = (lam - G) / (lam + G)
        drte = None
        if with_jac:
            drte = dG * (-2 * lam / (lam + G) ** 2)[None]
        return rte, drte

    def hz_freq(self, rho, thick):
        """Вторичное Hz(ω, r) для единичного момента: (nω, nr)."""
        rte, _ = self._recursion(1.0 / np.asarray(rho, float), np.asarray(thick, float), False)
        return rte @ self.W.T

    def dbdt(self, rho, thick):
        """dBz/dt при выключении единичного момента: (nt, nr)."""
        return self.A @ self.hz_freq(rho, thick).imag

    def dbdt_jac(self, rho, thick):
        """dB/dt (nt, nr) и производные по log10 ρ слоёв (nt, nr, nlay)."""
        sig = 1.0 / np.asarray(rho, float)
        rte, drte = self._recursion(sig, np.asarray(thick, float), True)
        H = rte @ self.W.T
        dH = np.einsum("jwl,rl->wrj", drte, self.W)              # (nω, nr, nlay)
        d = self.A @ H.imag
        J = np.einsum("tw,wrj->trj", self.A, dH.imag) * (-sig * np.log(10))[None, None, :]
        return d, J
