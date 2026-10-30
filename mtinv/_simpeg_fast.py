"""Ускорение 2D МТ в SimPEG: явная матрица чувствительности.

Стандартные Jvec/Jtvec в SimPEG для 2D МТ на каждой CG-итерации заново
строят проекции и производные — это очень медленно. Здесь J собирается один
раз на модель (решения с многими правыми частями) и далее используется
как плотная матрица.
"""

import numpy as np
from simpeg.electromagnetics import natural_source as nsem
from simpeg.utils import Zero


class _StoredJMixin:
    def getJ(self, m, f=None):
        self.model = m
        if getattr(self, "_Jmatrix", None) is not None:
            return self._Jmatrix
        if f is None:
            f = self.fields(m)
        J = np.zeros((self.survey.nD, self.model.size))
        slices = self.survey.get_all_slices()
        for A_i, freq in zip(self.Ainv, self.survey.frequencies):
            for src in self.survey.get_sources_by_frequency(freq):
                u_src = f[src, self._solutionType]
                for rx in src.receiver_list:
                    v = np.eye(rx.nD, dtype=float)
                    df_duT, df_dmT = rx.evalDeriv(src, self.mesh, f, v=v, adjoint=True)
                    X = A_i * np.asarray(df_duT)
                    du = -self.getADeriv(freq, u_src, X, adjoint=True)
                    rhs = self.getRHSDeriv(freq, src, X, adjoint=True)
                    if not isinstance(rhs, Zero):
                        du = du + rhs
                    if not isinstance(df_dmT, Zero) and not (
                            isinstance(df_dmT, (list, tuple)) and isinstance(df_dmT[0], Zero)):
                        dm = np.asarray(df_dmT[0] if isinstance(df_dmT, (list, tuple))
                                        and len(df_dmT) == 1 else df_dmT)
                        if dm.ndim == 1:
                            dm = dm.reshape(du.shape, order="F")
                        elif dm.shape != du.shape:
                            dm = dm.T
                        du = du + dm
                    J[slices[src, rx]] = np.real(np.asarray(du)).T
        self._Jmatrix = J
        return J

    def Jvec(self, m, v, f=None):
        return self.getJ(m, f) @ v

    def Jtvec(self, m, v, f=None):
        return self.getJ(m, f).T @ v

    def Jvec_approx(self, m, v, f=None):
        return self.Jvec(m, v, f)

    def Jtvec_approx(self, m, v, f=None):
        return self.Jtvec(m, v, f)


class _LogRhoMixin:
    """Кажущееся сопротивление в данных — ln(ρa) вместо ρa.

    В линейном масштабе с относительной ошибкой занижение ρa моделью
    штрафуется не больше чем на 1/ошибка σ, а завышение — без ограничений;
    в логарифмах невязка симметрична. Фаза не меняется."""

    def _rho_rows(self):
        if getattr(self, "_rows_cache", None) is None:
            rows = np.zeros(self.survey.nD, bool)
            slices = self.survey.get_all_slices()
            for src in self.survey.source_list:
                for rx in src.receiver_list:
                    if rx.component == "apparent_resistivity":
                        rows[slices[src, rx]] = True
            self._rows_cache = rows
        return self._rows_cache

    def dpred(self, m=None, f=None):
        d = np.array(super().dpred(m, f=f), float)
        rows = self._rho_rows()
        d[rows] = np.log(np.maximum(d[rows], 1e-12))
        return d

    def getJ(self, m, f=None):
        if getattr(self, "_Jmatrix", None) is not None and getattr(self, "_J_is_log", False):
            return self._Jmatrix
        if f is None:
            f = self.fields(m)
        J = super().getJ(m, f)
        d = np.array(nsem.simulation.BaseFDEMSimulation.dpred(self, m, f=f), float)
        rows = self._rho_rows()
        J[rows] /= np.maximum(d[rows], 1e-12)[:, None]
        self._Jmatrix = J
        self._J_is_log = True
        return J

    @property
    def _delete_on_model_update(self):
        return super()._delete_on_model_update + ["_J_is_log"]


class FastTE(_StoredJMixin, nsem.simulation.Simulation2DElectricField):
    pass


class FastTM(_StoredJMixin, nsem.simulation.Simulation2DMagneticField):
    pass


class FastLogTE(_LogRhoMixin, FastTE):
    pass


class FastLogTM(_LogRhoMixin, FastTM):
    pass
