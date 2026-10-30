import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtinv.tem import LoopTEMForward, TEMForward  # noqa: E402
from mtinv.tem_fast import VMDKernel  # noqa: E402
from mtinv.tsikl import build_soundings, parse_name  # noqa: E402


def _write_pod(path, picket, wi, r, times_ms, values):
    lines = ["ТестЭК     Участок", "27.08.2026     Дата", "6              Профиль",
             f"{picket}           Пикет", f"100 50 {r} 0      Ltr Lrc r Angle",
             f"{wi} 100 100   W*I La Lb",
             "215 16480398 8264758.5 92.34484 74.44172 214.9 17:20:58 27.08.2026  Z X Y Долг. Шир. Выс. Время",
             str(len(times_ms))]
    lines += [f"{t:.7g}      {v:.7g}" for t, v in zip(times_ms, values)]
    lines += ["", "19.03          Общая длительность задержки, мкс"]
    with open(path, "wb") as f:
        f.write("\r\n".join(lines).encode("cp1251"))


def test_parse_name_orders():
    assert parse_name("06-220б_1.pod") == ("06", "220", "б", [1])
    assert parse_name("06-225м_2_3.pod") == ("06", "225", "м", [2, 3])
    assert parse_name("01-17б_3_2.pod") == ("01", "17", "б", [3, 2])


def test_build_soundings_merge(tmp_path):
    t = np.logspace(-2, 1, 40)            # мс
    true = 1e3 * t ** -2.5                # условная ЭДС/I (мкВ/А)
    rng = np.random.default_rng(0)
    for cur, wi, sat in (("б", 16.0, 30.0 * 16), ("м", 0.5, 1e9)):
        for dup in (1, 2, 3):
            v = np.minimum(true * wi * (1 + 0.01 * rng.normal(size=t.size)), sat)
            _write_pod(tmp_path / f"06-220{cur}_{dup}.pod", "220" + cur, wi, 0, t, v)
            _write_pod(tmp_path / f"06-225{cur}_2_{dup}.pod", "225" + cur, wi, 100, t, v)
    snds, skipped = build_soundings(sorted(str(p) for p in tmp_path.iterdir()), units="uV",
                                    rx_mode="loop", t_min=None)
    assert not skipped
    names = [s.name for s in snds]
    assert names == ["06-220", "06-225_2"]
    s = snds[0]
    assert s.is_central and snds[1].offset == 100 and s.rx_side == 50
    ok = ~s.mask
    # после сшивки и нормировки: ЭДС/I / площадь рамки 50×50, мкВ -> В
    expect = true * 1e-6 / 2500
    assert np.allclose(s.signal[ok], np.interp(s.times[ok], t * 1e-3, expect), rtol=0.05)
    assert s.raw["info"]["t_cross"] is not None


def test_rho_tau_matches_proba_example():
    """Пример из описания «ПРОБА-WIN»: A=200, lизм=80, I·W=43.3 А,
    t=0.1 мс, ЭДС=3230 мВ -> Rotau=66.76 Ом·м."""
    from mtinv.tem import rho_tau

    sig = 3.230 / (80 * 80) / 43.3
    assert abs(rho_tau([1e-4], [sig], 200 / np.sqrt(np.pi))[0] - 66.76) < 0.05


def test_ramp_array_matches_scalar():
    t = np.logspace(-4.5, -2.5, 6)
    a = LoopTEMForward(t, 100.0, ramp=20e-6)([100.0], [])
    b = LoopTEMForward(t, 100.0, ramp=np.full(6, 20e-6))([100.0], [])
    assert np.allclose(a, b)


def test_fast_kernel_matches_simpeg_loop():
    t = np.logspace(-5, -2.5, 8)
    a = LoopTEMForward(t, 100.0)([30.0, 100.0, 3.0], [300.0, 1700.0])
    b = TEMForward(t, 100.0)([30.0, 100.0, 3.0], [300.0, 1700.0])
    assert np.allclose(a, b, rtol=0.03)   # квадрат vs круг той же площади


def test_fast_kernel_jacobian():
    t = np.logspace(-5, -2, 10)
    K = VMDKernel(5, 200, t)
    rho = np.array([50.0, 200.0, 10.0, 100.0])
    th = np.array([50.0, 200.0, 400.0])
    d, J = K.dbdt_jac(rho, th)
    h = 1e-5
    for j in range(len(rho)):
        r2 = rho.copy()
        r2[j] *= 10 ** h
        fd = (K.dbdt(r2, th) - d) / h
        assert np.allclose(J[:, :, j], fd, rtol=1e-3, atol=1e-6 * np.abs(J).max())


def test_coil_receiver_is_point_with_equivalent_area(tmp_path):
    t = np.logspace(-2, 1, 30)
    v = 1e3 * t ** -2.5
    for dup in (1, 2):
        _write_pod(tmp_path / f"06-220б_{dup}.pod", "220б", 16.0, 0, t, v * 16.0)
    (s,), _ = build_soundings(sorted(str(p) for p in tmp_path.iterdir()), units="uV")
    assert s.rx_side == 0                       # ПДИ — точечный датчик
    ok = ~s.mask
    assert s.times[ok].min() >= 50e-6           # начальное время ПДИ-50
    assert np.allclose(s.signal[ok], np.interp(s.times[ok], t * 1e-3, v * 1e-6 / 2500), rtol=1e-6)


def test_time_shift_between_currents_is_found():
    from mtinv.tsikl import estimate_time_shift

    t = np.logspace(-4.5, -3, 40)
    f = lambda tt: 1e-6 * (tt / 1e-4) ** -2.5          # noqa: E731
    vb = f(t)
    vs = 0.9 * f(t - 10e-6)        # малый ток «опережает» на 10 мкс
    sh = estimate_time_shift(t, vb, vs, 50e-6)
    assert abs(sh - 10e-6) < 1.5e-6
