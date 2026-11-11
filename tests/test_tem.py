import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtinv.inv1d import rho_phase_1d  # noqa: E402
from mtinv.joint1d import joint_inversion, mt_data, static_shift_from_model  # noqa: E402
from mtinv.mtdata import Station  # noqa: E402
from mtinv.tem import TEMForward, TEMSounding, read_tem, rho_tau  # noqa: E402

RHO = [30.0, 100.0, 3.0, 100.0]
TH = [300.0, 1700.0, 2000.0]


def _station(kx=1.0, ky=1.0):
    f = np.logspace(np.log10(320), -3, 28)
    r, p = rho_phase_1d(RHO, TH, f)
    Z = np.zeros((28, 2, 2), complex)
    amp = np.sqrt(r * 5 * f)
    Z[:, 0, 1] = amp * np.exp(1j * np.deg2rad(p)) * np.sqrt(kx)
    Z[:, 1, 0] = -amp * np.exp(1j * np.deg2rad(p)) * np.sqrt(ky)
    return Station("A", 72, 95, 0, f, Z, np.abs(Z) * 0.03 + 1e-9)


def _sounding():
    t = np.logspace(-5, -2, 25)
    sig = TEMForward(t, 200.0)(RHO, TH)
    return TEMSounding("Z", 72, 95, t, sig, 0.03 * sig, tx_side=200.0)


def test_halfspace_rho_tau_late_time():
    t = np.logspace(-4, -2, 5)
    sig = TEMForward(t, 100.0)([100.0], [])
    assert np.allclose(rho_tau(t, sig, 100.0 / np.sqrt(np.pi))[-2:], 100, rtol=0.02)


def test_read_tem_units(tmp_path):
    p = tmp_path / "z.txt"
    p.write_text("# NAME = Z1\n# TX_SIDE = 100\n# DATA = rho_tau\n# TIME_UNITS = us\n"
                 "время;ρτ\n10;50,5\n20;51\n40;52\n")
    s = read_tem(str(p))
    assert s.name == "Z1" and np.isclose(s.times[0], 1e-5)
    assert np.allclose(s.rho_tau(), [50.5, 51, 52])


def test_static_shift_from_tem_model():
    st = _station(0.4, 1.8)
    snd = _sounding()
    r = joint_inversion(tem=snd)
    doi = snd.depth_of_investigation()
    kx = static_shift_from_model(st, r.tops, r.rho, "xy", max_depth=doi)["k"]
    ky = static_shift_from_model(st, r.tops, r.rho, "yx", max_depth=doi)["k"]
    assert abs(np.log(kx / 2.5)) < 0.15
    assert abs(np.log(ky / (1 / 1.8))) < 0.15
    st.ss = np.array([kx, ky])
    rho, *_ = st.rho_phase("xy")
    rt, _ = rho_phase_1d(RHO, TH, st.freq)
    assert np.allclose(rho, rt, rtol=0.15)


def test_joint_inversion_finds_shift():
    st = _station(0.4, 0.4)
    r = joint_inversion(mt=mt_data(st, "xy"), tem=_sounding(), invert_ss=True, max_iter=15)
    assert abs(np.log(r.ss / 2.5)) < 0.2


def test_loop_forward_matches_empymod():
    """Независимая сверка прямой задачи ЗСБ (петля 100×100, центр и разнос 100 м под 45°,
    ступенька и линейный фронт 20 мкс) с empymod: петля из четырёх электрических линий,
    другой фильтр Ганкеля (wer_201_2018), свой переход во время."""
    import pytest

    empymod = pytest.importorskip("empymod")
    from mtinv.tem import LoopTEMForward

    mu0 = 4e-7 * np.pi
    t = np.logspace(np.log10(1e-4), -2, 12)
    tau = 20e-6
    a = 50.0
    segs = [(-a, a, -a, -a), (a, a, -a, a), (a, -a, a, a), (-a, -a, a, -a)]
    res, thick = [300.0, 20.0, 1000.0], [150.0, 300.0]
    x, w = np.polynomial.legendre.leggauss(12)
    u, wu = 0.5 * (x + 1), 0.5 * w
    for r, ang in ((0.0, 0.0), (100.0, 45.0)):
        rx, ry = r * np.cos(np.radians(ang)), r * np.sin(np.radians(ang))
        tf = np.exp(np.linspace(np.log(t.min()) - 0.3, np.log(t.max() + tau) + 0.3, 500))
        h = 0
        for x0, x1, y0, y1 in segs:
            h = h + np.asarray(empymod.bipole(
                src=[x0, x1, y0, y1, 0.0, 0.0], rec=[rx, ry, 0.0, 0, 90], depth=[0.0, 150.0, 450.0],
                res=[2e14] + res, freqtime=tf, signal=-1, mrec=True, strength=1, srcpts=11,
                ht="dlf", htarg={"dlf": "wer_201_2018"}, ft="dlf", verb=1))
        step = -mu0 * np.gradient(h, np.log(tf)) / tf

        def f(q):
            return np.interp(np.log(q), np.log(tf), step)

        ours = LoopTEMForward(t, 100.0, 100.0, 0.0, r, ang)(np.array(res), np.array(thick))
        assert np.allclose(ours, f(t), rtol=2e-3)
        ref_ramp = np.array([np.sum(wu * f(ti + u * tau)) for ti in t])
        ours_ramp = LoopTEMForward(t, 100.0, 100.0, 0.0, r, ang, tau)(np.array(res), np.array(thick))
        assert np.allclose(ours_ramp, ref_ramp, rtol=2e-3)
