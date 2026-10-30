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
