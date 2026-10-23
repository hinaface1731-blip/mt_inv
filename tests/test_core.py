import os
import sys
import textwrap

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtinv.edi import read_edi, write_edi  # noqa: E402
from mtinv.inv1d import occam1d, rho_phase_1d  # noqa: E402
from mtinv.mtdata import Station  # noqa: E402
from mtinv.profile import Profile  # noqa: E402

EDI_Z_TIPPER = textwrap.dedent("""\
    >HEAD
      DATAID="S01"
      ACQBY="test"
      LAT=-71:30:00
      LONG=101.25
      ELEV=120

    >INFO MAXLINES=1000
      какой-то текст LAT=0 не должен мешать

    >=DEFINEMEAS
      MAXCHAN=7
      REFLAT=-71:30:00
      REFLONG=101:15:00

    >HMEAS ID=1001.001 CHTYPE=HX X=0.0 Y=0.0 AZM=0.0
    >EMEAS ID=1004.001 CHTYPE=EX X=-50 Y=0 X2=50 Y2=0

    >=MTSECT
      SECTID="S01"
      NFREQ=3

    >FREQ NFREQ=3 ORDER=DEC //3
      100.0 10.0 1.0
    >ZROT //3
      0 0 0
    >ZXXR ROT=ZROT //3
      0.1 0.1 0.1
    >ZXXI ROT=ZROT //3
      0.1 0.1 0.1
    >ZXX.VAR ROT=ZROT //3
      0.01 0.01 0.01
    >ZXYR ROT=ZROT //3
      10.0 3.0 1.0
    >ZXYI ROT=ZROT //3
      10.0 3.0 1.0
    >ZXY.VAR ROT=ZROT //3
      1.0 0.09 0.01
    >ZYXR ROT=ZROT //3
      -10.0 -3.0 -1.0
    >ZYXI ROT=ZROT //3
      -10.0 -3.0 -1.0
    >ZYX.VAR ROT=ZROT //3
      1.0 0.09 0.01
    >ZYYR ROT=ZROT //3
      0 0 1.0E+32
    >ZYYI ROT=ZROT //3
      0 0 1.0E+32
    >TROT.EXP //3
      0 0 0
    >TXR.EXP //3
      0.1 0.2 0.3
    >TXI.EXP //3
      0.0 0.1 0.0
    >TXVAR.EXP //3
      0.0001 0.0001 0.0001
    >TYR.EXP //3
      -0.1 -0.2 -0.3
    >TYI.EXP //3
      0.0 0.0 0.0
    >TYVAR.EXP //3
      0.0001 0.0001 0.0001
    >END
""")

EDI_RHO = textwrap.dedent("""\
    >HEAD
      DATAID=S02
      LAT=70.5
      LONG=95.0
    >=MTSECT
    >FREQ //2
      1.0 0.1
    >RHOXY //2
      100 100
    >PHSXY //2
      45 45
    >RHOXY.ERR //2
      10 10
    >RHOYX //2
      100 100
    >PHSYX //2
      -135 -135
    >END
""")


def test_read_z_tipper(tmp_path):
    p = tmp_path / "a.edi"
    p.write_text(EDI_Z_TIPPER)
    st = read_edi(str(p))
    assert st.name == "S01"
    assert np.isclose(st.lat, -71.5) and np.isclose(st.lon, 101.25)
    assert np.allclose(st.freq, [100, 10, 1])
    assert np.isnan(st.Z[2, 1, 1])
    assert st.has_tipper and np.isclose(st.T[1, 0], 0.2 + 0.1j)
    rho, phs, *_ = st.rho_phase("xy")
    assert np.allclose(phs, 45)
    assert np.isclose(rho[0], 0.2 / 100 * 200)
    _, phy, *_ = st.rho_phase("yx")
    assert np.allclose(phy, 45)  # свёрнута в 1-й квадрант
    # запись/чтение
    q = tmp_path / "b.edi"
    write_edi(st, str(q))
    st2 = read_edi(str(q))
    assert np.allclose(st2.Z[:, 0, 1], st.Z[:, 0, 1])
    assert np.isclose(st2.lat, st.lat, atol=1e-5)


def test_read_rho_phase(tmp_path):
    p = tmp_path / "c.edi"
    p.write_text(EDI_RHO)
    st = read_edi(str(p))
    rho, phs, erho, _ = st.rho_phase("xy")
    assert np.allclose(rho, 100) and np.allclose(phs, 45)
    assert np.allclose(erho, 10)
    r2, p2, *_ = st.rho_phase("det")
    assert np.allclose(r2, 100) and np.allclose(p2, 45)


def test_rotation_roundtrip():
    f = np.array([1.0, 10.0])
    Z = np.zeros((2, 2, 2), complex)
    Z[:, 0, 1] = 1 + 1j
    Z[:, 1, 0] = -2 - 2j
    st = Station("x", 70, 90, 0, f, Z, np.abs(Z) * 0.05 + 1e-3)
    r = st.rotated(30).rotated(-30)
    assert np.allclose(r.Z, st.Z)
    r90 = st.rotated(90)
    assert np.allclose(r90.Z[:, 0, 1], -st.Z[:, 1, 0])
    pt = st.phase_tensor()
    assert np.allclose(pt["beta"], 0, atol=1e-9)


def test_occam_recovers_layer():
    f = np.logspace(-3, 3, 40)
    r, p = rho_phase_1d([100, 10, 1000], [1000, 2000], f)
    res = occam1d(f, r, p, n_layers=40)
    assert res.rms < 1.2
    z = 0.5 * (res.tops[:-1] + res.tops[1:])
    k = np.argmin(np.abs(z - 2000))
    assert res.rho[k] < 40


def test_profile_projection():
    sts = [Station(f"s{i}", 72.0, 95.0 + 0.1 * i, 0, np.array([1.0]),
                   np.ones((1, 2, 2), complex), np.ones((1, 2, 2))) for i in range(5)]
    p = Profile.fit("p", sts)
    assert abs(p.azimuth - 90) < 1
    a, off = p.project(np.array([s.lat for s in sts]), np.array([s.lon for s in sts]))
    assert np.all(np.diff(a) > 0)
    assert np.allclose(off, 0, atol=50)
