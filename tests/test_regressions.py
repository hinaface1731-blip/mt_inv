"""Regressions runnable with unittest as well as pytest (NumPy only)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mtinv.edi import read_edi, write_edi
from mtinv.inv1d import rho_phase_1d, occam1d, layer_mesh
from mtinv.inv2d import Inv2DSettings, prepare_data
from mtinv.mtdata import Station
from mtinv.profile import Profile


def station(full=True):
    z = np.array([[[0.2 + 0.1j, 2 + 3j], [-3 - 4j, 0.4 + 0.2j]],
                  [[0.3 + 0.2j, 4 + 5j], [-5 - 6j, 0.5 + 0.3j]]])
    if not full:
        z[:, 0, 0] = np.nan
        z[:, 1, 1] = np.nan
    return Station("s", 70.0, 95.0, 100.0, np.array([10., 1.]), z,
                   np.full(z.shape, 0.1), np.array([[.1 + .2j, .3 + .4j]] * 2),
                   np.full((2, 2), .01))


class EDIRegressions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "station.edi"

    def tearDown(self):
        self.tmp.cleanup()

    def test_zero_rotation_preserves_missing_diagonals(self):
        st = station(False)
        rot = st.rotated(0)
        np.testing.assert_allclose(rot.Z, st.Z, equal_nan=True)
        np.testing.assert_allclose(rot.Zerr, st.Zerr)

    def test_right_angle_preserves_offdiagonals(self):
        st = station(False)
        rot = st.rotated(90)
        np.testing.assert_allclose(rot.Z[:, 0, 1], -st.Z[:, 1, 0])
        np.testing.assert_allclose(rot.Z[:, 1, 0], -st.Z[:, 0, 1])

    def test_oblique_rotation_does_not_invent_missing_tensor(self):
        rot = station(False).rotated(30)
        self.assertFalse(np.isfinite(rot.Z).any())

    def test_edi_rotation_roundtrip(self):
        st = station()
        write_edi(st.rotated(37), str(self.path))
        read = read_edi(str(self.path))
        np.testing.assert_allclose(read.Z, st.Z, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(read.T, st.T, rtol=1e-6, atol=1e-6)
        self.assertEqual(read.zrot, 0)

    def test_frequency_dependent_z_and_t_rotation(self):
        st = station()
        rot = st.rotated(30)
        rot.Z[1] = st.rotated(60).Z[1]
        rot.T[1] = st.rotated(60).T[1]
        write_edi(rot, str(self.path))
        text = self.path.read_text(encoding="utf-8")
        text = text.replace(" 3.000000E+01  3.000000E+01", " 3.000000E+01  6.000000E+01")
        self.path.write_text(text, encoding="utf-8")
        read = read_edi(str(self.path))
        np.testing.assert_allclose(read.Z, st.Z, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(read.T, st.T, rtol=1e-6, atol=1e-6)

    def test_static_shift_written_exactly_once(self):
        st = station()
        st.ss = np.array([4., .25])
        write_edi(st, str(self.path))
        read = read_edi(str(self.path))
        np.testing.assert_allclose(read.rho_phase("xy")[0], st.rho_phase("xy")[0], rtol=1e-6)
        np.testing.assert_allclose(read.rho_phase("yx")[0], st.rho_phase("yx")[0], rtol=1e-6)

    def test_scalar_rotation_applies_to_all_frequencies(self):
        st = station()
        write_edi(st.rotated(30), str(self.path))
        text = self.path.read_text(encoding="utf-8")
        text = text.replace(" 3.000000E+01  3.000000E+01", " 3.000000E+01")
        self.path.write_text(text, encoding="utf-8")
        np.testing.assert_allclose(read_edi(str(self.path)).Z, st.Z, rtol=1e-6, atol=1e-6)

    def test_explicit_missing_rotation_reference_rejected(self):
        write_edi(station(), str(self.path))
        text = self.path.read_text(encoding="utf-8").replace("ROT=ZROT", "ROT=MISSING")
        self.path.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing rotation block"):
            read_edi(str(self.path))

    def test_invalid_and_duplicate_frequencies_rejected(self):
        st = station()
        for freq in ([0., 1.], [1., 1.], [-1., 1.]):
            st.freq = np.array(freq)
            write_edi(st, str(self.path))
            with self.assertRaises(ValueError):
                read_edi(str(self.path))

    def test_halfspace_analytical_solution(self):
        freq = np.logspace(-4, 4, 30)
        for rho in (1., 100., 10000.):
            ra, ph = rho_phase_1d([rho], [], freq)
            np.testing.assert_allclose(ra, rho, rtol=1e-12)
            np.testing.assert_allclose(ph, 45., atol=1e-12)

    def test_invalid_1d_inputs(self):
        for args in (([100., 10.], [], [1.]), ([0.], [], [1.]), ([100.], [], [0.])):
            with self.assertRaises(ValueError):
                rho_phase_1d(*args)
        with self.assertRaises(ValueError):
            layer_mesh(z_first=100, z_max=10)
        with self.assertRaises(ValueError):
            occam1d([1., 10.], [100., 100.], [45., 45.], use_rho=False, use_phase=False)

    def test_resistivity_only_fit_keeps_points_without_phase(self):
        result = occam1d([1., 10.], [100., 100.], [np.nan, np.nan],
                         use_phase=False, n_layers=10, max_iter=2)
        self.assertLess(result.rms, 1e-5)

    def test_incomplete_tensor_works_in_2d_without_rotation(self):
        st = station(False)
        p = Profile.fit("p", [st])
        d = prepare_data([st], p, Inv2DSettings(strike=0))
        self.assertTrue(np.isfinite(d.obs["te"]["rho"]).any())
        self.assertTrue(np.isfinite(d.obs["tm"]["rho"]).any())

    def test_duplicate_stations_are_not_moved_silently(self):
        a, b = station(), station()
        b.name = "b"
        with self.assertRaisesRegex(ValueError, "duplicates"):
            prepare_data([a, b], Profile.fit("p", [a, b]), Inv2DSettings())


if __name__ == "__main__":
    unittest.main()
