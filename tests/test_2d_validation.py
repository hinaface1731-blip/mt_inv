"""Independent limits, mesh checks and cache regression for the 2D solver."""
import numpy as np
import discretize
from simpeg import maps
from simpeg.electromagnetics import natural_source as nsem

from mtinv._simpeg_fast import FastLogTE, FastLogTM
from mtinv.inv1d import rho_phase_1d
from mtinv.inv2d import Inv2DData, forward_2d, terrain_geometry


def test_2d_halfspace_matches_analytical_and_refines():
    freqs = np.array([.1, 10.])
    for h in (500., 250.):
        mesh = discretize.TensorMesh([
            np.r_[h * 1.4 ** np.arange(12, 0, -1), np.full(12, h), h * 1.4 ** np.arange(1, 13)],
            np.r_[h * 1.4 ** np.arange(15, 0, -1), np.full(12, h)]], origin="CN")
        pred = forward_2d(mesh, np.full(mesh.n_cells, 100.), np.array([-500., 500.]), freqs)
        for mode in ("te", "tm"):
            np.testing.assert_allclose(pred[mode]["rho"], 100., rtol=.03)
            np.testing.assert_allclose(pred[mode]["phs"], 45., atol=1.)


def test_2d_layered_earth_matches_independent_1d_recursion():
    hz = np.r_[100. * 1.4 ** np.arange(18, 0, -1), np.full(10, 100.)]
    hx = np.r_[500. * 1.4 ** np.arange(14, 0, -1), np.full(10, 500.),
               500. * 1.4 ** np.arange(1, 15)]
    mesh = discretize.TensorMesh([hx, hz], origin="CN")
    rho = np.where(mesh.cell_centers[:, 1] > -1000., 100., 10.)
    f = np.array([1., 10., 100.])
    ra, ph = rho_phase_1d([100., 10.], [1000.], f)
    pred = forward_2d(mesh, rho, np.array([0.]), f)
    for mode in ("te", "tm"):
        np.testing.assert_allclose(pred[mode]["rho"][:, 0], ra, rtol=.06)
        np.testing.assert_allclose(pred[mode]["phs"][:, 0], ph, atol=2.)


def test_log_jacobian_cache_updates_without_dpred_call():
    mesh = discretize.TensorMesh([np.full(12, 200.), np.full(12, 200.)], origin="CN")
    for Sim, orient in ((FastLogTE, "xy"), (FastLogTM, "yx")):
        def make():
            rx = [nsem.receivers.Impedance(np.array([[0., 0.]]), orientation=orient, component=c)
                  for c in ("apparent_resistivity", "phase")]
            return Sim(mesh, survey=nsem.Survey([nsem.sources.Planewave(rx, frequency=10.)]),
                       sigmaMap=maps.ExpMap(mesh))
        sim = make()
        m = np.full(mesh.n_cells, np.log(.01))
        sim.getJ(m)
        m2 = m + .3 * np.sin(np.arange(mesh.n_cells))
        changed = sim.getJ(m2).copy()
        np.testing.assert_allclose(changed, make().getJ(m2), rtol=1e-10, atol=1e-10)
        rng = np.random.default_rng(2)
        v, w = rng.normal(size=len(m2)), rng.normal(size=sim.survey.nD)
        np.testing.assert_allclose(w @ sim.Jvec(m2, v), v @ sim.Jtvec(m2, w), rtol=1e-12)


def test_staircase_topography_receiver_locations_and_flat_limit():
    mesh = discretize.TensorMesh([np.full(4, 100.), np.full(6, 100.)], origin=[-200., -400.])
    d = Inv2DData(["a", "b"], np.array([-150., 150.]), np.array([1.]), z=np.array([0., -100.]))
    active, rz, err = terrain_geometry(mesh, d)
    assert active.any() and (~active).any()
    assert np.max(np.abs(err)) <= 100.
    assert rz[0] >= rz[1]
    d.z[:] = 0
    flat, rz, _ = terrain_geometry(mesh, d)
    np.testing.assert_array_equal(flat, mesh.cell_centers[:, 1] < 0)
    np.testing.assert_allclose(rz, 0)
