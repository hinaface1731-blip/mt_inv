"""Интерфейс собирается (ловит ошибки импорта и создания вкладок)."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_main_window_builds():
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mtinv.gui.main_window import MainWindow

    w = MainWindow()
    assert w.tabs.count() == 6
    w.close()
    app.processEvents()


def test_curves_fill_buttons():
    import glob

    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mtinv.gui.main_window import MainWindow

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    w = MainWindow()
    w.load_paths(sorted(glob.glob(os.path.join(root, "demo_data", "*.edi")))[:3])
    st = next(iter(w.stations.values()))
    st.mask[10:11] = True
    w.curves.set_station(st)
    w.curves.fill_all()
    assert st.filled[10:11].all() and not st.mask[10:11].any()
    w.curves.unfill_all()
    assert st.mask[10:11].all()
    w.close()
    app.processEvents()


def test_result_roundtrip_preserves_topography_and_profile(tmp_path, monkeypatch):
    import numpy as np
    import discretize
    from PySide6 import QtWidgets
    from mtinv.gui.main_window import MainWindow
    from mtinv.gui.tab2d import load_npz
    from mtinv.inv2d import Inv2DData, Inv2DSettings, Inv2DResult
    from mtinv.profile import Profile

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    mesh = discretize.TensorMesh([np.full(3, 100.), np.full(4, 100.)], origin="CN")
    data = Inv2DData(["a", "b"], np.array([-50., 50.]), np.array([1.]),
                     z=np.array([0., -100.]), elevation_reference=250.)
    profile = Profile("p", ["a", "b"], 70., 95., 70., 96.).to_dict()
    result = Inv2DResult(mesh, np.full(mesh.n_cells, 100.), data, {}, [], 1.,
                         (-100., 100., -300., 0.), Inv2DSettings(),
                         log=["test log"], profile=profile)
    w.tab2d.result = result
    target = tmp_path / "result.npz"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **kw: (str(target), ""))
    w.tab2d.save_result()
    loaded = load_npz(str(target))
    np.testing.assert_allclose(loaded.data.z, data.z)
    assert loaded.data.elevation_reference == 250.
    assert loaded.profile == profile and loaded.log == ["test log"]
    w.close()
    app.processEvents()

