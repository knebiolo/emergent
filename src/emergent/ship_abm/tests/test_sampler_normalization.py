import numpy as np
import numpy as np
import types
from datetime import datetime

import emergent.ship_abm.ofs_loader as ofs_loader
from emergent.ship_abm.simulation_core import simulation


def test_normalize_various_return_shapes(monkeypatch):
    # Create a simulation instance with a small number of agents and no ENC load
    sim = simulation(port_name='Baltimore', dt=1.0, T=10.0, n_agents=1, load_enc=False)

    # Define raw samplers with different return shapes
    def raw_N2(lons, lats, when):
        lons = np.atleast_1d(lons)
        return np.column_stack((np.ones_like(lons) * 1.0, np.ones_like(lons) * 2.0))

    def raw_2N(lons, lats, when):
        lons = np.atleast_1d(lons)
        return np.vstack((np.ones_like(lons) * 1.0, np.ones_like(lons) * 2.0))

    def raw_flat(lons, lats, when):
        lons = np.atleast_1d(lons)
        N = lons.size
        return np.repeat(np.array([3.0, 4.0]), N)

    def raw_pair(lons, lats, when):
        return np.array([5.0, 6.0])

    # Monkeypatch ofs_loader to return our raw samplers for current and wind
    monkeypatch.setattr(ofs_loader, 'get_current_fn', lambda port_name, start=None: raw_N2)
    monkeypatch.setattr(ofs_loader, 'get_wind_fn', lambda port_name, start=None: raw_N2)

    # Load environmental forcing which will wrap and normalize the samplers
    sim.load_environmental_forcing(start=datetime.utcnow())

    # Query with several points and assert shape (N,2)
    lons = np.array([0.0, 1.0, 2.0])
    lats = np.array([0.0, 0.0, 0.0])
    out = sim.current_fn(lons, lats, datetime.utcnow())
    assert out.shape == (3, 2)
    assert np.allclose(out[:, 0], 1.0)
    assert np.allclose(out[:, 1], 2.0)

    # Now directly test other raw shapes by monkeypatching get_current_fn and reloading
    monkeypatch.setattr(ofs_loader, 'get_current_fn', lambda port_name, start=None: raw_2N)
    sim.load_environmental_forcing(start=datetime.utcnow())
    out2 = sim.current_fn(lons, lats, datetime.utcnow())
    assert out2.shape == (3, 2)
    assert np.allclose(out2[:, 0], 1.0)
    assert np.allclose(out2[:, 1], 2.0)

    monkeypatch.setattr(ofs_loader, 'get_current_fn', lambda port_name, start=None: raw_flat)
    sim.load_environmental_forcing(start=datetime.utcnow())
    out3 = sim.current_fn(lons, lats, datetime.utcnow())
    assert out3.shape == (3, 2)
    assert np.allclose(out3[:, 0], 3.0)
    assert np.allclose(out3[:, 1], 4.0)

    monkeypatch.setattr(ofs_loader, 'get_current_fn', lambda port_name, start=None: raw_pair)
    sim.load_environmental_forcing(start=datetime.utcnow())
    out4 = sim.current_fn(lons, lats, datetime.utcnow())
    assert out4.shape == (3, 2)
    assert np.allclose(out4[0], [5.0, 6.0])


def test_kdtree_resample_fallback(monkeypatch):
    # Create simulation (no ENC) and monkeypatch ofs_loader.get_current_fn to return a sampler
    sim = simulation(port_name='Baltimore', dt=1.0, T=10.0, n_agents=1, load_enc=False)

    # Build a raw sampler that *raises* when called but exposes native unstructured points
    def raw_fail(lons, lats, when):
        raise RuntimeError("no direct query")

    # Attach metadata for KDTree resampling (unstructured)
    native_pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    uvals = np.array([10.0, 20.0, 30.0])
    vvals = np.array([1.0, 2.0, 3.0])
    raw_fail._native = 'unstructured'
    raw_fail._valid_pts = native_pts
    raw_fail._u_valid = uvals
    raw_fail._v_valid = vvals

    monkeypatch.setattr(ofs_loader, 'get_current_fn', lambda port_name, start=None: raw_fail)
    monkeypatch.setattr(ofs_loader, 'get_wind_fn', lambda port_name, start=None: raw_fail)

    sim.load_environmental_forcing(start=datetime.utcnow())

    # Query points that are close to the native_pts above
    qlons = np.array([0.01, 0.99, 0.01])
    qlats = np.array([0.01, 0.01, 0.99])
    out = sim.current_fn(qlons, qlats, datetime.utcnow())
    assert out.shape == (3, 2)
    # Nearest mapping should return the corresponding native u/v pairs
    assert np.allclose(out[0], [10.0, 1.0])
    assert np.allclose(out[1], [20.0, 2.0])
    assert np.allclose(out[2], [30.0, 3.0])
