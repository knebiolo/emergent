import numpy as np
import pytest
from emergent.fish_passage.io import map_hecras_for_agents, HECRASMap
from emergent.fish_passage.tests.fixtures.hdf5_plan_fixture import create_minimal_plan


def test_direct_plan_call(tmp_path):
    plan = create_minimal_plan(tmp_path / 'plan_agents1.h5')
    pts = np.array([[0.0, 0.0], [1.0, 1.0]])
    # Call with explicit field name present in fixture
    out = map_hecras_for_agents(str(plan), pts, field_names=['Fields/depth'])
    # Accept either a dict mapping or an ndarray-like return
    assert isinstance(out, dict) or hasattr(out, '__array__') or isinstance(out, (list, tuple, np.ndarray))


def test_sim_registered_adapter(tmp_path):
    plan = create_minimal_plan(tmp_path / 'plan_agents2.h5')
    sim = type('S', (), {})()
    sim._hecras_maps = {}
    # register adapter keyed by (plan_path, fields)
    adapter = HECRASMap(str(plan), field_names=['Fields/depth'])
    key = (str(plan), tuple(['Fields/depth']))
    sim._hecras_maps[key] = adapter

    pts = np.array([[0.0, 0.0]])
    res = map_hecras_for_agents(sim, pts, plan_path=str(plan), field_names=['Fields/depth'])
    # adapter.map_idw returns an ndarray
    assert hasattr(res, '__len__')


def test_raises_when_no_adapter(tmp_path):
    plan = create_minimal_plan(tmp_path / 'plan_agents3.h5')
    sim = type('S', (), {})()
    sim._hecras_maps = {}
    pts = np.array([[0.0, 0.0]])
    with pytest.raises(KeyError):
        map_hecras_for_agents(sim, pts, plan_path=str(plan), field_names=['depth'])
