import os
import numpy as np
import h5py
from emergent.fish_passage import io as fp_io


class FakeAdapter:
    def __init__(self, plan_path=None):
        self.calls = []
        self.coords = None
        self.values = None
        if plan_path is not None:
            with h5py.File(plan_path, 'r') as ph:
                self.coords = np.asarray(ph['Geometry/Nodes/Coordinates'])
                # Values stored under Results/Results_0001/Cell Hydraulic Depth/Values
                self.values = np.asarray(ph['Results/Results_0001/Cell Hydraulic Depth/Values'])

    def map_idw(self, pts, k=8):
        self.calls.append(('map_idw', self.coords.shape if self.coords is not None else None,
                           self.values.shape if self.values is not None else None, pts.shape, k))
        # return simple nearest value (first column of values)
        d = np.linalg.norm(pts[:, None, :] - self.coords[None, :, :], axis=2)
        idx = np.argmin(d, axis=1)
        return self.values[idx, 0]


class FakeSim:
    def __init__(self):
        self._hecras_maps = {}
        self.env = {}


def make_minimal_plan(tmp_path):
    p = tmp_path / 'plan.h5'
    with h5py.File(p, 'w') as f:
        coords_arr = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        coords = f.create_dataset('Geometry/Nodes/Coordinates', data=coords_arr)
        # also write dataset path expected by ensure_hdf_coords_from_hecras
        f.create_dataset('Geometry/2D Flow Areas/2D area/Cells Center Coordinate', data=coords_arr)
        results = f.create_group('Results')
        res = results.create_group('Results_0001')
        chd = res.create_group('Cell Hydraulic Depth')
        chd.create_dataset('Values', data=np.array([[0.0], [1.0], [0.2]]))
    return str(p)


def test_map_hecras_for_agents_with_fake_adapter(tmp_path):
    plan = make_minimal_plan(tmp_path)
    sim = FakeSim()
    adapter = FakeAdapter(plan)
    # register adapter keyed by (plan_path, ('Values',)) to simulate simulation._hecras_maps
    sim._hecras_maps[(plan, ('Values',))] = adapter

    pts = np.array([[1.0, 0.0], [16.0, 0.0]])
    out = fp_io.map_hecras_for_agents(sim, pts, plan, field_names=['Values'], k=3)
    assert out.shape == (2,)
    # values are nearest from dataset [0.0, 1.0, 0.2] -> expect [0.0, 0.2]
    assert np.allclose(out, np.array([0.0, 0.2]), atol=1e-6)


def test_map_hecras_to_env_rasters_writes_rasters(tmp_path):
    plan = make_minimal_plan(tmp_path)
    sim = FakeSim()
    adapter = FakeAdapter(plan)
    sim._hecras_maps[(plan, ('Values',))] = adapter

    # create a simulation HDF5 file for rasters
    simfile = tmp_path / 'sim.h5'
    sim.hdf5 = h5py.File(str(simfile), 'w')
    # ensure coords are populated for the simulation
    fp_io.ensure_hdf_coords_from_hecras(sim, plan, target_shape=(2, 2))

    fp_io.map_hecras_to_env_rasters(sim, plan, field_names=['Values'], k=3)

    # map_hecras_to_env_rasters should have written entries to sim.hdf5['environment']
    env = sim.hdf5['environment']
    assert 'Values' in env
    arr = np.asarray(env['Values'])
    assert isinstance(arr, np.ndarray)
    assert arr.size == 2 * 2
    # close the sim hdf5
    sim.hdf5.close()
