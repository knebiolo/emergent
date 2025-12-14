import io
import h5py
import numpy as np
from types import SimpleNamespace

from emergent.fish_passage.io import compute_alongstream_raster, compute_coarsened_alongstream_raster


def make_simulation_with_env(h=5, w=5):
    bio = io.BytesIO()
    f = h5py.File(bio, mode='w')
    env = f.create_group('environment')
    xs = np.linspace(0.0, 4.0, w)
    ys = np.linspace(4.0, 0.0, h)
    xv, yv = np.meshgrid(xs, ys)
    env.create_dataset('x_coords', data=xv)
    env.create_dataset('y_coords', data=yv)

    # create a simple wetted mask: bottom half wetted
    wett = np.zeros((h, w), dtype=np.uint8)
    wett[2:, :] = 1
    env.create_dataset('wetted', data=wett)

    sim = SimpleNamespace()
    sim.hdf5 = f
    # leave depth_rast_transform None so px=py=1.0
    sim.depth_rast_transform = None
    return sim


def test_compute_alongstream_raster_basic():
    sim = make_simulation_with_env(6, 6)
    arr = compute_alongstream_raster(sim, depth_name='depth', wetted_name='wetted', out_name='asd')
    assert arr.shape == (6, 6)
    env = sim.hdf5['environment']
    # where wetted, values should be finite or zero (outlet)
    wett = np.asarray(env['wetted'])
    mask = (wett != 0)
    assert np.all(np.isfinite(arr[mask]))
    # outlet is minimal among wetted cells
    min_val = np.nanmin(arr[mask])
    assert min_val >= 0.0


def test_compute_coarsened_alongstream_raster_basic():
    sim = make_simulation_with_env(8, 8)
    up = compute_coarsened_alongstream_raster(sim, factor=2, depth_name='depth', wetted_name='wetted', out_name='asd_coarse')
    assert up.shape == (8, 8)
    env = sim.hdf5['environment']
    wett = np.asarray(env['wetted'])
    mask = (wett != 0)
    # coarse sampling can leave some fine cells NaN after upsampling; require majority finite
    finite_count = np.sum(np.isfinite(up[mask]))
    total = np.sum(mask)
    assert finite_count / float(max(1, total)) >= 0.5
