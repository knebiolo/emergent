import tempfile
import os
import numpy as np
import rasterio
from rasterio.transform import Affine
import geopandas as gpd
import pandas as pd

from emergent.salmon_abm import io, hdf5_io


def test_safe_hdf5_open_with_dict():
    store = {}
    with io.safe_hdf5_open(store) as db:
        assert db is store
        db['foo'] = np.array([1, 2, 3])
    assert 'foo' in store


def test_write_sim_initial_and_read_back():
    store = {}
    sim_state = {'num_agents': 4, 'num_timesteps': 10, 'sex': np.array([0,1,0,1])}
    io.write_sim_initial(store, sim_state)
    # datasets should exist in dict-like store
    assert 'agent_data/sex' in store
    assert store['agent_data/sex'].shape == (4,)
    assert 'agent_data/X' in store
    assert store['agent_data/X'].shape == (4, 10)


def test_enviro_load_many(tmp_path):
    # write a small TIF
    out = tmp_path / 'd.tif'
    arr = np.arange(4, dtype=np.float32).reshape((2,2))
    transform = Affine.translation(0, 0)
    with rasterio.open(
        str(out), 'w', driver='GTiff', height=2, width=2, count=1, dtype='float32', transform=transform
    ) as dst:
        dst.write(arr, 1)

    res = io.enviro_load_many({'depth': str(out), 'missing': 'nope.tif'})
    assert 'depth' in res and res['depth'] is not None
    assert res['missing'] is None


def test_longitudinal_chainage():
    from shapely.geometry import LineString
    gdf = gpd.GeoDataFrame({'geometry': [LineString([(0,0),(3,4)])]})
    out = io.longitudinal_chainage(gdf)
    assert 'geometry_length' in out.columns
    assert out.iloc[0]['geometry_length'] == 5.0
    assert len(out.iloc[0]['chainage_coords']) == 2


def test_movie_frames_from_stack_and_trajs():
    stack = np.zeros((3, 4, 4), dtype=float)
    trajs = {0: [(1,1), (2,2), (3,3)]}
    frames = io.movie_frames_from_stack(stack, trajs=trajs)
    assert len(frames) == 3
    # ensure highlight value present
    vals = [np.nanmax(f) for f in frames]
    assert any(v > 0 for v in vals)
