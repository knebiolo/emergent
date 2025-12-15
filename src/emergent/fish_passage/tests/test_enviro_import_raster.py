import pytest
import numpy as np
from pathlib import Path

try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False


@pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio not installed')
def test_enviro_import_from_geotiff(tmp_path):
    from emergent.fish_passage.io import enviro_import

    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tf = from_origin(0.0, 2.0, 1.0, 1.0)
    path = tmp_path / 'small.tif'
    with rasterio.open(
        str(path), 'w', driver='GTiff', height=data.shape[0], width=data.shape[1], count=1, dtype='float32', transform=tf
    ) as dst:
        dst.write(data, 1)

    class S:
        pass

    sim = S()
    sim.hdf5 = __import__('h5py').File(str(tmp_path / 'sim_env.h5'), 'w')

    enviro_import(sim, str(path), 'depth')

    assert 'environment' in sim.hdf5
    env = sim.hdf5['environment']
    assert 'depth' in env
    assert np.allclose(np.asarray(env['depth']), data.astype('f4'))

    sim.hdf5.close()
